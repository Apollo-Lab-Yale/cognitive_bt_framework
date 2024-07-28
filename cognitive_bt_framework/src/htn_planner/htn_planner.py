from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI
from cognitive_bt_framework.utils.htn_db_utils import setup_database
from cognitive_bt_framework.utils.logic_utils import cosine_similarity
from cachetools import LRUCache

import sqlite3
import os
from cachetools import LRUCache
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch

class HTNPlanner:
    def __init__(self, embedding_fn, llm_interface=None, db_path='./behavior_tree.db'):
        self.llm_interface = llm_interface
        self.get_embedding = embedding_fn
        self.db_path = db_path
        self.cache = LRUCache(maxsize=100)  # Adjust maxsize based on expected workload
        if not os.path.exists(self.db_path):
            setup_database(self.db_path)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embedding_model = AutoModel.from_pretrained("bert-base-uncased")
        self.task_embeddings = {}
        self.similarity_threshold = 0.0

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def get_nl_embedding(self, text):
        inputs = self.embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.embedding_model(**inputs)
        return outputs.pooler_output[0]

    def find_closest_task(self, input_text):
        input_embedding = self.get_nl_embedding(input_text)
        similarities = {task: torch.cosine_similarity(input_embedding, emb, dim=0).item() for task, emb in
                        self.task_embeddings.items()}
        print(similarities)
        if len(list(similarities.keys())) == 0:
            return None
        best_match = max(similarities, key=similarities.get)
        print(best_match)
        if self.similarity_threshold > similarities[best_match]:
            return None
        return best_match

    def decompose_task(self, task_name, task_embedding):
        with self.connect_db() as conn:
            cursor = conn.cursor()

            # Check if the task exists and get its ID
            cursor.execute("SELECT TaskID FROM Tasks WHERE TaskName = ?", (task_name,))
            task_id_row = cursor.fetchone()

            if task_id_row is None:
                # If the task doesn't exist, insert it into Tasks table first
                cursor.execute("INSERT INTO Tasks (TaskName) VALUES (?)", (task_name,))
                task_id = cursor.lastrowid  # Get the ID of the newly inserted task
            else:
                task_id = task_id_row[0]

            # Now, proceed to handle the decomposition with the valid TaskID
            cursor.execute(
                "SELECT DecompositionText FROM Decompositions WHERE TaskID = ?",
                (task_id,))
            row = cursor.fetchone()

            if row:
                decomposition = row[0]
            else:
                if self.llm_interface:
                    decomposition_text = self.llm_interface.get_task_decomposition(task_name)
                    decomposition = decomposition_text.split('\n')

                    # Insert the new decomposition now that you have a valid TaskID
                    cursor.execute(
                        "INSERT INTO Decompositions (TaskID, DecompositionText, CreationDate, CreatedBy) VALUES (?, ?, ?, 'system')",
                        (task_id, decomposition_text, datetime.now()))

            self.cache[task_name] = decomposition  # Cache the decomposition
            conn.commit()
        return decomposition

    def store_feedback(self, task_name, user_id, feedback):
        """Store user feedback for a given task's decomposition."""
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Feedback (DecompositionID, UserID, FeedbackText, FeedbackDate) VALUES ((SELECT DecompositionID FROM Decompositions JOIN Tasks ON Decompositions.TaskID = Tasks.TaskID WHERE TaskName = ?), ?, ?, ?)",
                (task_name, user_id, feedback, datetime.now()))

    def get_feedback(self, task_name):
        """Retrieve all feedback for a given task."""
        feedback_list = []
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT FeedbackText FROM Feedback JOIN Decompositions ON Feedback.DecompositionID = Decompositions.DecompositionID JOIN Tasks ON Decompositions.TaskID = Tasks.TaskID WHERE TaskName = ?",
                (task_name,))
            feedback_list = [fb[0] for fb in cursor.fetchall()]
        return feedback_list

    def refine_decomposition_with_feedback(self, task_name, user_id, user_feedback):
        with self.connect_db() as conn:
            cursor = conn.cursor()

            # Fetch the latest DecompositionID and text for this task
            cursor.execute('''
                SELECT Decompositions.DecompositionID, Decompositions.DecompositionText
                FROM Decompositions
                INNER JOIN Tasks ON Decompositions.TaskID = Tasks.TaskID
                WHERE Tasks.TaskName = ?
                ORDER BY Decompositions.CreationDate DESC
                LIMIT 1
            ''', (task_name,))
            result = cursor.fetchone()

            if not result:
                print("Original decomposition not found.")
                return None

            decomposition_id, original_decomposition = result
            refined_decomposition = self.llm_interface.query_llm_for_refinement(task_name, original_decomposition,
                                                                                user_feedback)

            if refined_decomposition:
                # Update the cache with the new decomposition
                self.cache[task_name] = refined_decomposition

                # Insert the refined decomposition into the Decompositions table
                cursor.execute('''
                    INSERT INTO Decompositions (TaskID, DecompositionText, CreationDate, CreatedBy)
                    VALUES ((SELECT TaskID FROM Tasks WHERE TaskName = ?), ?, datetime('now'), 'system')
                ''', (task_name, refined_decomposition))

                new_decomposition_id = cursor.lastrowid

                # Store the adjustment reason as user feedback in the DecompositionAdjustments table
                cursor.execute('''
                    INSERT INTO DecompositionAdjustments (DecompositionID, PreviousDecompositionText, NewDecompositionText, AdjustmentReason, AdjustmentDate)
                    VALUES (?, ?, ?, ?, datetime('now'))
                ''', (new_decomposition_id, original_decomposition, refined_decomposition, user_feedback))

                # Optionally, log the feedback in the Feedback table
                cursor.execute('''
                    INSERT INTO Feedback (DecompositionID, UserID, FeedbackText, FeedbackDate)
                    VALUES (?, ?, ?, datetime('now'))
                ''', (decomposition_id, user_id, user_feedback))

                conn.commit()
                print("Refined Decomposition:", refined_decomposition)
                return refined_decomposition
            else:
                print("Unable to refine decomposition based on feedback.")
                return original_decomposition

# Example usage
if __name__ == "__main__":
    # Assuming `llm_interface` is an instance of `LLMInterface` set up as previously described
    llm_interface = LLMInterface()
    planner = HTNPlanner(llm_interface)
    # Decompose a task using the LLM (assuming the LLM interface can handle this task)
    # print(planner.llm_interface.get_task_id('fill a cup with water from the sink'))
    # print(planner.llm_interface.get_task_id('get a cup of water'))
    # print()
    print(planner.decompose_task("CleanKitchen"))

