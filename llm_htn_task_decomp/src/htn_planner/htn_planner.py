from llm_htn_task_decomp.src.llm_interface.llm_interface_openai import LLMInterface
from llm_htn_task_decomp.utils.db_utils import setup_database
from cachetools import LRUCache

import sqlite3
import os
from cachetools import LRUCache
from datetime import datetime


class HTNPlanner:
    def __init__(self, llm_interface=None, db_path='./task_decomposition.db'):
        self.llm_interface = llm_interface
        self.db_path = db_path
        self.cache = LRUCache(maxsize=100)  # Adjust maxsize based on expected workload
        if not os.path.exists(self.db_path):
            setup_database(self.db_path)

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def decompose_task(self, task_name):
        if task_name in self.cache:
            return self.cache[task_name]

        decomposition = None
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DecompositionText FROM Decompositions JOIN Tasks ON Decompositions.TaskID = Tasks.TaskID WHERE TaskName = ?",
                (task_name,))
            row = cursor.fetchone()
            if row:
                decomposition = row[0]
            else:
                if self.llm_interface:
                    decomposition = self.llm_interface.get_task_decomposition(task_name)
                    # Assuming a task has already been added to Tasks table
                    cursor.execute(
                        "INSERT INTO Decompositions (TaskID, DecompositionText, CreationDate, CreatedBy) VALUES ((SELECT TaskID FROM Tasks WHERE TaskName = ?), ?, ?, 'system')",
                        (task_name, decomposition, datetime.now()))

            self.cache[task_name] = decomposition  # Cache the decomposition
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
    print(planner.decompose_task("plan a team building event"))
