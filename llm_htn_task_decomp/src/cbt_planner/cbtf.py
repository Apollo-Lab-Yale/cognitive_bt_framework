import sqlite3
import os
import numpy as np
from cachetools import LRUCache
from transformers import AutoTokenizer, AutoModel
import torch

from llm_htn_task_decomp.src.llm_interface.llm_interface_openai import LLMInterface
from llm_htn_task_decomp.utils.bt_db_utils import setup_database, add_behavior_tree, store_feedback
from llm_htn_task_decomp.utils.bt_utils import parse_node, parse_bt_xml, Node, Selector, Sequence, Action, Condition, BASE_EXAMPLE
from llm_htn_task_decomp.src.sim.ai2_thor.utils import AI2THOR_ACTIONS, AI2THOR_PREDICATES
from llm_htn_task_decomp.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv

class CognitiveBehaviorTreeFramework:
    def __init__(self, robot_interface, actions=AI2THOR_ACTIONS, conditions=AI2THOR_PREDICATES, db_path='./behavior_tree.db', model_name="gpt-3.5-turbo", sim=True):
        self.robot_interface = robot_interface
        self.db_path = db_path
        self.llm_interface = LLMInterface(model_name)
        self.cache = LRUCache(maxsize=100)
        self.actions = actions
        if not os.path.exists(self.db_path):
            setup_database(self.db_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.cosine_similarity_threshold = 0.85  # Cosine similarity threshold
        self.example = BASE_EXAMPLE
        self.conditions = conditions

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def find_most_similar_task(self, embedding, task_name):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TaskID, TaskName FROM Tasks")
            best_match = None
            highest_similarity = 0
            for row in cursor.fetchall():
                try:
                    existing_task_name = row[1]
                    existing_embedding = np.frombuffer(row[0], dtype=np.float32)
                    similarity = np.dot(embedding, existing_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(existing_embedding))
                    if task_name == existing_task_name or (similarity > highest_similarity and similarity > self.cosine_similarity_threshold):
                        best_match = row[0]
                        highest_similarity = similarity
                except Exception as e:
                    pass
            return best_match

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def load_or_generate_bt(self, task_id, task_name):
        bt_xml = None
        if task_id:
            bt_xml = self.load_behavior_tree(task_id)
        else:
            bt_xml = self.llm_interface.get_behavior_tree(task_name, self.actions, self.conditions, self.example)
            self.save_behavior_tree(task_name, bt_xml, task_id)
        return parse_bt_xml(bt_xml), bt_xml

    def load_behavior_tree(self, task_id):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT BehaviorTreeXML FROM BehaviorTrees WHERE TaskID = ?", (task_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def save_behavior_tree(self, task_name, bt_xml, task_id):
        with self.connect_db() as conn:
            if not task_id:
                cursor = conn.cursor()
                embedding = self.get_embedding(task_name).tobytes()
                cursor.execute("INSERT INTO Tasks (TaskName, TaskID) VALUES (?, ?)", (task_name, embedding))
                task_id = embedding
            self.cache[task_id] = bt_xml
            add_behavior_tree(conn, task_id, task_name, "Generated BT", bt_xml, 'system')

    def execute_behavior_tree(self, bt_root):
        # Placeholder for behavior tree execution logic
        print("Executing Behavior Tree:", bt_root)

        return bt_root.execute(self.robot_interface.get_state(), interface=self.robot_interface)

    def simulate_feedback(self, msg):
        # Placeholder for feedback simulation
        return "Feedback based on monitoring"

    def refine_and_update_bt(self, task_name, task_id, bt_xml, feedback):
        refined_bt_xml = self.llm_interface.refine_behavior_tree(task_name, self.actions, bt_xml, feedback)
        if refined_bt_xml:
            self.save_behavior_tree(task_name, task_id, refined_bt_xml)
            print("Behavior Tree refined and updated.")
        else:
            print("Unable to refine behavior tree based on feedback.")

    def manage_task(self, task_name):
        # Load or generate a behavior tree for the task
        task_embedding = self.get_embedding(task_name)

        task_incomplete = True
        while task_incomplete:
            task_id = self.find_most_similar_task(task_embedding, task_name)
            bt_root, bt_xml = self.load_or_generate_bt(task_id, task_name)
            # Execute the behavior tree
            success, msg = self.execute_behavior_tree(bt_root)
            # Get feedback based on execution, simulated here as a function
            if not success:
                print(f"Failed to execute behavior tree due to {msg}.")
                # Refine the behavior tree based on feedback
                self.refine_and_update_bt(task_name, task_embedding, bt_xml, msg)

if __name__ == "__main__":
    sim = AI2ThorSimEnv()
    cbtf = CognitiveBehaviorTreeFramework(sim)
    print(cbtf.manage_task("wash the mug in the sink"))
