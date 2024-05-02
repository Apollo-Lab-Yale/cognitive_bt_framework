import sqlite3
from cachetools import LRUCache
from llm_htn_task_decomp.src.llm_interface.llm_interface_openai import LLMInterface
from llm_htn_task_decomp.utils.bt_db_utils import setup_database, add_behavior_tree, store_feedback
import os

class BehaviorTreePlanner:
    def __init__(self, llm_interface=None, db_path='./behavior_tree.db'):
        self.llm_interface = llm_interface
        self.db_path = db_path
        self.cache = LRUCache(maxsize=100)  # Adjust maxsize based on expected workload
        if not os.path.exists(self.db_path):
            setup_database(self.db_path)

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def load_behavior_tree(self, task_name):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TaskID FROM Tasks WHERE TaskName = ?", (task_name,))
            task_id_row = cursor.fetchone()

            if task_id_row is None:
                # If the task doesn't exist, create an initial placeholder behavior tree
                initial_description = "Initial setup for " + task_name
                bt_xml = "<root><action name='Start'/></root>"  # Placeholder BT XML
                add_task_and_behavior_tree(self.db_path, task_name, initial_description, bt_xml, 'system')
                cursor.execute("SELECT TaskID FROM Tasks WHERE TaskName = ?", (task_name,))
                task_id_row = cursor.fetchone()

            task_id = task_id_row[0]

            cursor.execute("SELECT BehaviorTreeXML FROM BehaviorTrees WHERE TaskID = ?", (task_id,))
            row = cursor.fetchone()

            if row:
                bt_xml = row[0]
            else:
                # If no behavior tree is found, query LLM to create one
                bt_xml = self.llm_interface.get_behavior_tree(task_name)
                cursor.execute("INSERT INTO BehaviorTrees (TaskID, BehaviorTreeXML, CreationDate, CreatedBy) VALUES (?, ?, datetime('now'), 'system')",
                               (task_id, bt_xml))
            self.cache[task_name] = bt_xml
            conn.commit()

        return bt_xml

    def refine_behavior_tree_with_feedback(self, task_name, user_id, user_feedback):
        current_bt_xml = self.load_behavior_tree(task_name)
        refined_bt_xml = self.llm_interface.query_llm_for_refinement(task_name, current_bt_xml, user_feedback)

        if refined_bt_xml:
            with self.connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT TaskID FROM Tasks WHERE TaskName = ?", (task_name,))
                task_id_row = cursor.fetchone()
                task_id = task_id_row[0]

                cursor.execute("INSERT INTO BehaviorTrees (TaskID, BehaviorTreeXML, CreationDate, CreatedBy) VALUES (?, ?, datetime('now'), 'system')",
                               (task_id, refined_bt_xml))
                behavior_tree_id = cursor.lastrowid
                store_feedback(self.db_path, behavior_tree_id, user_id, user_feedback)
                self.cache[task_name] = refined_bt_xml
                conn.commit()
            print("Behavior Tree refined and updated.")
        else:
            print("Unable to refine behavior tree based on feedback.")

    def execute_behavior_tree(self, bt_xml):
        # Placeholder for behavior tree execution logic
        # Detailed execution would depend on the robot's control system API
        print("Executing Behavior Tree:", bt_xml)

# Example usage
if __name__ == "__main__":
    llm_interface = LLMInterface()  # Assuming this is set up correctly
    planner = BehaviorTreePlanner(llm_interface)
    bt_xml = planner.load_behavior_tree("clean the kitchen")
    planner.execute_behavior_tree(bt_xml)
