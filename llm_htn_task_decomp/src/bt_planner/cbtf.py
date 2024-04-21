import sqlite3
import os
from cachetools import LRUCache
from llm_htn_task_decomp.src.llm_interface.llm_interface_openai import LLMInterface
from llm_htn_task_decomp.utils.bt_db_utils import setup_database, add_task_and_behavior_tree, store_feedback

SIM_ACTION_MAP = {
    
}


class CognitiveBehaviorTreeFramework:
    def __init__(self, db_path='./behavior_tree.db', model_name="gpt-3.5-turbo", sim=True):
        self.db_path = db_path
        self.llm_interface = LLMInterface(model_name)
        self.cache = LRUCache(maxsize=100)
        if not os.path.exists(self.db_path):
            setup_database(self.db_path)

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def manage_task(self, task_name):
        # Load or generate a behavior tree for the task
        bt_xml = self.load_or_generate_bt(task_name)
        # Execute the behavior tree
        success, msg = self.execute_behavior_tree(bt_xml)
        # Get feedback based on execution, simulated here as a function
        feedback = self.simulate_feedback()
        # Refine the behavior tree based on feedback
        self.refine_and_update_bt(task_name, bt_xml, feedback)

    def load_or_generate_bt(self, task_name):
        bt_xml = self.load_behavior_tree(task_name)
        if not bt_xml:
            # Generate a new behavior tree if one does not exist
            bt_xml = self.llm_interface.get_behavior_tree(task_name)
            self.save_behavior_tree(task_name, bt_xml)
        return bt_xml

    def load_behavior_tree(self, task_name):
        # Try to retrieve from cache first
        bt_xml = self.cache.get(task_name)
        if not bt_xml:
            # Connect to the database and retrieve the behavior tree
            with self.connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT BehaviorTreeXML FROM BehaviorTrees WHERE TaskName = ?", (task_name,))
                row = cursor.fetchone()
                if row:
                    bt_xml = row[0]
                    self.cache[task_name] = bt_xml  # Update cache
        return bt_xml

    def save_behavior_tree(self, task_name, bt_xml):
        # Save to database and update cache
        with self.connect_db() as conn:
            add_task_and_behavior_tree(conn, task_name, "Generated BT", bt_xml, 'system')
            self.cache[task_name] = bt_xml

    def execute_behavior_tree(self, bt_xml):
        # Placeholder for behavior tree execution logic
        print("Executing Behavior Tree:", bt_xml)
        return True, "success"

    def simulate_feedback(self):
        # Placeholder for feedback simulation
        return "Feedback based on monitoring"

    def refine_and_update_bt(self, task_name, bt_xml, feedback):
        refined_bt_xml = self.llm_interface.refine_behavior_tree(task_name, bt_xml, feedback)
        if refined_bt_xml:
            self.save_behavior_tree(task_name, refined_bt_xml)
            print("Behavior Tree refined and updated.")
        else:
            print("Unable to refine behavior tree based on feedback.")

if __name__ == "__main__":
    cbtf = CognitiveBehaviorTreeFramework()
    cbtf.manage_task("clean the kitchen")
