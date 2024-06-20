import sqlite3
import os
import numpy as np
from cachetools import LRUCache
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch
import json
import datetime
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterface
from cognitive_bt_framework.utils.db_utils import setup_database, add_behavior_tree
from cognitive_bt_framework.utils.bt_utils import parse_node, parse_bt_xml, BASE_EXAMPLE, FORMAT_EXAMPLE
from cognitive_bt_framework.utils.goal_gen_aithor import get_wash_mug_in_sink_goal, get_make_coffee, get_put_apple_in_fridge_goal
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_ACTIONS, AI2THOR_PREDICATES
from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from cognitive_bt_framework.src.cbt_planner.memory import Memory
from cognitive_bt_framework.src.bt_validation.validate import validate_bt
from cognitive_bt_framework.utils.logic_utils import cosine_similarity, stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

DEFAULT_DB_PATH = '/home/liam/dev/cognitive_bt_framework/cognitive_bt_framework/src/'

class CognitiveBehaviorTreeFramework:
    def __init__(self, robot_interface, actions=AI2THOR_ACTIONS, conditions=AI2THOR_PREDICATES, db_path=DEFAULT_DB_PATH, model_name="gpt-4o", sim=True):
        self.robot_interface = robot_interface
        self.db_path = db_path
        self.llm_interface = LLMInterface(model_name)
        self.bt_cache = LRUCache(maxsize=100)
        self.memory = Memory(db_path + f'behavior_tree.db')
        self.db_path += f'behavior_tree.db'
        self.actions = actions
        setup_database(self.db_path)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.keyword_model = KeyBERT('all-MiniLM-L6-v2')
        self.keyword_embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.object_names = set(robot_interface.get_object_classes())
        self.cosine_similarity_threshold = 0.90  # Cosine similarity threshold
        self.example = BASE_EXAMPLE
        self.conditions = conditions
        self.known_objects = []
        self.goal = None

    def get_embedding(self, text):
        # pp_text = preprocess_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().tobytes()

    def get_keywords(self, text, top_n=10):
        keywords = self.keyword_model.extract_keywords(text, top_n=top_n)
        return [keyword for keyword, score in keywords]


    def get_keyword_similarity(self, keywords1, keywords2):
        embeddings1 = self.keyword_embedder.encode(keywords1, convert_to_tensor=True)
        embeddings2 = self.keyword_embedder.encode(keywords2, convert_to_tensor=True)

        # Compute cosine similarity between embeddings
        cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_similarities

    def find_most_similar_task_decomp(self, embedding, task_name):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TaskID, TaskName, SubTasks FROM Tasks")
            best_match = None
            highest_similarity = 0.85
            for row in cursor.fetchall():
                try:
                    existing_task_name = row[1]
                    existing_embedding = bytes(row[0])
                    similarity = cosine_similarity(embedding, existing_embedding)[0][0]
                    print(existing_task_name, similarity)
                    if task_name == existing_task_name or similarity > highest_similarity:
                        best_match = row[2]
                        highest_similarity = similarity
                except Exception as e:
                    print(e)
                    pass
            return best_match

    def find_most_similar_task(self, embedding, task_name):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TaskID, TaskName, SubTasks FROM Tasks")
            best_match = None
            highest_similarity = -1
            for row in cursor.fetchall():
                try:
                    existing_task_name = row[1]
                    existing_embedding = np.frombuffer(row[0], dtype=np.float32)
                    similarity = cosine_similarity(embedding, existing_embedding)
                    if task_name == existing_task_name or similarity > highest_similarity:
                        best_match = row[2]
                        highest_similarity = similarity
                except Exception as e:
                    pass
            return best_match

    def set_goal(self, goal):
        self.goal = goal

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def load_or_generate_bt(self, task_id, task_name):
        bt_xml = self.load_behavior_tree(task_id)
        if bt_xml is None:
            bt_xml = self.llm_interface.get_behavior_tree(task_name, self.actions, self.conditions, self.example, self.object_names)
            # isValid = validate_bt(bt_xml)
            # if isValid != 'Valid':
            #     raise isValid
            self.save_behavior_tree(task_name, bt_xml, task_id)
        return parse_bt_xml(bt_xml , self.actions, self.conditions), bt_xml

    def load_behavior_tree(self, task_id):
        if task_id in self.bt_cache:
            return self.bt_cache[task_id]
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT BehaviorTreeXML FROM BehaviorTrees WHERE TaskID = ?", (task_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def save_behavior_tree(self, task_name, bt_xml, task_id):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            embedding = self.get_embedding(task_name)
            task_id = embedding
            self.bt_cache[task_id] = bt_xml
            add_behavior_tree(conn, task_id, task_name, "Generated BT", bt_xml, 'system')

    def execute_behavior_tree(self, bt_root):
        # Placeholder for behavior tree execution logic
        print("Executing Behavior Tree:", bt_root)
        return bt_root.execute(self.robot_interface.get_state(), interface=self.robot_interface, memory=self.memory)

    def simulate_feedback(self, msg):
        # Placeholder for feedback simulation
        return "Feedback based on monitoring"

    def update_known_objects(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ObjectId from ObjectStates")
            rows = cursor.fetchall()
            if rows is not None:
                obs = []
                for row in rows:
                    splt = row[0].split('|')
                    if splt[-1][0].isalpha():
                        obs.append(splt[-1])
                    else:
                        obs.append(splt[0])
                self.known_objects = set(obs)
            else:
                self.known_objects = set([obj.split('|')[0] if not obj.split('|')[-1].isalpha() else obj.split('|')[-1] for obj in self.memory.object_cache])

    def update_bt(self, task_name, task_id, bt_xml):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            if task_id is None:
                embedding = self.get_embedding(task_name)
                task_id = embedding
            sql = "INSERT OR REPLACE INTO BehaviorTrees ( TaskID, BehaviorTreeXML) VALUES ( ?, ?)"
            # Execute the SQL command
            cursor.execute(sql, (task_id, bt_xml))
            conn.commit()
            self.bt_cache[task_id] = bt_xml

    def refine_and_update_bt(self, task_name, task_id, bt_xml, feedback):
        refined_bt_xml = self.llm_interface.refine_behavior_tree(task_name, self.actions, self.conditions, bt_xml, feedback, self.object_names, self.example)
        if refined_bt_xml:
            # isValid = validate_bt(refined_bt_xml)
            # if isValid != 'Valid':
            #     raise Exception(f"Behavior tree is not valid: {isValid}")
            self.update_bt(task_name, task_id, refined_bt_xml)
            return parse_bt_xml(refined_bt_xml, self.actions, self.conditions)
            print("Behavior Tree refined and updated.")
        else:
            print("Unable to refine behavior tree based on feedback.")

    def generate_decomposition(self, task_name, task_embedding):
        decomp = self.llm_interface.get_task_decomposition(task_name).split('\n')
        print(decomp)
        task_name, decomp = decomp[0], decomp[1:]
        decomp_embeddings = [(subtask, self.get_embedding(subtask)) for subtask in decomp]
        with self.connect_db() as conn:
            cursor = conn.cursor()
            decomp_txt = "\n".join(f"{sub[0]}--{sub[1]}" for sub in decomp_embeddings)
            cursor.execute("INSERT INTO Tasks (TaskName, TaskID, SubTasks) VALUES (?, ?, ?)", (task_name, task_embedding, decomp_txt))
            for sub_task in decomp_embeddings:
                print(sub_task[0], sub_task[1])
                cursor.execute("INSERT INTO Tasks (TaskName, TaskID, SubTasks) VALUES (?, ?, ?)",
                               (sub_task[0], sub_task[1], f"{sub_task[0]}--{sub_task[1]}"))
        return decomp_embeddings

    def manage_task(self, task_name):
        episode_id = self.memory.start_new_episode(task_name)
        task_name = self.llm_interface.get_task_id(task_name)
        print(task_name)
        # Load or generate a behavior tree for the task
        task_embedding = self.get_embedding(task_name)
        task_incomplete = True
        sub_tasks = self.find_most_similar_task_decomp(task_embedding, task_name)
        self.update_known_objects()
        if sub_tasks is None:
            sub_tasks = self.generate_decomposition(task_name, task_embedding)
        else:
            sub_tasks = [task.split('--') for task in sub_tasks.split('\n')]
        for sub_task in sub_tasks:
            print("***********")
            print(sub_task)
            print(len(sub_task))
            (sub_task_name, sub_task_id) = sub_task
            try:
                bt_root, bt_xml = self.load_or_generate_bt(sub_task_id, sub_task_name)
                success, msg, subtree_xml = self.execute_behavior_tree(bt_root)
            except Exception as e:
                print(f"Failed to execute or parse behavior tree due to {e}.")
                subtree_xml = "NO SUBTREE DUE TO FAILURE"
                success = False
                msg = str(e)
            while not self.robot_interface.check_goal(self.goal):
                self.update_known_objects()
                # Get feedback based on execution, simulated here as a function
                if success and task_incomplete:
                    msg = f"Execution of behavior tree ended in success but task {task_name} is NOT COMPLETED."
                print(f"Failed to execute behavior tree due to {msg}.")
                # Refine the behavior tree based on feedback
                try:
                    new_root = self.refine_and_update_bt(sub_task_name, sub_task_id, subtree_xml, msg)
                    success, msg, subtree_xml = self.execute_behavior_tree(new_root)
                except Exception as e:
                    print(f"Failed to execute behavior tree due to {e}.")
                    print(type(e))
                    success = False
                    msg = str(e)
            print('Success!')
    def manage_task_nohtn(self, task_name):
        episode_id = self.memory.start_new_episode(task_name)
        task_id = self.llm_interface.get_task_id(task_name)
        # Load or generate a behavior tree for the task
        task_embedding = self.get_embedding(task_id)
        task_incomplete = True
        task_id = self.find_most_similar_task(task_embedding, task_name)
        self.update_known_objects()
        try:
            bt_root, bt_xml = self.load_or_generate_bt(task_id, task_name)
            success, msg, subtree_xml = self.execute_behavior_tree(bt_root)
        except Exception as e:
            print(f"Failed to execute or parse behavior tree due to {e}.")
            subtree_xml = "NO SUBTREE DUE TO FAILURE"
            success = False
            msg = str(e)
        while not self.robot_interface.check_goal(self.goal):
            self.update_known_objects()
            # Get feedback based on execution, simulated here as a function
            if success and task_incomplete:
                msg = f"Execution of behavior tree ended in success but task {task_name} is NOT COMPLETED."
            print(f"Failed to execute behavior tree due to {msg}.")
            # Refine the behavior tree based on feedback
            try:
                new_root = self.refine_and_update_bt(task_name, task_id, subtree_xml, msg)
                success, msg, subtree_xml = self.execute_behavior_tree(new_root)
            except Exception as e:
                print(f"Failed to execute behavior tree due to {e}.")
                print(type(e))
                success = False
                msg = str(e)
        print('Success!')

if __name__ == "__main__":
    sim = AI2ThorSimEnv()
    # goal, _ = get_make_coffee(sim)
    cbtf = CognitiveBehaviorTreeFramework(sim)
    cbtf.set_goal('apple')
    print(cbtf.manage_task("put the apple in the fridge"))
    # print(cbtf.llm_interface.get_task_id(" ".join(cbtf.get_keywords("bring me a cup of water"))))
