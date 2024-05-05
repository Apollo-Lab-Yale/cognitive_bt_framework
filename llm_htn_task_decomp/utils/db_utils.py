import sqlite3
import datetime
import json

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create Tasks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Tasks (
        TaskID STRING PRIMARY KEY,
        TaskName TEXT NOT NULL,
        InitialDescription TEXT
    )''')

    # Create BehaviorTrees table to replace Decompositions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS BehaviorTrees (
        BehaviorTreeID INTEGER PRIMARY KEY,
        TaskID STRING NOT NULL,
        BehaviorTreeXML TEXT,
        CreationDate TEXT,
        CreatedBy TEXT,
        FOREIGN KEY (TaskID) REFERENCES Tasks (TaskID)
    )''')

    # Update Feedback table to reference BehaviorTrees instead of Decompositions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Feedback (
        FeedbackID INTEGER PRIMARY KEY,
        BehaviorTreeID INTEGER NOT NULL,
        UserID TEXT,
        FeedbackText TEXT,
        FeedbackDate TEXT,
        FOREIGN KEY (BehaviorTreeID) REFERENCES BehaviorTrees (BehaviorTreeID)
    )''')

    # Create BehaviorTreeAdjustments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS BehaviorTreeAdjustments (
        AdjustmentID INTEGER PRIMARY KEY,
        BehaviorTreeID INTEGER NOT NULL,
        PreviousBehaviorTreeXML TEXT,
        NewBehaviorTreeXML TEXT,
        AdjustmentReason TEXT,
        AdjustmentDate TEXT,
        FOREIGN KEY (BehaviorTreeID) REFERENCES BehaviorTrees (BehaviorTreeID)
    )''')

    cursor.execute('''
           CREATE TABLE IF NOT EXISTS Episodes (
               EpisodeID TEXT PRIMARY KEY,
               TaskName TEXT
           );
       ''')
    cursor.execute('''
           CREATE TABLE IF NOT EXISTS ObjectStates (
               StateID INTEGER PRIMARY KEY AUTOINCREMENT,
               ObjectID TEXT,
               State TEXT,
               EpisodeID TEXT,
               Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
               FOREIGN KEY (EpisodeID) REFERENCES Episodes(EpisodeID)
           );
       ''')
    conn.commit()
    conn.close()

def add_behavior_tree(conn, task_id, task_name, initial_description, bt_xml, created_by):
    cursor = conn.cursor()

    # # Insert the task
    # cursor.execute("INSERT INTO Tasks (TaskName, TaskID, InitialDescription) VALUES (?, ?, ?)", (task_name, task_id, initial_description))
    # task_id = cursor.lastrowid

    # Insert the initial behavior tree
    cursor.execute("INSERT INTO BehaviorTrees (TaskID, BehaviorTreeXML, CreationDate, CreatedBy) VALUES (?, ?, datetime('now'), ?)", (task_id, bt_xml, created_by))
    conn.commit()

def store_feedback(conn, behavior_tree_id, user_id, feedback_text):
    cursor = conn.cursor()

    cursor.execute("INSERT INTO Feedback (BehaviorTreeID, UserID, FeedbackText, FeedbackDate) VALUES (?, ?, ?, datetime('now'))", (behavior_tree_id, user_id, feedback_text))
    conn.commit()

def get_behavior_trees_with_feedback(task_id):
    conn = sqlite3.connect('behavior_tree.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT b.BehaviorTreeXML, f.FeedbackText
    FROM BehaviorTrees b
    LEFT JOIN Feedback f ON b.BehaviorTreeID = f.BehaviorTreeID
    WHERE b.TaskID = ?
    ''', (task_id,))

    for row in cursor.fetchall():
        print("Behavior Tree XML:", row[0])
        print("Feedback:", row[1] if row[1] else "No feedback yet", "\n")

    conn.close()


def start_new_episode(db_path, task_name):
    episode_id = datetime.datetime.now().isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Episodes (EpisodeID, TaskName) VALUES (?, ?)", (episode_id, task_name))
    conn.commit()
    conn.close()
    return episode_id

def store_object_state(db_path, object_id, state, episode_id):
    timestamp = datetime.datetime.now().isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ObjectStates (ObjectID, State, EpisodeID, Timestamp) VALUES (?, ?, ?, ?)",
                   (object_id, json.dumps(state), episode_id, timestamp))
    conn.commit()
    conn.close()

def retrieve_object_states_by_episode(db_path, episode_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ObjectStates WHERE EpisodeID = ? ORDER BY Timestamp", (episode_id,))
    results = cursor.fetchall()
    conn.close()
    return results

def get_episodes_by_task(db_path, task_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT EpisodeID FROM Episodes WHERE TaskName = ?", (task_name,))
    episodes = cursor.fetchall()
    conn.close()
    return [episode[0] for episode in episodes]

def retrieve_object_states_by_object_id(db_path, object_id):
    """
    Retrieves all records for a specific object across all episodes, sorted by timestamp.

    Parameters:
    - db_path (str): The path to the SQLite database file.
    - object_id (str): The unique identifier of the object.

    Returns:
    - list of tuples: Each tuple contains data about the object state at different timestamps.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT EpisodeID, State, Timestamp 
        FROM ObjectStates 
        WHERE ObjectID = ? 
        ORDER BY Timestamp ASC
    """, (object_id,))
    results = cursor.fetchall()
    conn.close()
    return results

def store_multiple_object_states(db_path, object_states, episode_id):
    """
    Stores states for multiple objects in a single database transaction.

    Parameters:
    - db_path (str): The path to the SQLite database file.
    - object_states (list of tuples): Each tuple contains (object_id, state), where state is a dictionary.
    - episode_id (str): The unique identifier of the episode.
    """
    # Convert state dictionaries to JSON strings within the tuple list
    object_states_json = [(obj_id, json.dumps(state), episode_id, datetime.datetime.now().isoformat()) for obj_id, state in object_states]

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Prepare the SQL query for multiple inserts
    cursor.executemany(
        "INSERT INTO ObjectStates (ObjectID, State, EpisodeID, Timestamp) VALUES (?, ?, ?, ?)",
        object_states_json
    )

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

# if __name__ == "__main__":
#     setup_database('behavior_tree.db')
#     with sqlite3.connect('behavior_tree.db') as conn:
#         add_task_and_behavior_tree(conn, "Clean the kitchen", "Initial setup for cleaning the kitchen", "<root></root>", "system")
#         behavior_tree_id = 1  # Assuming this ID was assigned
#         store_feedback(conn, behavior_tree_id, "user1", "Feedback based on initial trial")
#         get_behavior_trees_with_feedback(1)
