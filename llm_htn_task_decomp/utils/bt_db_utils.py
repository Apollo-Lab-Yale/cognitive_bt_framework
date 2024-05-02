import sqlite3

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

# if __name__ == "__main__":
#     setup_database('behavior_tree.db')
#     with sqlite3.connect('behavior_tree.db') as conn:
#         add_task_and_behavior_tree(conn, "Clean the kitchen", "Initial setup for cleaning the kitchen", "<root></root>", "system")
#         behavior_tree_id = 1  # Assuming this ID was assigned
#         store_feedback(conn, behavior_tree_id, "user1", "Feedback based on initial trial")
#         get_behavior_trees_with_feedback(1)
