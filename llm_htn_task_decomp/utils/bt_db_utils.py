import sqlite3

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create Tasks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Tasks (
        TaskID INTEGER PRIMARY KEY,
        TaskName TEXT NOT NULL,
        InitialDescription TEXT
    )''')

    # Create BehaviorTrees table to replace Decompositions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS BehaviorTrees (
        TaskID INTEGER NOT NULL,
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
        FOREIGN KEY (BehaviorTreeID) REFERENCES BehaviorTrees (TaskID)
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
        FOREIGN KEY (BehaviorTreeID) REFERENCES BehaviorTrees (TaskID)
    )''')

    conn.commit()
    conn.close()


def add_task_and_behavior_tree(task_name, initial_description, bt_xml, created_by):
    conn = sqlite3.connect('behavior_tree.db')
    cursor = conn.cursor()

    # Insert the task
    cursor.execute("INSERT INTO Tasks (TaskName, InitialDescription) VALUES (?, ?)", (task_name, initial_description))
    task_id = cursor.lastrowid

    # Insert the initial behavior tree
    cursor.execute("INSERT INTO BehaviorTrees (TaskID, BehaviorTreeXML, CreationDate, CreatedBy) VALUES (?, ?, datetime('now'), ?)", (task_id, bt_xml, created_by))

    conn.commit()
    conn.close()

def store_feedback(behavior_tree_id, user_id, feedback_text):
    conn = sqlite3.connect('behavior_tree.db')
    cursor = conn.cursor()

    cursor.execute("INSERT INTO Feedback (BehaviorTreeID, UserID, FeedbackText, FeedbackDate) VALUES (?, ?, ?, datetime('now'))", (behavior_tree_id, user_id, feedback_text))

    conn.commit()
    conn.close()

def store_feedback(behavior_tree_id, user_id, feedback_text):
    conn = sqlite3.connect('behavior_tree.db')
    cursor = conn.cursor()

    cursor.execute("INSERT INTO Feedback (BehaviorTreeID, UserID, FeedbackText, FeedbackDate) VALUES (?, ?, ?, datetime('now'))", (behavior_tree_id, user_id, feedback_text))

    conn.commit()
    conn.close()

def get_behavior_trees_with_feedback(task_id):
    conn = sqlite3.connect('behavior_tree.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT b.BehaviorTreeXML, f.FeedbackText
    FROM BehaviorTrees b
    LEFT JOIN Feedback f ON b.TaskID = f.BehaviorTreeID
    WHERE b.TaskID = ?
    ''', (task_id,))

    for row in cursor.fetchall():
        print("Behavior Tree XML:", row[0])
        print("Feedback:", row[1] if row[1] else "No feedback yet", "\n")

    conn.close()
