import sqlite3

conn = sqlite3.connect("audit.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT,
    message TEXT
)
""")

conn.commit()

def save_audit(agent, message):

    cursor.execute(
        "INSERT INTO audit_logs (agent, message) VALUES (?, ?)",
        (agent, message)
    )

    conn.commit()