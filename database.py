# database.py
import sqlite3

def create_database():
    conn = sqlite3.connect("health_tracker.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, age INTEGER, weight REAL, height REAL, fitness_goal TEXT)''')
    conn.commit()
    conn.close()

def add_user(name, age, weight, height, goal):
    conn = sqlite3.connect("health_tracker.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (name, age, weight, height, fitness_goal) VALUES (?, ?, ?, ?, ?)",
              (name, age, weight, height, goal))
    conn.commit()
    conn.close()
