#!/usr/bin/env python3
"""
Add AdoptedPet table to database
"""

import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'instance', 'pet_adoption.db')

print(f"Adding 'adopted_pets' table to database: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create adopted_pets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS adopted_pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER UNIQUE NOT NULL,
            pet_name VARCHAR(100),
            pet_type VARCHAR(50),
            adopted_by_user_id INTEGER NOT NULL,
            adoption_request_id INTEGER NOT NULL,
            adopted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (adopted_by_user_id) REFERENCES users(id),
            FOREIGN KEY (adoption_request_id) REFERENCES adoption_requests(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("✅ Table 'adopted_pets' created successfully!")
    print("\nYou can now:")
    print("1. Restart the Flask server")
    print("2. Approve adoption requests")
    print("3. Pets will automatically be removed from available listings")
    
except Exception as e:
    print(f"❌ Error: {e}")
