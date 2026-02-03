#!/usr/bin/env python3
"""
Add HiddenPet table to database
"""

import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'instance', 'pet_adoption.db')

print(f"Adding 'hidden_pets' table to database: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create hidden_pets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hidden_pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER UNIQUE NOT NULL,
            reason VARCHAR(200),
            hidden_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            hidden_by_admin_id INTEGER,
            FOREIGN KEY (hidden_by_admin_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("✅ Table 'hidden_pets' created successfully!")
    print("\nPhase 3 CRUD Operations Now Available:")
    print("✅ View all pets")
    print("✅ Filter & search pets")
    print("✅ View pet details")
    print("✅ Hide pets (soft delete)")
    print("✅ Unhide pets (reactivate)")
    print("\nRestart the server and check Admin Panel → Pets!")
    
except Exception as e:
    print(f"❌ Error: {e}")
