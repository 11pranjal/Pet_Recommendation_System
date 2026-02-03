#!/usr/bin/env python3
"""
Add CustomPet table to database
"""

import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'instance', 'pet_adoption.db')

print(f"Adding 'custom_pets' table to database: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create custom_pets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            type VARCHAR(50) NOT NULL,
            breed VARCHAR(100) NOT NULL,
            age_years INTEGER NOT NULL,
            age_months INTEGER DEFAULT 0,
            size VARCHAR(20) NOT NULL,
            color VARCHAR(50) NOT NULL,
            gender VARCHAR(10) NOT NULL,
            weight_kg FLOAT NOT NULL,
            vaccinated BOOLEAN DEFAULT 0,
            health_condition VARCHAR(50) DEFAULT 'Good',
            kid_friendly BOOLEAN DEFAULT 1,
            energy_level VARCHAR(20) DEFAULT 'Moderate',
            description TEXT,
            pet_characteristics TEXT,
            fee FLOAT DEFAULT 100.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_by_admin_id INTEGER,
            FOREIGN KEY (created_by_admin_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("✅ Table 'custom_pets' created successfully!")
    print("\n🎉 Phase 3 NOW FULLY COMPLETE:")
    print("✅ View all pets")
    print("✅ Filter & search pets")
    print("✅ View pet details")
    print("✅ Hide pets (soft delete)")
    print("✅ Unhide pets (reactivate)")
    print("✅ ➕ ADD NEW PETS! ← NEW!")
    print("\nYou'll now see the '➕ Add New Pet' button!")
    print("Restart the server and test it!")
    
except Exception as e:
    print(f"❌ Error: {e}")
