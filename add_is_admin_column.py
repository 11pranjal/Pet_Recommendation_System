"""
Migration script to add is_admin column to existing User table
Run this if you get errors about is_admin column missing
"""

import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from database import db, User
from app_complete import app

def migrate():
    """Add is_admin column to User table"""
    with app.app_context():
        try:
            # Try to add the column using SQLite ALTER TABLE
            with db.engine.connect() as conn:
                # Check if column already exists
                result = conn.execute(db.text("PRAGMA table_info(user)"))
                columns = [row[1] for row in result]
                
                if 'is_admin' in columns:
                    print("✓ is_admin column already exists!")
                    return
                
                # Add the column
                conn.execute(db.text("ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0"))
                conn.commit()
                print("✓ Successfully added is_admin column to User table")
                print("✓ All existing users are set to is_admin=False by default")
                print("\nTo create an admin user, run:")
                print("  python setup_admin.py")
                
        except Exception as e:
            print(f"✗ Error during migration: {e}")
            print("\nIf you're still having issues, try:")
            print("1. Backup your database: cp instance/pet_adoption.db instance/pet_adoption.db.backup")
            print("2. Delete the database: rm instance/pet_adoption.db")
            print("3. Restart the app to create a fresh database")
            sys.exit(1)

if __name__ == "__main__":
    print("🔄 Starting migration: Adding is_admin column...")
    migrate()
    print("✅ Migration complete!")
