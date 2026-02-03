#!/usr/bin/env python3
"""
Admin Setup Script for Pet Adoption System
This script helps set up the first admin user
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask
from database import db, User
from getpass import getpass
import re

app = Flask(__name__)
# Use absolute path to ensure database is found
db_path = os.path.join(os.path.dirname(__file__), 'instance', 'pet_adoption.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def validate_password(password):
    """Validate password meets requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[@$!%*?&#]', password):
        return False, "Password must contain at least one special character (@$!%*?&#)"
    return True, "Valid"

def create_admin_user():
    """Create an admin user interactively"""
    with app.app_context():
        print("\n" + "="*60)
        print("🔧 PET ADOPTION SYSTEM - ADMIN SETUP")
        print("="*60 + "\n")
        
        # Check if there are any admins
        admin_count = User.query.filter_by(is_admin=True).count()
        if admin_count > 0:
            print(f"ℹ️  There are already {admin_count} admin(s) in the system.")
            response = input("Do you want to create another admin? (yes/no): ").lower()
            if response not in ['yes', 'y']:
                print("Aborted.")
                return
        
        print("Let's create an admin account!\n")
        
        # Username
        while True:
            username = input("Username (min 3 characters): ").strip()
            if len(username) < 3:
                print("❌ Username must be at least 3 characters\n")
                continue
            
            # Check if username exists
            if User.query.filter_by(username=username).first():
                print("❌ Username already exists\n")
                continue
            
            break
        
        # Email
        while True:
            email = input("Email: ").strip().lower()
            if "@" not in email:
                print("❌ Invalid email address\n")
                continue
            
            # Check if email exists
            if User.query.filter_by(email=email).first():
                print("❌ Email already registered\n")
                continue
            
            break
        
        # Password
        while True:
            password = getpass("Password (min 8 chars, uppercase, lowercase, number, special char): ")
            valid, message = validate_password(password)
            
            if not valid:
                print(f"❌ {message}\n")
                continue
            
            password_confirm = getpass("Confirm password: ")
            if password != password_confirm:
                print("❌ Passwords don't match\n")
                continue
            
            break
        
        # Full name (optional)
        full_name = input("Full name (optional): ").strip()
        
        # Create user
        try:
            user = User(
                username=username,
                email=email,
                full_name=full_name if full_name else None,
                is_admin=True
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            print("\n" + "="*60)
            print("✅ ADMIN USER CREATED SUCCESSFULLY!")
            print("="*60)
            print(f"Username: {username}")
            print(f"Email: {email}")
            print(f"Role: Admin")
            print("="*60 + "\n")
            print("You can now login with these credentials.")
            print("After login, you'll be presented with a choice:")
            print("  - Customer View: Browse and adopt pets")
            print("  - Admin Panel: Manage the system\n")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Error creating admin user: {e}\n")

def make_user_admin():
    """Make an existing user an admin"""
    with app.app_context():
        print("\n" + "="*60)
        print("🔧 MAKE EXISTING USER ADMIN")
        print("="*60 + "\n")
        
        # Show all users
        users = User.query.all()
        if not users:
            print("No users found in the system.")
            return
        
        print("Current users:")
        for user in users:
            role = "Admin" if user.is_admin else "User"
            print(f"  - {user.username} ({user.email}) - {role}")
        
        print()
        username = input("Enter username to make admin: ").strip()
        
        user = User.query.filter_by(username=username).first()
        if not user:
            print(f"❌ User '{username}' not found.")
            return
        
        if user.is_admin:
            print(f"ℹ️  User '{username}' is already an admin.")
            return
        
        user.is_admin = True
        db.session.commit()
        
        print(f"\n✅ User '{username}' is now an admin!\n")

def main():
    """Main menu"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            create_admin_user()
        elif sys.argv[1] == "promote":
            make_user_admin()
        else:
            print("Usage:")
            print("  python3 setup_admin.py create   - Create new admin user")
            print("  python3 setup_admin.py promote  - Make existing user admin")
    else:
        print("\n" + "="*60)
        print("🔧 PET ADOPTION SYSTEM - ADMIN SETUP")
        print("="*60 + "\n")
        print("1. Create new admin user")
        print("2. Make existing user admin")
        print("3. Exit")
        print()
        
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == "1":
            create_admin_user()
        elif choice == "2":
            make_user_admin()
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
