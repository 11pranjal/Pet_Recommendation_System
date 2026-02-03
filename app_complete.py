"""
Complete Pet Adoption System with Authentication
- User registration and login
- Database integration
- Session management
- Pet recommendations (KNN + SBERT)
- Adoption requests
"""

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime
from functools import wraps
import os
import sys
import re

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import db, User, AdoptionRequest, Favorite, AdoptedPet, HiddenPet, CustomPet
from recommendation_engine import PetRecommendationEngine

app = Flask(__name__, static_folder="frontend", static_url_path="/")
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pet_adoption.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
CORS(app, supports_credentials=True)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'serve_login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize recommendation engine
try:
    engine = PetRecommendationEngine(model_dir="model")
    print("✅ Recommendation engine initialized successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️  Please run: python3 backend/enhanced_training.py")
    engine = None

# Password validation regex
PASSWORD_REGEX = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')

# Admin required decorator
def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"ok": False, "message": "Authentication required"}), 401
        if not current_user.is_admin:
            return jsonify({"ok": False, "message": "Admin privileges required"}), 403
        return f(*args, **kwargs)
    return decorated_function

# Helper function to get adopted pet IDs
def get_adopted_pet_ids():
    """Get list of adopted pet IDs"""
    adopted = AdoptedPet.query.all()
    return set(pet.pet_id for pet in adopted)

# Helper function to get hidden pet IDs
def get_hidden_pet_ids():
    """Get list of hidden/deactivated pet IDs"""
    hidden = HiddenPet.query.all()
    return set(pet.pet_id for pet in hidden)

# Helper function to get all unavailable pet IDs
def get_unavailable_pet_ids():
    """Get all pet IDs that shouldn't be shown (adopted + hidden)"""
    return get_adopted_pet_ids() | get_hidden_pet_ids()

# ==================== STATIC ROUTES ====================

@app.route("/")
def index():
    """Serve login page if not authenticated, else redirect based on role"""
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect("/admin-choice")
        return redirect("/dashboard")
    return send_from_directory(app.static_folder, "login.html")

@app.route("/login")
def serve_login():
    """Serve login page"""
    if current_user.is_authenticated:
        return redirect("/dashboard")
    return send_from_directory(app.static_folder, "login.html")

@app.route("/register")
def serve_register():
    """Serve registration page"""
    if current_user.is_authenticated:
        return redirect("/dashboard")
    return send_from_directory(app.static_folder, "register.html")

@app.route("/dashboard")
@login_required
def serve_dashboard():
    """Serve dashboard (requires authentication)"""
    return send_from_directory(app.static_folder, "dashboard.html")

@app.route("/admin-choice")
@login_required
def serve_admin_choice():
    """Serve admin choice page (only for admins)"""
    if not current_user.is_admin:
        return redirect("/dashboard")
    return send_from_directory(app.static_folder, "admin-choice.html")

@app.route("/admin-dashboard")
@login_required
def serve_admin_dashboard():
    """Serve admin dashboard (only for admins)"""
    if not current_user.is_admin:
        return redirect("/dashboard")
    return send_from_directory(app.static_folder, "admin-dashboard.html")

@app.route("/admin-pets.html")
@login_required
def serve_admin_pets():
    """Serve admin pets management page (only for admins)"""
    if not current_user.is_admin:
        return redirect("/dashboard")
    return send_from_directory(app.static_folder, "admin-pets.html")

@app.route("/<path:path>")
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

# ==================== AUTH ROUTES ====================

@app.route("/api/register", methods=["POST"])
def api_register():
    """User registration with validation"""
    try:
        data = request.json or {}
        
        # Validate required fields
        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        full_name = data.get("full_name", "").strip()
        
        # Validation
        if not username or len(username) < 3:
            return jsonify({"ok": False, "message": "Username must be at least 3 characters"}), 400
        
        if not email or "@" not in email:
            return jsonify({"ok": False, "message": "Valid email is required"}), 400
        
        if not password:
            return jsonify({"ok": False, "message": "Password is required"}), 400
        
        # Password criteria validation
        if len(password) < 8:
            return jsonify({"ok": False, "message": "Password must be at least 8 characters"}), 400
        
        if not re.search(r'[A-Z]', password):
            return jsonify({"ok": False, "message": "Password must contain at least one uppercase letter"}), 400
        
        if not re.search(r'[a-z]', password):
            return jsonify({"ok": False, "message": "Password must contain at least one lowercase letter"}), 400
        
        if not re.search(r'\d', password):
            return jsonify({"ok": False, "message": "Password must contain at least one number"}), 400
        
        if not re.search(r'[@$!%*?&#]', password):
            return jsonify({"ok": False, "message": "Password must contain at least one special character (@$!%*?&#)"}), 400
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            return jsonify({"ok": False, "message": "Username already exists"}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({"ok": False, "message": "Email already registered"}), 400
        
        # Create new user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            phone=data.get("phone", ""),
            address=data.get("address", "")
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            "ok": True,
            "message": "Registration successful! Please login.",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Registration error: {str(e)}"}), 500

@app.route("/api/login", methods=["POST"])
def api_login():
    """User login"""
    try:
        data = request.json or {}
        identifier = data.get("identifier", "").strip().lower()  # username or email
        password = data.get("password", "")
        
        if not identifier or not password:
            return jsonify({"ok": False, "message": "Username/email and password required"}), 400
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()
        
        if not user or not user.check_password(password):
            return jsonify({"ok": False, "message": "Invalid credentials"}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Login user
        login_user(user, remember=True)
        
        return jsonify({
            "ok": True,
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_admin": user.is_admin
            }
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Login error: {str(e)}"}), 500

@app.route("/api/logout", methods=["POST"])
@login_required
def api_logout():
    """User logout"""
    logout_user()
    return jsonify({"ok": True, "message": "Logged out successfully"})

@app.route("/api/user/me", methods=["GET"])
@login_required
def get_current_user():
    """Get current user info"""
    return jsonify({
        "ok": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "phone": current_user.phone,
            "address": current_user.address,
            "is_admin": current_user.is_admin,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None
        }
    })

@app.route("/api/user/profile", methods=["PUT"])
@login_required
def update_profile():
    """Update user profile"""
    try:
        data = request.json or {}
        
        if 'full_name' in data:
            current_user.full_name = data['full_name'].strip()
        if 'phone' in data:
            current_user.phone = data['phone'].strip()
        if 'address' in data:
            current_user.address = data['address'].strip()
        
        db.session.commit()
        
        return jsonify({"ok": True, "message": "Profile updated successfully"})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Update error: {str(e)}"}), 500

# ==================== RECOMMENDATION ROUTES ====================

@app.route("/api/recommend/quiz", methods=["POST"])
@login_required
def recommend_from_quiz():
    """Get recommendations based on quiz answers (requires login)"""
    if engine is None:
        return jsonify({"ok": False, "message": "Models not loaded. Please run training first."}), 500
    
    try:
        data = request.json or {}
        answers = data.get("answers", {})
        top_k = data.get("top_k", 5)
        
        recommendations = engine.recommend_from_quiz(answers, top_k=top_k)
        
        # Filter out adopted and hidden pets
        unavailable_pet_ids = get_unavailable_pet_ids()
        recommendations = [pet for pet in recommendations if pet.get('pet_id') not in unavailable_pet_ids]
        
        return jsonify({
            "ok": True,
            "method": "quiz",
            "count": len(recommendations),
            "recommendations": recommendations
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/recommend/text", methods=["POST"])
@login_required
def recommend_from_text():
    """Get recommendations based on text description (requires login)"""
    if engine is None:
        return jsonify({"ok": False, "message": "Models not loaded. Please run training first."}), 500
    
    try:
        data = request.json or {}
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        pet_type_filter = data.get("pet_type", None)
        
        if not query or len(query.strip()) < 3:
            return jsonify({"ok": False, "message": "Please provide a description (min 3 characters)"}), 400
        
        recommendations = engine.recommend_from_text(query, top_k=top_k, pet_type_filter=pet_type_filter)
        
        # Filter out adopted and hidden pets
        unavailable_pet_ids = get_unavailable_pet_ids()
        recommendations = [pet for pet in recommendations if pet.get('pet_id') not in unavailable_pet_ids]
        
        return jsonify({
            "ok": True,
            "method": "text_search",
            "query": query,
            "count": len(recommendations),
            "recommendations": recommendations
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

# ==================== PET ROUTES ====================

@app.route("/api/pets", methods=["GET"])
@login_required
def get_all_pets():
    """Get all pets with optional filters (requires login)"""
    if engine is None:
        return jsonify({"ok": False, "message": "Models not loaded"}), 500
    
    try:
        pet_type = request.args.get("type")
        size = request.args.get("size")
        kid_friendly = request.args.get("kid_friendly") == "true"
        energy_level = request.args.get("energy_level")
        limit = int(request.args.get("limit", 50))
        
        filters = {}
        if pet_type:
            filters['type'] = pet_type
        if size:
            filters['size'] = size
        if kid_friendly:
            filters['kid_friendly'] = True
        if energy_level:
            filters['energy_level'] = energy_level
        
        # Get pets from CSV/pickle
        pets = engine.get_all_pets(filters)
        
        # Get custom pets from database
        custom_pets_query = CustomPet.query
        if pet_type:
            custom_pets_query = custom_pets_query.filter_by(type=pet_type)
        if size:
            custom_pets_query = custom_pets_query.filter_by(size=size)
        if kid_friendly:
            custom_pets_query = custom_pets_query.filter_by(kid_friendly=True)
        if energy_level:
            custom_pets_query = custom_pets_query.filter_by(energy_level=energy_level)
        
        custom_pets = custom_pets_query.all()
        custom_pets_dict = [pet.to_dict() for pet in custom_pets]
        
        # Combine both lists
        all_pets = pets + custom_pets_dict
        
        # Filter out adopted and hidden pets
        unavailable_pet_ids = get_unavailable_pet_ids()
        all_pets = [pet for pet in all_pets if pet.get('pet_id') not in unavailable_pet_ids and str(pet.get('pet_id')) not in [str(pid) for pid in unavailable_pet_ids]]
        
        all_pets = all_pets[:limit]
        
        return jsonify({
            "ok": True,
            "count": len(all_pets),
            "pets": all_pets
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/pets/<int:pet_id>", methods=["GET"])
@login_required
def get_pet_details(pet_id):
    """Get detailed information for a specific pet"""
    if engine is None:
        return jsonify({"ok": False, "message": "Models not loaded"}), 500
    
    try:
        pet = engine.get_pet_by_id(pet_id)
        
        if pet is None:
            return jsonify({"ok": False, "message": "Pet not found"}), 404
        
        return jsonify({
            "ok": True,
            "pet": pet
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/stats", methods=["GET"])
def get_statistics():
    """Get database statistics (public)"""
    if engine is None:
        return jsonify({"ok": False, "message": "Models not loaded"}), 500
    
    try:
        stats = engine.get_statistics()
        stats['total_users'] = User.query.count()
        stats['total_adoptions'] = AdoptionRequest.query.count()
        
        return jsonify({
            "ok": True,
            "stats": stats
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

# ==================== ADOPTION ROUTES ====================

@app.route("/api/adopt", methods=["POST"])
@login_required
def submit_adoption_request():
    """Submit an adoption request (requires login)"""
    try:
        data = request.json or {}
        
        pet_id = data.get("pet_id")
        pet_name = data.get("pet_name")
        pet_type = data.get("pet_type", "")
        message = data.get("message", "")
        
        if not pet_id or not pet_name:
            return jsonify({"ok": False, "message": "Pet ID and name are required"}), 400
        
        # Check if pet is already adopted
        adopted_pet = AdoptedPet.query.filter_by(pet_id=pet_id).first()
        if adopted_pet:
            return jsonify({"ok": False, "message": "Sorry, this pet has already been adopted"}), 400
        
        # Check if user already requested this pet
        existing = AdoptionRequest.query.filter_by(
            user_id=current_user.id,
            pet_id=pet_id
        ).first()
        
        if existing:
            return jsonify({"ok": False, "message": "You already have a request for this pet"}), 400
        
        # Create adoption request
        adoption_request = AdoptionRequest(
            user_id=current_user.id,
            pet_id=pet_id,
            pet_name=pet_name,
            pet_type=pet_type,
            message=message
        )
        
        db.session.add(adoption_request)
        db.session.commit()
        
        return jsonify({
            "ok": True,
            "message": "Adoption request submitted successfully!",
            "request_id": adoption_request.id
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/adoptions/my", methods=["GET"])
@login_required
def get_my_adoption_requests():
    """Get current user's adoption requests"""
    try:
        requests = AdoptionRequest.query.filter_by(user_id=current_user.id).order_by(
            AdoptionRequest.created_at.desc()
        ).all()
        
        return jsonify({
            "ok": True,
            "count": len(requests),
            "requests": [req.to_dict() for req in requests]
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/favorites", methods=["GET"])
@login_required
def get_favorites():
    """Get user's favorite pets"""
    try:
        favorites = Favorite.query.filter_by(user_id=current_user.id).all()
        pet_ids = [fav.pet_id for fav in favorites]
        
        return jsonify({
            "ok": True,
            "count": len(pet_ids),
            "pet_ids": pet_ids
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/favorites/<int:pet_id>", methods=["POST"])
@login_required
def add_favorite(pet_id):
    """Add pet to favorites"""
    try:
        # Check if already favorited
        existing = Favorite.query.filter_by(user_id=current_user.id, pet_id=pet_id).first()
        if existing:
            return jsonify({"ok": False, "message": "Already in favorites"}), 400
        
        favorite = Favorite(user_id=current_user.id, pet_id=pet_id)
        db.session.add(favorite)
        db.session.commit()
        
        return jsonify({"ok": True, "message": "Added to favorites"})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/favorites/<int:pet_id>", methods=["DELETE"])
@login_required
def remove_favorite(pet_id):
    """Remove pet from favorites"""
    try:
        favorite = Favorite.query.filter_by(user_id=current_user.id, pet_id=pet_id).first()
        if not favorite:
            return jsonify({"ok": False, "message": "Not in favorites"}), 404
        
        db.session.delete(favorite)
        db.session.commit()
        
        return jsonify({"ok": True, "message": "Removed from favorites"})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

# ==================== ADMIN ROUTES ====================

@app.route("/api/admin/stats", methods=["GET"])
@admin_required
def admin_get_stats():
    """Get admin statistics"""
    try:
        stats = {}
        
        # User statistics
        stats['total_users'] = User.query.count()
        stats['admin_users'] = User.query.filter_by(is_admin=True).count()
        stats['regular_users'] = stats['total_users'] - stats['admin_users']
        
        # Adoption statistics
        stats['total_adoptions'] = AdoptionRequest.query.count()
        stats['pending_adoptions'] = AdoptionRequest.query.filter_by(status='pending').count()
        stats['approved_adoptions'] = AdoptionRequest.query.filter_by(status='approved').count()
        stats['rejected_adoptions'] = AdoptionRequest.query.filter_by(status='rejected').count()
        
        # Pet statistics
        # Get adopted pets count
        stats['adopted_pets'] = AdoptedPet.query.count()
        
        # Get hidden pets count
        stats['hidden_pets'] = HiddenPet.query.count()
        
        # Get total pets (CSV + custom)
        if engine:
            csv_pet_count = engine.get_statistics()['total_pets']
        else:
            csv_pet_count = 0
        
        custom_pet_count = CustomPet.query.count()
        stats['total_pets'] = csv_pet_count + custom_pet_count
        
        # Calculate available pets
        # Available = Total - Adopted - Hidden
        stats['available_pets'] = stats['total_pets'] - stats['adopted_pets'] - stats['hidden_pets']
        
        # Pet type breakdown (from engine)
        if engine:
            stats['pets_by_type'] = engine.get_statistics()['by_type']
        else:
            stats['pets_by_type'] = {}
        
        # Favorites
        stats['total_favorites'] = Favorite.query.count()
        
        return jsonify({
            "ok": True,
            "stats": stats
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/adoptions", methods=["GET"])
@admin_required
def admin_get_adoptions():
    """Get all adoption requests with filters"""
    try:
        status_filter = request.args.get("status")  # pending, approved, rejected, all
        pet_type_filter = request.args.get("pet_type")
        search_query = request.args.get("search", "").strip().lower()
        
        # Base query
        query = AdoptionRequest.query
        
        # Apply filters
        if status_filter and status_filter != "all":
            query = query.filter_by(status=status_filter)
        
        if pet_type_filter:
            query = query.filter_by(pet_type=pet_type_filter)
        
        # Get all matching requests
        requests = query.order_by(AdoptionRequest.created_at.desc()).all()
        
        # Apply search filter (on user name or pet name)
        if search_query:
            requests = [req for req in requests 
                       if search_query in req.pet_name.lower() 
                       or search_query in (req.user.full_name or req.user.username).lower()
                       or search_query in req.user.email.lower()]
        
        return jsonify({
            "ok": True,
            "count": len(requests),
            "requests": [req.to_dict() for req in requests]
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/adoptions/<int:request_id>", methods=["PUT"])
@admin_required
def admin_update_adoption(request_id):
    """Update adoption request status (approve/reject)"""
    try:
        data = request.json or {}
        new_status = data.get("status")
        
        if new_status not in ['approved', 'rejected', 'pending']:
            return jsonify({"ok": False, "message": "Invalid status"}), 400
        
        adoption_request = AdoptionRequest.query.get(request_id)
        
        if not adoption_request:
            return jsonify({"ok": False, "message": "Adoption request not found"}), 404
        
        # If approving, mark pet as adopted
        if new_status == 'approved':
            # Check if pet is already adopted
            existing_adoption = AdoptedPet.query.filter_by(pet_id=adoption_request.pet_id).first()
            if existing_adoption:
                return jsonify({
                    "ok": False, 
                    "message": "This pet has already been adopted by another user"
                }), 400
            
            # Mark pet as adopted
            adopted_pet = AdoptedPet(
                pet_id=adoption_request.pet_id,
                pet_name=adoption_request.pet_name,
                pet_type=adoption_request.pet_type,
                adopted_by_user_id=adoption_request.user_id,
                adoption_request_id=adoption_request.id
            )
            db.session.add(adopted_pet)
            
            # Reject all other pending requests for this pet
            other_requests = AdoptionRequest.query.filter(
                AdoptionRequest.pet_id == adoption_request.pet_id,
                AdoptionRequest.id != request_id,
                AdoptionRequest.status == 'pending'
            ).all()
            
            for req in other_requests:
                req.status = 'rejected'
                req.updated_at = datetime.utcnow()
        
        # Update the adoption request status
        adoption_request.status = new_status
        adoption_request.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        message = f"Adoption request {new_status}"
        if new_status == 'approved':
            message += f" and pet marked as adopted. {len(other_requests)} other pending requests were automatically rejected."
        
        return jsonify({
            "ok": True,
            "message": message,
            "request": adoption_request.to_dict()
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/users", methods=["GET"])
@admin_required
def admin_get_users():
    """Get all users"""
    try:
        search_query = request.args.get("search", "").strip().lower()
        
        # Get all users
        users = User.query.order_by(User.created_at.desc()).all()
        
        # Apply search filter
        if search_query:
            users = [user for user in users 
                    if search_query in user.username.lower() 
                    or search_query in (user.full_name or "").lower()
                    or search_query in user.email.lower()]
        
        return jsonify({
            "ok": True,
            "count": len(users),
            "users": [user.to_dict() for user in users]
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/users/<int:user_id>", methods=["GET"])
@admin_required
def admin_get_user_details(user_id):
    """Get detailed user information"""
    try:
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"ok": False, "message": "User not found"}), 404
        
        # Get user's adoption requests
        adoption_requests = AdoptionRequest.query.filter_by(user_id=user_id).order_by(
            AdoptionRequest.created_at.desc()
        ).all()
        
        # Get user's favorites
        favorites = Favorite.query.filter_by(user_id=user_id).all()
        
        user_data = user.to_dict()
        user_data['adoption_requests'] = [req.to_dict() for req in adoption_requests]
        user_data['favorites_count'] = len(favorites)
        
        return jsonify({
            "ok": True,
            "user": user_data
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/users/<int:user_id>/toggle-admin", methods=["PUT"])
@admin_required
def admin_toggle_user_admin(user_id):
    """Toggle user admin status"""
    try:
        # Prevent self-demotion
        if user_id == current_user.id:
            return jsonify({"ok": False, "message": "Cannot modify your own admin status"}), 400
        
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"ok": False, "message": "User not found"}), 404
        
        user.is_admin = not user.is_admin
        db.session.commit()
        
        action = "promoted to admin" if user.is_admin else "removed from admin"
        
        return jsonify({
            "ok": True,
            "message": f"User {action}",
            "user": user.to_dict()
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def admin_delete_user(user_id):
    """Delete a user (and their associated data)"""
    try:
        # Prevent self-deletion
        if user_id == current_user.id:
            return jsonify({"ok": False, "message": "Cannot delete your own account"}), 400
        
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"ok": False, "message": "User not found"}), 404
        
        username = user.username
        
        # Delete user (cascade will handle adoption_requests and favorites)
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({
            "ok": True,
            "message": f"User '{username}' deleted successfully"
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/adopted-pets", methods=["GET"])
@admin_required
def admin_get_adopted_pets():
    """Get all adopted pets"""
    try:
        adopted_pets = AdoptedPet.query.order_by(AdoptedPet.adopted_at.desc()).all()
        
        return jsonify({
            "ok": True,
            "count": len(adopted_pets),
            "adopted_pets": [pet.to_dict() for pet in adopted_pets]
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/pets/<int:pet_id>/hide", methods=["POST"])
@admin_required
def admin_hide_pet(pet_id):
    """Hide/deactivate a pet"""
    try:
        data = request.json or {}
        reason = data.get("reason", "Admin deactivated")
        
        # Check if already hidden
        existing = HiddenPet.query.filter_by(pet_id=pet_id).first()
        if existing:
            return jsonify({"ok": False, "message": "Pet is already hidden"}), 400
        
        # Check if adopted
        adopted = AdoptedPet.query.filter_by(pet_id=pet_id).first()
        if adopted:
            return jsonify({"ok": False, "message": "Cannot hide an adopted pet"}), 400
        
        # Hide the pet
        hidden_pet = HiddenPet(
            pet_id=pet_id,
            reason=reason,
            hidden_by_admin_id=current_user.id
        )
        db.session.add(hidden_pet)
        db.session.commit()
        
        return jsonify({
            "ok": True,
            "message": "Pet hidden successfully"
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/pets/<int:pet_id>/unhide", methods=["POST"])
@admin_required
def admin_unhide_pet(pet_id):
    """Unhide/reactivate a pet"""
    try:
        hidden_pet = HiddenPet.query.filter_by(pet_id=pet_id).first()
        
        if not hidden_pet:
            return jsonify({"ok": False, "message": "Pet is not hidden"}), 404
        
        db.session.delete(hidden_pet)
        db.session.commit()
        
        return jsonify({
            "ok": True,
            "message": "Pet unhidden successfully"
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/hidden-pets", methods=["GET"])
@admin_required
def admin_get_hidden_pets():
    """Get all hidden pets"""
    try:
        hidden_pets = HiddenPet.query.all()
        
        return jsonify({
            "ok": True,
            "count": len(hidden_pets),
            "hidden_pet_ids": [pet.pet_id for pet in hidden_pets]
        })
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/pets", methods=["POST"])
@admin_required
def admin_add_pet():
    """Add a new custom pet"""
    try:
        data = request.json or {}
        
        # Validate required fields
        required_fields = ['name', 'type', 'breed', 'age_years', 'size', 'color', 'gender', 'weight_kg']
        for field in required_fields:
            if field not in data:
                return jsonify({"ok": False, "message": f"Missing required field: {field}"}), 400
        
        # Create new custom pet
        custom_pet = CustomPet(
            name=data['name'],
            type=data['type'],
            breed=data['breed'],
            age_years=int(data['age_years']),
            age_months=data.get('age_months', 0),
            size=data['size'],
            color=data['color'],
            gender=data['gender'],
            weight_kg=float(data['weight_kg']),
            vaccinated=data.get('vaccinated', False),
            health_condition=data.get('health_condition', 'Good'),
            kid_friendly=data.get('kid_friendly', True),
            energy_level=data.get('energy_level', 'Moderate'),
            description=data.get('description', ''),
            pet_characteristics=data.get('description', ''),
            fee=data.get('fee', 100.0),
            created_by_admin_id=current_user.id
        )
        
        db.session.add(custom_pet)
        db.session.commit()
        
        return jsonify({
            "ok": True,
            "message": f"Pet '{custom_pet.name}' added successfully",
            "pet": custom_pet.to_dict()
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/pets/<pet_id>", methods=["DELETE"])
@admin_required
def admin_delete_pet(pet_id):
    """Delete a custom pet permanently"""
    try:
        # Only allow deletion of custom pets (not CSV pets)
        if not str(pet_id).startswith('custom_'):
            return jsonify({"ok": False, "message": "Can only delete custom pets. Use Hide for CSV pets."}), 400
        
        # Extract the numeric ID
        custom_id = int(str(pet_id).replace('custom_', ''))
        
        custom_pet = CustomPet.query.get(custom_id)
        
        if not custom_pet:
            return jsonify({"ok": False, "message": "Pet not found"}), 404
        
        # Check if pet is adopted
        adopted_pet = AdoptedPet.query.filter_by(pet_id=pet_id).first()
        if adopted_pet:
            return jsonify({"ok": False, "message": "Cannot delete an adopted pet"}), 400
        
        pet_name = custom_pet.name
        
        # Delete associated adoption requests
        AdoptionRequest.query.filter_by(pet_id=pet_id).delete()
        
        # Delete the pet
        db.session.delete(custom_pet)
        db.session.commit()
        
        return jsonify({
            "ok": True,
            "message": f"Pet '{pet_name}' deleted permanently"
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

# ==================== HEALTH CHECK ====================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "ok": True,
        "status": "running",
        "models_loaded": engine is not None,
        "pets_count": len(engine.pets_database) if engine else 0,
        "database_connected": db.engine.url.database is not None
    })

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "message": "Not found"}), 404

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"ok": False, "message": "Authentication required"}), 401

@app.errorhandler(500)
def server_error(e):
    return jsonify({"ok": False, "message": "Server error"}), 500

# ==================== DATABASE INITIALIZATION ====================

def init_database():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")

# ==================== MAIN ====================

if __name__ == "__main__":
    # Initialize database
    init_database()
    
    if engine is None:
        print("\n" + "="*60)
        print("⚠️  WARNING: Models not loaded!")
        print("="*60)
        print("Please run the training script first:")
        print("  $ python3 backend/enhanced_training.py")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("🐾 PET ADOPTION SYSTEM - COMPLETE VERSION 🐾")
        print("="*60)
        print("✓ Authentication: Enabled")
        print("✓ Database: SQLite")
        stats = engine.get_statistics()
        print(f"✓ Total Pets: {stats['total_pets']}")
        for pet_type, count in stats['by_type'].items():
            print(f"  - {pet_type}: {count}")
        print("="*60)
        print("🚀 Starting Flask server on http://localhost:5000")
        print("="*60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)

