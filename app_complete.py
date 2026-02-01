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
import os
import sys
import re

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import db, User, AdoptionRequest, Favorite
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

# ==================== STATIC ROUTES ====================

@app.route("/")
def index():
    """Serve login page if not authenticated, else dashboard"""
    if current_user.is_authenticated:
        return send_from_directory(app.static_folder, "dashboard.html")
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
                "full_name": user.full_name
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
        
        pets = engine.get_all_pets(filters)
        pets = pets[:limit]
        
        return jsonify({
            "ok": True,
            "count": len(pets),
            "pets": pets
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

