"""
Complete Pet Adoption System with Authentication
- User registration and login
- Database integration
- Session management
- Pet recommendations (KNN + SBERT) with hard-constraint filtering
- Adoption requests
"""

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime, date, timedelta
from functools import wraps
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv
import os
import sys
import re
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env file
load_dotenv()

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import db, User, AdoptionRequest, Favorite, AdoptedPet, HiddenPet, CustomPet
from recommendation_engine import PetRecommendationEngine
from pet_image_mapper import get_pet_image_url

app = Flask(__name__, static_folder="frontend", static_url_path="/")

# ── Pet image directory ──
PETIMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "petimage", "petimage")

# ── Upload config ──
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "pets")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload

app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pet_adoption.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Gmail SMTP Configuration ──
MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
MAIL_SENDER_NAME = "Adoptly"

# ── Token serializer for email verification & password reset ──
token_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

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

# Z-score normalization constants from the original dataset
# (Pet_Recommendation_System.csv has z-scored AgeMonths & WeightKg; custom
#  pets must be converted to the same z-score space before KNN matching.)
# Computed from dataset/fully_updated_pet_dataset.csv:
DATASET_AGE_MEAN = 81.5571788413    # months
DATASET_AGE_STD = 57.8522771189     # months
DATASET_WEIGHT_MEAN = 6.8641311952  # kg
DATASET_WEIGHT_STD = 8.7927497646   # kg

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

# ==================== EMAIL HELPERS ====================

def generate_verification_token(email):
    """Generate a signed token for email verification"""
    return token_serializer.dumps(email, salt='email-verification')

def verify_email_token(token, max_age=3600):
    """Verify an email verification token (valid for 1 hour)"""
    try:
        email = token_serializer.loads(token, salt='email-verification', max_age=max_age)
        return email
    except (SignatureExpired, BadSignature):
        return None

def generate_reset_token(email):
    """Generate a signed token for password reset"""
    return token_serializer.dumps(email, salt='password-reset')

def verify_reset_token(token, max_age=3600):
    """Verify a password reset token (valid for 1 hour)"""
    try:
        email = token_serializer.loads(token, salt='password-reset', max_age=max_age)
        return email
    except (SignatureExpired, BadSignature):
        return None

def validate_nepal_phone(phone):
    """Validate Nepal phone number.
    Must be exactly 10 digits (with optional +977 country code).
    Returns cleaned 10-digit number or None if invalid.
    """
    if not phone:
        return None
    cleaned = re.sub(r'[\s\-\(\)]', '', phone.strip())
    # Remove country code if present
    if cleaned.startswith('+977'):
        cleaned = cleaned[4:]
    elif cleaned.startswith('977') and len(cleaned) > 10:
        cleaned = cleaned[3:]
    # Must be exactly 10 digits starting with 96, 97, or 98
    if re.match(r'^(96|97|98)\d{8}$', cleaned):
        return cleaned
    return None

def send_email(to_email, subject, html_body):
    """Send an email via Gmail SMTP"""
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{MAIL_SENDER_NAME} <{MAIL_USERNAME}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_USERNAME, to_email, msg.as_string())

        print(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Email send error: {e}")
        return False

def send_verification_email(user_email, username, token):
    """Send email verification link"""
    verify_url = f"{request.host_url}verify-email?token={token}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 520px; margin: 0 auto; padding: 32px 24px; background: #f4f1ec; border-radius: 16px;">
        <div style="text-align: center; margin-bottom: 24px;">
            <div style="display: inline-block; background: linear-gradient(135deg, #4a7560, #7daa8e); padding: 14px; border-radius: 14px; margin-bottom: 12px;">
                <span style="font-size: 28px; color: white;">&#128062;</span>
            </div>
            <h1 style="color: #2c3e36; font-size: 22px; margin: 0;">Welcome to Adoptly!</h1>
        </div>
        <div style="background: white; border-radius: 14px; padding: 28px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);">
            <p style="color: #2c3e36; font-size: 15px; margin: 0 0 8px;">Hi <strong>{username}</strong>,</p>
            <p style="color: #5f7268; font-size: 14px; line-height: 1.6; margin: 0 0 24px;">
                Thank you for creating an account! Please verify your email address to start finding your perfect companion.
            </p>
            <div style="text-align: center; margin: 24px 0;">
                <a href="{verify_url}" style="display: inline-block; background: linear-gradient(135deg, #4a7560, #7daa8e); color: white; text-decoration: none; padding: 13px 36px; border-radius: 10px; font-weight: 700; font-size: 15px; box-shadow: 0 3px 12px rgba(125,170,142,0.3);">
                    Verify Email Address
                </a>
            </div>
            <p style="color: #8fa398; font-size: 12px; text-align: center; margin: 20px 0 0;">
                This link expires in 1 hour. If you didn't create an account, you can ignore this email.
            </p>
        </div>
        <p style="color: #8fa398; font-size: 11px; text-align: center; margin-top: 16px;">
            &copy; Adoptly — Find your perfect companion
        </p>
    </div>
    """
    return send_email(user_email, "Verify Your Email — Adoptly", html)

def send_reset_email(user_email, username, token):
    """Send password reset link"""
    reset_url = f"{request.host_url}reset-password?token={token}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 520px; margin: 0 auto; padding: 32px 24px; background: #f4f1ec; border-radius: 16px;">
        <div style="text-align: center; margin-bottom: 24px;">
            <div style="display: inline-block; background: linear-gradient(135deg, #4a7560, #7daa8e); padding: 14px; border-radius: 14px; margin-bottom: 12px;">
                <span style="font-size: 28px; color: white;">&#128274;</span>
            </div>
            <h1 style="color: #2c3e36; font-size: 22px; margin: 0;">Password Reset</h1>
        </div>
        <div style="background: white; border-radius: 14px; padding: 28px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);">
            <p style="color: #2c3e36; font-size: 15px; margin: 0 0 8px;">Hi <strong>{username}</strong>,</p>
            <p style="color: #5f7268; font-size: 14px; line-height: 1.6; margin: 0 0 24px;">
                We received a request to reset your password. Click the button below to set a new password.
            </p>
            <div style="text-align: center; margin: 24px 0;">
                <a href="{reset_url}" style="display: inline-block; background: linear-gradient(135deg, #4a7560, #7daa8e); color: white; text-decoration: none; padding: 13px 36px; border-radius: 10px; font-weight: 700; font-size: 15px; box-shadow: 0 3px 12px rgba(125,170,142,0.3);">
                    Reset Password
                </a>
            </div>
            <p style="color: #8fa398; font-size: 12px; text-align: center; margin: 20px 0 0;">
                This link expires in 1 hour. If you didn't request this, you can safely ignore this email.
            </p>
        </div>
        <p style="color: #8fa398; font-size: 11px; text-align: center; margin-top: 16px;">
            &copy; Adoptly — Find your perfect companion
        </p>
    </div>
    """
    return send_email(user_email, "Reset Your Password — Adoptly", html)

def send_adoption_request_notification(admin_email, user_name, user_email_addr, pet_name, message):
    """Notify admin when a new adoption request is submitted"""
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 520px; margin: 0 auto; padding: 32px 24px; background: #f4f1ec; border-radius: 16px;">
        <div style="text-align: center; margin-bottom: 24px;">
            <div style="display: inline-block; background: linear-gradient(135deg, #4a7560, #7daa8e); padding: 14px; border-radius: 14px; margin-bottom: 12px;">
                <span style="font-size: 28px; color: white;">&#128062;</span>
            </div>
            <h1 style="color: #2c3e36; font-size: 22px; margin: 0;">New Adoption Request</h1>
        </div>
        <div style="background: white; border-radius: 14px; padding: 28px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);">
            <p style="color: #2c3e36; font-size: 15px; margin: 0 0 16px;">A new adoption request has been submitted:</p>
            <table style="width: 100%; font-size: 14px; color: #5f7268;">
                <tr><td style="padding: 6px 0; font-weight: 600;">Pet:</td><td>{pet_name}</td></tr>
                <tr><td style="padding: 6px 0; font-weight: 600;">Requested by:</td><td>{user_name}</td></tr>
                <tr><td style="padding: 6px 0; font-weight: 600;">Email:</td><td>{user_email_addr}</td></tr>
                {f'<tr><td style="padding: 6px 0; font-weight: 600;">Message:</td><td>{message}</td></tr>' if message else ''}
            </table>
            <p style="color: #8fa398; font-size: 12px; text-align: center; margin: 20px 0 0;">
                Log in to the admin panel to review this request.
            </p>
        </div>
        <p style="color: #8fa398; font-size: 11px; text-align: center; margin-top: 16px;">
            &copy; Adoptly — Find your perfect companion
        </p>
    </div>
    """
    return send_email(admin_email, f"New Adoption Request — {pet_name}", html)

def send_adoption_status_notification(user_email_addr, username, pet_name, status):
    """Notify user when their adoption request is approved or rejected"""
    if status == 'approved':
        emoji = "&#127881;"
        title = "Adoption Approved!"
        color = "#22c55e"
        body = f"""
            <p style="color: #5f7268; font-size: 14px; line-height: 1.6; margin: 0 0 16px;">
                Great news! Your adoption request for <strong>{pet_name}</strong> has been <strong style="color: #22c55e;">approved</strong>!
            </p>
            <p style="color: #5f7268; font-size: 14px; line-height: 1.6; margin: 0 0 24px;">
                Please check your dashboard for further details. We will contact you soon to arrange the adoption process.
            </p>
        """
    else:
        emoji = "&#128532;"
        title = "Adoption Request Update"
        color = "#ef4444"
        body = f"""
            <p style="color: #5f7268; font-size: 14px; line-height: 1.6; margin: 0 0 16px;">
                We're sorry, but your adoption request for <strong>{pet_name}</strong> was <strong style="color: #ef4444;">not approved</strong> at this time.
            </p>
            <p style="color: #5f7268; font-size: 14px; line-height: 1.6; margin: 0 0 24px;">
                Don't worry — there are many other pets looking for a loving home! Visit your dashboard to explore more options.
            </p>
        """
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 520px; margin: 0 auto; padding: 32px 24px; background: #f4f1ec; border-radius: 16px;">
        <div style="text-align: center; margin-bottom: 24px;">
            <div style="display: inline-block; background: linear-gradient(135deg, #4a7560, #7daa8e); padding: 14px; border-radius: 14px; margin-bottom: 12px;">
                <span style="font-size: 28px; color: white;">{emoji}</span>
            </div>
            <h1 style="color: #2c3e36; font-size: 22px; margin: 0;">{title}</h1>
        </div>
        <div style="background: white; border-radius: 14px; padding: 28px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);">
            <p style="color: #2c3e36; font-size: 15px; margin: 0 0 8px;">Hi <strong>{username}</strong>,</p>
            {body}
        </div>
        <p style="color: #8fa398; font-size: 11px; text-align: center; margin-top: 16px;">
            &copy; Adoptly — Find your perfect companion
        </p>
    </div>
    """
    return send_email(user_email_addr, f"{title} — {pet_name}", html)

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
    """Serve landing page if not authenticated, else redirect based on role"""
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect("/admin-choice")
        return redirect("/dashboard")
    return send_from_directory(app.static_folder, "index.html")

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

@app.route("/forgot-password")
def serve_forgot_password():
    """Serve forgot password page"""
    return send_from_directory(app.static_folder, "forgot-password.html")

@app.route("/reset-password")
def serve_reset_password():
    """Serve reset password page"""
    return send_from_directory(app.static_folder, "reset-password.html")

@app.route("/verify-email")
def serve_verify_email():
    """Handle email verification link"""
    token = request.args.get("token", "")
    if not token:
        return send_from_directory(app.static_folder, "verify-email-result.html")
    
    email = verify_email_token(token)
    if email:
        user = User.query.filter_by(email=email).first()
        if user and not user.email_verified:
            user.email_verified = True
            db.session.commit()
        return redirect("/verify-email-success")
    else:
        return redirect("/verify-email-failed")

@app.route("/verify-email-success")
def serve_verify_email_success():
    """Serve verification success page"""
    return send_from_directory(app.static_folder, "verify-email-result.html")

@app.route("/verify-email-failed")
def serve_verify_email_failed():
    """Serve verification failed page"""
    return send_from_directory(app.static_folder, "verify-email-result.html")

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

@app.route("/petimage/<path:filepath>")
def serve_pet_image(filepath):
    """Serve pet images from the petimage directory"""
    return send_from_directory(PETIMAGE_DIR, filepath)

@app.route("/uploads/pets/<path:filepath>")
def serve_uploaded_pet_image(filepath):
    """Serve admin-uploaded pet images"""
    return send_from_directory(UPLOAD_DIR, filepath)

@app.route("/api/admin/upload-pet-image", methods=["POST"])
@admin_required
def upload_pet_image():
    """Upload a pet image. Returns the filename to store in the DB."""
    if 'image' not in request.files:
        return jsonify({"ok": False, "message": "No image file provided"}), 400

    f = request.files['image']
    if f.filename == '':
        return jsonify({"ok": False, "message": "No file selected"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_IMG_EXT:
        return jsonify({"ok": False, "message": f"Invalid file type. Allowed: {', '.join(ALLOWED_IMG_EXT)}"}), 400

    # Generate unique filename to avoid collisions
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    f.save(save_path)

    return jsonify({
        "ok": True,
        "filename": unique_name,
        "image_url": f"/uploads/pets/{unique_name}"
    })

@app.route("/api/admin/delete-pet-image", methods=["POST"])
@admin_required
def delete_pet_image():
    """Delete an uploaded pet image file."""
    data = request.json or {}
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"ok": False, "message": "No filename provided"}), 400

    # Secure: only allow simple filenames, no path traversal
    safe = secure_filename(filename)
    fpath = os.path.join(UPLOAD_DIR, safe)
    if os.path.isfile(fpath):
        os.remove(fpath)
    return jsonify({"ok": True})

@app.route("/<path:path>")
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

# ==================== AUTH ROUTES ====================

@app.route("/api/register", methods=["POST"])
def api_register():
    """User registration with email verification"""
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
        
        # Validate phone number (Nepal format)
        phone = data.get("phone", "").strip()
        if phone:
            validated_phone = validate_nepal_phone(phone)
            if not validated_phone:
                return jsonify({"ok": False, "message": "Please enter a valid Nepal phone number (e.g. 98XXXXXXXX)"}), 400
            phone = validated_phone

        # Create new user (email not verified yet)
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            phone=phone,
            address=data.get("address", ""),
            email_verified=False
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Send verification email
        token = generate_verification_token(email)
        email_sent = send_verification_email(email, username, token)
        
        if email_sent:
            return jsonify({
                "ok": True,
                "message": "Account created! Please check your email to verify your account.",
                "needs_verification": True,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email
                }
            })
        else:
            return jsonify({
                "ok": True,
                "message": "Account created but we couldn't send the verification email. Please request a new one from the login page.",
                "needs_verification": True,
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
    """User login — requires verified email"""
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
        
        # Check if email is verified
        if not user.email_verified:
            return jsonify({
                "ok": False,
                "message": "Please verify your email before logging in. Check your inbox for the verification link.",
                "email_not_verified": True,
                "email": user.email
            }), 403
        
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

@app.route("/api/resend-verification", methods=["POST"])
def api_resend_verification():
    """Resend email verification link"""
    try:
        data = request.json or {}
        email = data.get("email", "").strip().lower()
        
        if not email:
            return jsonify({"ok": False, "message": "Email is required"}), 400
        
        user = User.query.filter_by(email=email).first()
        
        if not user:
            # Don't reveal whether email exists
            return jsonify({"ok": True, "message": "If an account exists with this email, a verification link has been sent."})
        
        if user.email_verified:
            return jsonify({"ok": False, "message": "This email is already verified. You can log in."}), 400
        
        token = generate_verification_token(email)
        send_verification_email(email, user.username, token)
        
        return jsonify({"ok": True, "message": "Verification email sent! Please check your inbox."})
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/forgot-password", methods=["POST"])
def api_forgot_password():
    """Send password reset link"""
    try:
        data = request.json or {}
        email = data.get("email", "").strip().lower()
        
        if not email:
            return jsonify({"ok": False, "message": "Email is required"}), 400
        
        user = User.query.filter_by(email=email).first()
        
        # Always return success to prevent email enumeration
        if not user:
            return jsonify({"ok": True, "message": "If an account exists with this email, a password reset link has been sent."})
        
        token = generate_reset_token(email)
        send_reset_email(email, user.username, token)
        
        return jsonify({"ok": True, "message": "If an account exists with this email, a password reset link has been sent."})
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/reset-password", methods=["POST"])
def api_reset_password():
    """Reset password using token"""
    try:
        data = request.json or {}
        token = data.get("token", "")
        new_password = data.get("password", "")
        
        if not token:
            return jsonify({"ok": False, "message": "Invalid reset link"}), 400
        
        if not new_password:
            return jsonify({"ok": False, "message": "Password is required"}), 400
        
        # Validate password criteria
        if len(new_password) < 8:
            return jsonify({"ok": False, "message": "Password must be at least 8 characters"}), 400
        if not re.search(r'[A-Z]', new_password):
            return jsonify({"ok": False, "message": "Password must contain at least one uppercase letter"}), 400
        if not re.search(r'[a-z]', new_password):
            return jsonify({"ok": False, "message": "Password must contain at least one lowercase letter"}), 400
        if not re.search(r'\d', new_password):
            return jsonify({"ok": False, "message": "Password must contain at least one number"}), 400
        if not re.search(r'[@$!%*?&#]', new_password):
            return jsonify({"ok": False, "message": "Password must contain at least one special character (@$!%*?&#)"}), 400
        
        # Verify token
        email = verify_reset_token(token)
        if not email:
            return jsonify({"ok": False, "message": "This reset link has expired or is invalid. Please request a new one."}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"ok": False, "message": "User not found"}), 404
        
        # Update password
        user.set_password(new_password)
        # Also verify email if not already done
        if not user.email_verified:
            user.email_verified = True
        db.session.commit()
        
        return jsonify({"ok": True, "message": "Password reset successful! You can now log in with your new password."})
    
    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

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
            phone = data['phone'].strip()
            if phone:
                validated_phone = validate_nepal_phone(phone)
                if not validated_phone:
                    return jsonify({"ok": False, "message": "Please enter a valid Nepal phone number (e.g. 98XXXXXXXX)"}), 400
                current_user.phone = validated_phone
            else:
                current_user.phone = ''
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

        # Filter out adopted and hidden pets (use 'id' — dataset pets don't have 'pet_id')
        unavailable_pet_ids = get_unavailable_pet_ids()
        recommendations = [pet for pet in recommendations if pet.get('id') not in unavailable_pet_ids]
        
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

        # Filter out adopted and hidden pets (use 'id' — dataset pets don't have 'pet_id')
        unavailable_pet_ids = get_unavailable_pet_ids()
        recommendations = [pet for pet in recommendations if pet.get('id') not in unavailable_pet_ids]
        
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
    """Get all pets with optional filters and pagination (requires login)"""
    if engine is None:
        return jsonify({"ok": False, "message": "Models not loaded"}), 500
    
    try:
        pet_type = request.args.get("type")
        size = request.args.get("size")
        kid_friendly = request.args.get("kid_friendly") == "true"
        energy_level = request.args.get("energy_level")
        
        # Pagination parameters
        # Support legacy 'limit' param (used by admin panel) — disables pagination
        legacy_limit = request.args.get("limit")
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 24))))
        
        filters = {}
        if pet_type:
            filters['type'] = pet_type
        if size:
            filters['size'] = size
        if kid_friendly:
            filters['kid_friendly'] = True
        if energy_level:
            filters['energy_level'] = energy_level
        
        # Get all pets (dataset + custom pets registered in engine)
        all_pets = engine.get_all_pets(filters)
        
        # Filter out adopted and hidden pets (use 'id' — dataset pets don't have 'pet_id')
        unavailable_pet_ids = get_unavailable_pet_ids()
        all_pets = [pet for pet in all_pets if pet.get('id') not in unavailable_pet_ids]
        
        total_count = len(all_pets)
        
        # Legacy mode: if 'limit' param is provided, return all up to that limit (no pagination)
        if legacy_limit is not None:
            limit = int(legacy_limit)
            limited_pets = all_pets[:limit]
            return jsonify({
                "ok": True,
                "count": len(limited_pets),
                "total_count": total_count,
                "page": 1,
                "per_page": limit,
                "total_pages": 1,
                "pets": limited_pets
            })
        
        # Pagination
        total_pages = max(1, -(-total_count // per_page))  # ceiling division
        page = min(page, total_pages)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_pets = all_pets[start:end]
        
        return jsonify({
            "ok": True,
            "count": len(paginated_pets),
            "total_count": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "pets": paginated_pets
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

        # Notify all admin users via email
        admins = User.query.filter_by(is_admin=True).all()
        user_name = current_user.full_name or current_user.username
        for admin in admins:
            if admin.email:
                send_adoption_request_notification(
                    admin.email, user_name, current_user.email, pet_name, message
                )

        return jsonify({
            "ok": True,
            "message": "Adoption request submitted successfully!",
            "request_id": adoption_request.id
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/adoptions/<int:request_id>", methods=["PUT"])
@login_required
def update_adoption_request(request_id):
    """Update the message on a pending adoption request"""
    try:
        adoption_request = AdoptionRequest.query.get(request_id)
        if not adoption_request:
            return jsonify({"ok": False, "message": "Request not found"}), 404

        if adoption_request.user_id != current_user.id:
            return jsonify({"ok": False, "message": "Unauthorized"}), 403

        if adoption_request.status != 'pending':
            return jsonify({"ok": False, "message": "Can only edit pending requests"}), 400

        data = request.json or {}
        new_message = data.get("message", "").strip()
        if not new_message:
            return jsonify({"ok": False, "message": "Message cannot be empty"}), 400

        adoption_request.message = new_message
        db.session.commit()

        return jsonify({"ok": True, "message": "Request updated successfully"})

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
    """Get user's favorite pets with full details"""
    try:
        favorites = Favorite.query.filter_by(user_id=current_user.id).all()
        pet_ids = [fav.pet_id for fav in favorites]
        
        # Get full pet details for each favorite
        pets = []
        if engine is not None:
            for pid in pet_ids:
                pet = engine.get_pet_by_id(pid)
                if pet:
                    pets.append(pet)
        
        return jsonify({
            "ok": True,
            "count": len(pet_ids),
            "pet_ids": pet_ids,
            "pets": pets
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
        
        # Get total pets (dataset + custom — all in engine now)
        if engine:
            stats['total_pets'] = engine.get_statistics()['total_pets']
        else:
            stats['total_pets'] = 0
        
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

        # Send email notification to the user
        requester = User.query.get(adoption_request.user_id)
        if requester and requester.email and new_status in ('approved', 'rejected'):
            send_adoption_status_notification(
                requester.email,
                requester.full_name or requester.username,
                adoption_request.pet_name,
                new_status
            )
            # Also notify other rejected users if pet was approved
            if new_status == 'approved':
                for req in other_requests:
                    other_user = User.query.get(req.user_id)
                    if other_user and other_user.email:
                        send_adoption_status_notification(
                            other_user.email,
                            other_user.full_name or other_user.username,
                            req.pet_name,
                            'rejected'
                        )

        resp_message = f"Adoption request {new_status}"
        if new_status == 'approved':
            resp_message += f" and pet marked as adopted. {len(other_requests)} other pending requests were automatically rejected."

        return jsonify({
            "ok": True,
            "message": resp_message,
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
    """Add a new custom pet (auto-assigns PetID continuing from dataset)"""
    try:
        data = request.json or {}
        
        # Validate required fields (name is auto-generated)
        required_fields = ['type', 'breed', 'age_months', 'size', 'color', 'gender', 'weight_kg']
        for field in required_fields:
            if field not in data or data[field] in (None, ''):
                return jsonify({"ok": False, "message": f"Missing required field: {field}"}), 400
        
        # Auto-assign next pet_id
        next_id = CustomPet.next_pet_id()
        
        # Map energy level to numeric for KNN
        energy_map = {'High': 0, 'Low': 1, 'Moderate': 2}
        size_map = {'Large': 0, 'Medium': 1, 'Small': 2}
        health_map = {'Excellent': 0, 'Good': 1, 'Fair': 2}
        
        age_months = int(data['age_months'])
        weight_kg = float(data['weight_kg'])
        vaccinated = data.get('vaccinated', False)
        kid_friendly = data.get('kid_friendly', True)
        meat_consumption = data.get('meat_consumption', True)
        shedding_level = int(data.get('shedding_level', 2))
        energy_level = data.get('energy_level', 'Moderate')
        size = data['size']
        health_condition = data.get('health_condition', 'Good')
        
        # Admin-uploaded image (filename from /api/admin/upload-pet-image)
        image_filename = data.get('image_filename', '') or None

        custom_pet = CustomPet(
            pet_id=next_id,
            type=data['type'],
            breed=data['breed'],
            age_months=age_months,
            size=size,
            color=data['color'],
            gender=data['gender'],
            weight_kg=weight_kg,
            vaccinated=vaccinated,
            health_condition=health_condition,
            kid_friendly=kid_friendly,
            energy_level=energy_level,
            food_preference=data.get('food_preference', 'Non-Vegetarian'),
            meat_consumption=meat_consumption,
            shedding_level=shedding_level,
            has_previous_owner=data.get('has_previous_owner', False),
            days_in_shelter=int(data.get('days_in_shelter', 0)),
            description=data.get('description', ''),
            pet_characteristics=data.get('pet_characteristics', ''),
            fee=data.get('fee', 100.0),
            image_path=image_filename,
            created_by_admin_id=current_user.id
        )
        
        db.session.add(custom_pet)
        db.session.commit()
        
        # Register with recommendation engine so it appears in quiz & text search
        if engine is not None:
            # Normalize values to match KNN feature encoding
            # AgeMonths and WeightKg are raw floats in the dataset
            # Normalize them the same way the scaler expects
            pet_entry = {
                'id': next_id,
                'pet_id': next_id,
                'index': len(engine.pets_database),
                'name': custom_pet.name,
                'type': data['type'],
                'breed': data['breed'],
                'age_months': age_months,
                'color': data['color'],
                'size': size,
                'weight_kg': weight_kg,
                'vaccinated': vaccinated,
                'health_condition': health_condition,
                'days_in_shelter': int(data.get('days_in_shelter', 0)),
                'shelter_entry_date': (date.today() - timedelta(days=int(data.get('days_in_shelter', 0)))).isoformat(),
                'has_previous_owner': data.get('has_previous_owner', False),
                'gender': data['gender'],
                'description': data.get('description', f"A {energy_level.lower()} energy {data['breed']} looking for a loving home."),
                'shedding_level': shedding_level,
                'food_preference': data.get('food_preference', 'Non-Vegetarian'),
                'meat_consumption': meat_consumption,
                'kid_friendly': kid_friendly,
                'energy_level': energy_level,
                'image_url': (f"/uploads/pets/{image_filename}" if image_filename
                              else get_pet_image_url(data['breed'], data['color'], size, age_months, next_id)),
                'pet_characteristics': data.get('pet_characteristics', ''),
                'pet_details': data.get('description', ''),
                'is_custom': True,
                'raw_features': {
                    'Size': size_map.get(size, 1),
                    'EnergyLevel': energy_map.get(energy_level, 2),
                    'kid_friendliness': int(kid_friendly),
                    'Vaccinated': int(vaccinated),
                    'shedding': shedding_level,
                    'MeatConsumption': int(meat_consumption),
                    'AgeMonths': (age_months - DATASET_AGE_MEAN) / DATASET_AGE_STD,
                    'WeightKg': (weight_kg - DATASET_WEIGHT_MEAN) / DATASET_WEIGHT_STD,
                    'HealthCondition': health_map.get(health_condition, 1),
                }
            }
            engine.register_custom_pet(pet_entry)
        
        return jsonify({
            "ok": True,
            "message": f"Pet '{custom_pet.name}' (#{next_id}) added successfully",
            "pet": custom_pet.to_dict()
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/pets/<int:pet_id>", methods=["PUT"])
@admin_required
def admin_edit_pet(pet_id):
    """Edit a custom pet (only custom pets with pet_id > 1985)"""
    try:
        if pet_id <= CustomPet.DATASET_MAX_ID:
            return jsonify({"ok": False, "message": "Can only edit custom pets. Dataset pets are read-only."}), 400
        
        custom_pet = CustomPet.query.filter_by(pet_id=pet_id).first()
        if not custom_pet:
            return jsonify({"ok": False, "message": "Pet not found"}), 404
        
        # Check if pet is adopted
        adopted_pet = AdoptedPet.query.filter_by(pet_id=pet_id).first()
        if adopted_pet:
            return jsonify({"ok": False, "message": "Cannot edit an adopted pet"}), 400
        
        data = request.json or {}
        
        # Map strings to numeric for KNN
        energy_map = {'High': 0, 'Low': 1, 'Moderate': 2}
        size_map = {'Large': 0, 'Medium': 1, 'Small': 2}
        health_map = {'Excellent': 0, 'Good': 1, 'Fair': 2}
        
        # Update fields if provided
        if 'type' in data:
            custom_pet.type = data['type']
        if 'breed' in data:
            custom_pet.breed = data['breed']
        if 'age_months' in data:
            custom_pet.age_months = int(data['age_months'])
        if 'size' in data:
            custom_pet.size = data['size']
        if 'color' in data:
            custom_pet.color = data['color']
        if 'gender' in data:
            custom_pet.gender = data['gender']
        if 'weight_kg' in data:
            custom_pet.weight_kg = float(data['weight_kg'])
        if 'vaccinated' in data:
            custom_pet.vaccinated = data['vaccinated']
        if 'health_condition' in data:
            custom_pet.health_condition = data['health_condition']
        if 'kid_friendly' in data:
            custom_pet.kid_friendly = data['kid_friendly']
        if 'energy_level' in data:
            custom_pet.energy_level = data['energy_level']
        if 'food_preference' in data:
            custom_pet.food_preference = data['food_preference']
        if 'meat_consumption' in data:
            custom_pet.meat_consumption = data['meat_consumption']
        if 'shedding_level' in data:
            custom_pet.shedding_level = int(data['shedding_level'])
        if 'has_previous_owner' in data:
            custom_pet.has_previous_owner = data['has_previous_owner']
        if 'days_in_shelter' in data:
            custom_pet.days_in_shelter = int(data['days_in_shelter'])
        if 'description' in data:
            custom_pet.description = data['description']
        if 'fee' in data:
            custom_pet.fee = float(data['fee'])
        if 'image_filename' in data:
            # Delete old uploaded image if replacing
            if custom_pet.image_path and data['image_filename'] != custom_pet.image_path:
                old_path = os.path.join(UPLOAD_DIR, secure_filename(custom_pet.image_path))
                if os.path.isfile(old_path):
                    os.remove(old_path)
            custom_pet.image_path = data['image_filename'] or None
        
        db.session.commit()
        
        # Update in recommendation engine
        if engine is not None:
            pet_entry = {
                'id': pet_id,
                'pet_id': pet_id,
                'index': 0,  # will be corrected by update_custom_pet
                'name': custom_pet.name,
                'type': custom_pet.type,
                'breed': custom_pet.breed,
                'age_months': custom_pet.age_months,
                'color': custom_pet.color,
                'size': custom_pet.size,
                'weight_kg': custom_pet.weight_kg,
                'vaccinated': custom_pet.vaccinated,
                'health_condition': custom_pet.health_condition,
                'days_in_shelter': custom_pet.days_in_shelter,
                'shelter_entry_date': (date.today() - timedelta(days=custom_pet.days_in_shelter or 0)).isoformat(),
                'has_previous_owner': custom_pet.has_previous_owner,
                'gender': custom_pet.gender,
                'description': custom_pet.description or f"A {custom_pet.energy_level.lower()} energy {custom_pet.breed} looking for a loving home.",
                'shedding_level': custom_pet.shedding_level,
                'food_preference': custom_pet.food_preference,
                'meat_consumption': custom_pet.meat_consumption,
                'kid_friendly': custom_pet.kid_friendly,
                'energy_level': custom_pet.energy_level,
                'image_url': (f"/uploads/pets/{custom_pet.image_path}" if custom_pet.image_path
                              else get_pet_image_url(custom_pet.breed, custom_pet.color, custom_pet.size, custom_pet.age_months, pet_id)),
                'pet_characteristics': custom_pet.pet_characteristics or '',
                'pet_details': custom_pet.description or '',
                'is_custom': True,
                'raw_features': {
                    'Size': size_map.get(custom_pet.size, 1),
                    'EnergyLevel': energy_map.get(custom_pet.energy_level, 2),
                    'kid_friendliness': int(custom_pet.kid_friendly),
                    'Vaccinated': int(custom_pet.vaccinated),
                    'shedding': custom_pet.shedding_level,
                    'MeatConsumption': int(custom_pet.meat_consumption),
                    'AgeMonths': (custom_pet.age_months - DATASET_AGE_MEAN) / DATASET_AGE_STD,
                    'WeightKg': (custom_pet.weight_kg - DATASET_WEIGHT_MEAN) / DATASET_WEIGHT_STD,
                    'HealthCondition': health_map.get(custom_pet.health_condition, 1),
                }
            }
            engine.update_custom_pet(pet_id, pet_entry)
        
        return jsonify({
            "ok": True,
            "message": f"Pet '{custom_pet.name}' updated successfully",
            "pet": custom_pet.to_dict()
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500

@app.route("/api/admin/pets/<int:pet_id>", methods=["DELETE"])
@admin_required
def admin_delete_pet(pet_id):
    """Delete a custom pet permanently (only custom pets with pet_id > 1985)"""
    try:
        # Only allow deletion of custom pets (ID > dataset max)
        if pet_id <= CustomPet.DATASET_MAX_ID:
            return jsonify({"ok": False, "message": "Can only delete custom pets. Use Hide for dataset pets."}), 400
        
        custom_pet = CustomPet.query.filter_by(pet_id=pet_id).first()
        
        if not custom_pet:
            return jsonify({"ok": False, "message": "Pet not found"}), 404
        
        # Check if pet is adopted
        adopted_pet = AdoptedPet.query.filter_by(pet_id=pet_id).first()
        if adopted_pet:
            return jsonify({"ok": False, "message": "Cannot delete an adopted pet"}), 400
        
        pet_name = custom_pet.name
        
        # Delete associated records
        AdoptionRequest.query.filter_by(pet_id=pet_id).delete()
        Favorite.query.filter_by(pet_id=pet_id).delete()
        HiddenPet.query.filter_by(pet_id=pet_id).delete()
        
        # Remove from recommendation engine if loaded
        if engine is not None:
            engine.pets_database = [p for p in engine.pets_database if p['id'] != pet_id]
            # Note: SBERT embeddings & KNN scaled arrays keep stale rows
            # but they'll never match since the pet is gone from pets_database
        
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

def load_custom_pets_into_engine():
    """Load any custom pets from DB into the recommendation engine on startup"""
    if engine is None:
        return
    custom_pets = CustomPet.query.all()
    if not custom_pets:
        return
    
    energy_map = {'High': 0, 'Low': 1, 'Moderate': 2}
    size_map = {'Large': 0, 'Medium': 1, 'Small': 2}
    health_map = {'Excellent': 0, 'Good': 1, 'Fair': 2}
    
    for cp in custom_pets:
        pet_entry = {
            'id': cp.pet_id,
            'pet_id': cp.pet_id,
            'index': len(engine.pets_database),
            'name': cp.name,
            'type': cp.type,
            'breed': cp.breed,
            'age_months': cp.age_months,
            'color': cp.color,
            'size': cp.size,
            'weight_kg': cp.weight_kg,
            'vaccinated': cp.vaccinated,
            'health_condition': cp.health_condition,
            'days_in_shelter': cp.days_in_shelter or 0,
            'has_previous_owner': cp.has_previous_owner,
            'gender': cp.gender,
            'description': cp.description or f"A {cp.energy_level.lower()} energy {cp.breed} looking for a loving home.",
            'shedding_level': cp.shedding_level,
            'food_preference': cp.food_preference,
            'meat_consumption': cp.meat_consumption,
            'kid_friendly': cp.kid_friendly,
            'energy_level': cp.energy_level,
            'image_url': (f"/uploads/pets/{cp.image_path}" if cp.image_path
                          else get_pet_image_url(cp.breed, cp.color, cp.size, cp.age_months, cp.pet_id)),
            'pet_characteristics': cp.pet_characteristics or '',
            'pet_details': cp.description or '',
            'is_custom': True,
            'raw_features': {
                'Size': size_map.get(cp.size, 1),
                'EnergyLevel': energy_map.get(cp.energy_level, 2),
                'kid_friendliness': int(cp.kid_friendly),
                'Vaccinated': int(cp.vaccinated),
                'shedding': cp.shedding_level,
                'MeatConsumption': int(cp.meat_consumption),
                'AgeMonths': (cp.age_months - DATASET_AGE_MEAN) / DATASET_AGE_STD,
                'WeightKg': (cp.weight_kg - DATASET_WEIGHT_MEAN) / DATASET_WEIGHT_STD,
                'HealthCondition': health_map.get(cp.health_condition, 1),
            }
        }
        engine.register_custom_pet(pet_entry)
    
    print(f"✅ Loaded {len(custom_pets)} custom pet(s) into recommendation engine")


def init_database():
    """Initialize database tables and load custom pets into engine"""
    with app.app_context():
        db.create_all()
        # ── Migrate: add image_path column if missing ──
        try:
            with db.engine.connect() as conn:
                cols = [r[1] for r in conn.execute(db.text("PRAGMA table_info(custom_pets)"))]
                if 'image_path' not in cols:
                    conn.execute(db.text("ALTER TABLE custom_pets ADD COLUMN image_path VARCHAR(300)"))
                    conn.commit()
                    print("✅ Migrated: added image_path column to custom_pets")
        except Exception as e:
            print(f"⚠️  Migration note: {e}")
        print("✅ Database tables ready!")
        load_custom_pets_into_engine()

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

        print("🚀 Starting Flask server on http://localhost:5000")
        print("="*60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
