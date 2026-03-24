"""
Database Models for Pet Adoption System
- User authentication
- Adoption requests tracking
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from pet_image_mapper import get_pet_image_url

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    adoption_requests = db.relationship('AdoptionRequest', backref='user', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'phone': self.phone,
            'address': self.address,
            'is_admin': self.is_admin,
            'email_verified': self.email_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


class AdoptionRequest(db.Model):
    """Adoption request tracking"""
    __tablename__ = 'adoption_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pet_id = db.Column(db.Integer, nullable=False)
    pet_name = db.Column(db.String(100))
    pet_type = db.Column(db.String(50))
    message = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_name': self.user.full_name or self.user.username,
            'user_email': self.user.email,
            'pet_id': self.pet_id,
            'pet_name': self.pet_name,
            'pet_type': self.pet_type,
            'message': self.message,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<AdoptionRequest {self.id} - {self.pet_name}>'


class Favorite(db.Model):
    """User's favorite pets"""
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pet_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint: one user can favorite a pet only once
    __table_args__ = (db.UniqueConstraint('user_id', 'pet_id', name='unique_favorite'),)
    
    def __repr__(self):
        return f'<Favorite {self.user_id} - {self.pet_id}>'


class AdoptedPet(db.Model):
    """Track adopted pets"""
    __tablename__ = 'adopted_pets'
    
    id = db.Column(db.Integer, primary_key=True)
    pet_id = db.Column(db.Integer, unique=True, nullable=False)
    pet_name = db.Column(db.String(100))
    pet_type = db.Column(db.String(50))
    adopted_by_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    adoption_request_id = db.Column(db.Integer, db.ForeignKey('adoption_requests.id'), nullable=False)
    adopted_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    adopter = db.relationship('User', backref='adopted_pets', lazy=True)
    adoption_request = db.relationship('AdoptionRequest', backref='adopted_pet', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'pet_id': self.pet_id,
            'pet_name': self.pet_name,
            'pet_type': self.pet_type,
            'adopted_by_user_id': self.adopted_by_user_id,
            'adopted_by': self.adopter.username,
            'adopter_full_name': self.adopter.full_name or self.adopter.username,
            'adopter_email': self.adopter.email,
            'adopter_phone': self.adopter.phone or '',
            'adopter_address': self.adopter.address or '',
            'adoption_request_id': self.adoption_request_id,
            'adoption_message': self.adoption_request.message if self.adoption_request else '',
            'adopted_at': self.adopted_at.isoformat() if self.adopted_at else None
        }
    
    def __repr__(self):
        return f'<AdoptedPet {self.pet_id} - {self.pet_name}>'


class HiddenPet(db.Model):
    """Track pets that are hidden/deactivated by admin"""
    __tablename__ = 'hidden_pets'
    
    id = db.Column(db.Integer, primary_key=True)
    pet_id = db.Column(db.Integer, unique=True, nullable=False)
    reason = db.Column(db.String(200))
    hidden_at = db.Column(db.DateTime, default=datetime.utcnow)
    hidden_by_admin_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    admin = db.relationship('User', backref='hidden_pets', lazy=True)
    
    def __repr__(self):
        return f'<HiddenPet {self.pet_id}>'


class CustomPet(db.Model):
    """Custom pets added by admin — IDs continue from max dataset PetID (1985)"""
    __tablename__ = 'custom_pets'
    
    # Dataset max PetID — new custom pets start from this + 1
    DATASET_MAX_ID = 1985
    
    id = db.Column(db.Integer, primary_key=True)
    pet_id = db.Column(db.Integer, unique=True, nullable=False)  # Globally unique, continues from 1986+
    type = db.Column(db.String(50), nullable=False)
    breed = db.Column(db.String(100), nullable=False)
    age_months = db.Column(db.Integer, nullable=False)        # Total months (matches dataset format)
    size = db.Column(db.String(20), nullable=False)
    color = db.Column(db.String(50), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)
    vaccinated = db.Column(db.Boolean, default=False)
    health_condition = db.Column(db.String(50), default='Good')
    kid_friendly = db.Column(db.Boolean, default=True)
    energy_level = db.Column(db.String(20), default='Moderate')
    food_preference = db.Column(db.Text, default='Non-Vegetarian')
    meat_consumption = db.Column(db.Boolean, default=True)
    shedding_level = db.Column(db.Integer, default=2)           # 0-5
    has_previous_owner = db.Column(db.Boolean, default=False)
    days_in_shelter = db.Column(db.Integer, default=0)
    description = db.Column(db.Text)
    pet_characteristics = db.Column(db.Text)
    fee = db.Column(db.Float, default=100.0)
    image_path = db.Column(db.String(300))  # Admin-uploaded image path (relative to uploads/pets/)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by_admin_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    admin = db.relationship('User', backref='custom_pets', lazy=True)
    
    @staticmethod
    def next_pet_id():
        """Get the next available pet_id (continues from dataset max)"""
        last = db.session.query(db.func.max(CustomPet.pet_id)).scalar()
        return max(CustomPet.DATASET_MAX_ID, last or CustomPet.DATASET_MAX_ID) + 1
    
    @property
    def name(self):
        """Auto-generated name: 'Breed #PetID'"""
        return f"{self.breed} #{self.pet_id}"
    
    def to_dict(self):
        """Convert to dictionary matching the format from CSV pets exactly"""
        return {
            'pet_id': self.pet_id,
            'id': self.pet_id,
            'name': self.name,
            'type': self.type,
            'breed': self.breed,
            'age_months': self.age_months,
            'age_years': self.age_months // 12,
            'age_remaining_months': self.age_months % 12,
            'size': self.size,
            'color': self.color,
            'gender': self.gender,
            'weight_kg': self.weight_kg,
            'vaccinated': self.vaccinated,
            'health': self.health_condition,
            'health_condition': self.health_condition,
            'kid_friendly': self.kid_friendly,
            'energy_level': self.energy_level,
            'food_preference': self.food_preference,
            'meat_consumption': self.meat_consumption,
            'shedding_level': self.shedding_level,
            'has_previous_owner': self.has_previous_owner,
            'days_in_shelter': self.days_in_shelter,
            'pet_characteristics': self.pet_characteristics or '',
            'description': self.description or f"A {self.energy_level.lower()} energy {self.breed} looking for a loving home.",
            'fee': self.fee,
            'is_custom': True,
            'image_url': (f"/uploads/pets/{self.image_path}" if self.image_path
                          else get_pet_image_url(self.breed, self.color, self.size, self.age_months, self.pet_id))
        }
    
    def __repr__(self):
        return f'<CustomPet #{self.pet_id} {self.breed}>'

