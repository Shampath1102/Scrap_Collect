from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)  # Add this line
    password_hash = db.Column(db.String(200))
    address = db.Column(db.String(200))
    role = db.Column(db.String(20))  # 'user' or 'collector'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.name} ({self.role})>"

class Pickup(db.Model):
    __tablename__ = 'pickup'  # Changed from 'pickups'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Changed from 'users.id'
    scrap_type = db.Column(db.String(100), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    address = db.Column(db.String(200), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(50), default='Pending')
    collector_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # Changed from 'users.id'
    estimated_price = db.Column(db.Float, nullable=True)

    user = db.relationship('User', foreign_keys=[user_id], backref='user_pickups')
    collector = db.relationship('User', foreign_keys=[collector_id], backref='assigned_pickups')

    def __repr__(self):
        return f"<Pickup #{self.id} - {self.status}>"
