from config import Config
from models import db, User
from flask_migrate import Migrate
from routes.auth import auth_bp
from routes.user import user_bp
from routes.collector import collector_bp
from routes.pickups import pickups_bp
import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json
from flask_login import LoginManager

app = Flask(__name__, instance_relative_config=True)
app.config.from_object(Config)
db.init_app(app)
migrate = Migrate(app, db)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app.config['UPLOAD_FOLDER'] = 'static/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(collector_bp)
app.register_blueprint(pickups_bp)

# Load models
scrap_type_model = load_model("scrap_type_model.h5")
condition_model = load_model("condition_model_binary.h5")

# Load scrap type class indices
with open("C:/Users/rupa1/OneDrive/Desktop/New LSM App/model_python/class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# Fine ➝ Broad category map
category_map = {
    'aluminum': ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'steel_food_cans'],
    'paper': ['magazines', 'newspaper', 'office_paper', 'paper_cups'],
    'cardboard': ['cardboard_boxes', 'cardboard_packaging'],
    'plastic': ['disposable_plastic_cutlery', 'plastic_cup_lids', 'plastic_detergent_bottles', 
                'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 
                'plastic_straws', 'plastic_trash_bags', 'plastic_water_bottles', 
                'styrofoam_cups', 'styrofoam_food_containers'],
    'glass': ['glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars'],
    'non_recyclable': ['clothing', 'coffee_grounds', 'eggshells', 'food_waste', 'shoes', 'tea_bags']
}

# Base price/kg for each broad category
base_prices = {
    'aluminum': 25,
    'plastic': 15,
    'paper': 5,
    'cardboard': 6,
    'glass': 10,
    'non_recyclable': 2,
    'unknown': 1,
}

# Condition multipliers
condition_multipliers = {
    'Good': 1.0,
    'Poor': 0.5
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Utility functions ---
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_scrap_type(img_path):
    img_array = preprocess_image(img_path)
    prediction = scrap_type_model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = idx_to_class[predicted_index]
    confidence = prediction[0][predicted_index]
    return predicted_class, confidence

def get_broad_category(predicted_class):
    for category, items in category_map.items():
        if predicted_class in items:
            return category
    return "unknown"

def predict_condition(img_path):
    img_array = preprocess_image(img_path)
    prediction = condition_model.predict(img_array)
    return "Good" if prediction[0][0] >= 0.5 else "Poor"

def estimate_price(category, condition, weight_kg):
    base_price = base_prices.get(category.lower(), 1)
    multiplier = condition_multipliers.get(condition, 0.5)
    return round(base_price * multiplier * weight_kg, 2)



@app.route('/estimate', methods=['POST'])
def estimate_scrap_value():
    image_file = request.files.get('image')
    weight = request.form.get('weight', type=float)

    if not image_file or weight is None:
        return render_template('dashboard_user.html', error='Image and weight are required')

    # Save image temporarily
    img_path = os.path.join("static", image_file.filename)
    image_file.save(img_path)

    # Predictions
    predicted_class, confidence = predict_scrap_type(img_path)
    broad_category = get_broad_category(predicted_class)
    condition = predict_condition(img_path)
    price = estimate_price(broad_category, condition, weight)

    # Cleanup image
    os.remove(img_path)

    return render_template('dashboard_user.html', 
        estimation_results={
            'fine_grained_class': predicted_class,
            'confidence': round(float(confidence), 2),
            'broad_category': broad_category,
            'condition': condition,
            'weight_kg': weight,
            'estimated_price': price
        })


@app.route('/')
def home():
    return render_template('index.html')

# CLI command to create DB
@app.cli.command("create-db")
def create_db():
    with app.app_context():
        db.create_all()
        print("✅ Database created successfully")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT not set
    app.run(host='0.0.0.0', port=port)




