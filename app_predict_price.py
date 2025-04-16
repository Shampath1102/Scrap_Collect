import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'static/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load models
scrap_type_model = load_model("C:/Users/rupa1/OneDrive/Desktop/New LSM App/Frontend/scrap_type_model.h5")
condition_model = load_model("C:/Users/rupa1/OneDrive/Desktop/New LSM App/Frontend/condition_model_binary.h5")

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimate', methods=['POST'])
def estimate_scrap_value():
    image_file = request.files.get('image')
    weight = request.form.get('weight', type=float)

    if not image_file or weight is None:
        return jsonify({'error': 'Image and weight are required'}), 400

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

    return jsonify({
    'fine_grained_class': str(predicted_class),
    'confidence': round(float(confidence), 2),
    'broad_category': str(broad_category),
    'condition': str(condition),
    'weight_kg': float(weight),
    'estimated_price': float(price)
})


if __name__ == '__main__':
    app.run(debug=True)
