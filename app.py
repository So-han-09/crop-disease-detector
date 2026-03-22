"""
app.py — Plant Disease Detection Flask App
Includes: User Login/Register, Dark/Light Theme, Modern UI
Run: python app.py
"""

import os, json, io, sys, logging, functools
import numpy as np
from PIL import Image
from flask import (Flask, request, render_template, jsonify,
                   redirect, url_for, session, flash)
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.translator import get_disease_name, get_ui_text, is_healthy
from utils.remedies   import get_remedy

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, 'models')
MODEL_PATH    = os.path.join(MODEL_DIR, 'plant_model.keras')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(MODEL_DIR, 'plant_model.h5')
CLASSES_PATH  = os.path.join(MODEL_DIR, 'class_names.json')
USERS_PATH    = os.path.join(BASE_DIR, 'data', 'users.json')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXT   = {'png','jpg','jpeg','bmp','webp'}
IMG_SIZE      = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'plant-disease-satya-2024-secret'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Model loading ─────────────────────────────────────────────────────────────
_model = None
_class_names = None

def _load_model():
    global _model, _class_names
    if _model is not None:
        return _model, _class_names
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
            _class_names = json.load(f)
        log.info("Loaded %d class names", len(_class_names))
    else:
        _class_names = [
            "Pepper__bell___Bacterial_spot","Pepper__bell___healthy",
            "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
            "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
            "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato__Tomato_mosaic_virus","Tomato_healthy",
        ]
    if not os.path.exists(MODEL_PATH):
        log.warning("Model not found at %s. Run python train_model.py first.", MODEL_PATH)
        return None, _class_names
    import tensorflow as tf
    log.info("Loading model from %s …", MODEL_PATH)
    _model = tf.keras.models.load_model(MODEL_PATH)
    log.info("Model loaded successfully.")
    return _model, _class_names

def _preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ── User system ───────────────────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_PATH, 'w') as f:
        json.dump(users, f, indent=2)

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        users = load_users()
        if username in users and check_password_hash(users[username]['password'], password):
            session['user'] = username
            session['fullname'] = users[username].get('fullname', username)
            return redirect(url_for('index'))
        flash('Invalid username or password. Please try again.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Username and password are required.')
            return render_template('register.html')
        users = load_users()
        if username in users:
            flash('Username already exists. Please choose another.')
            return render_template('register.html')
        users[username] = {
            'fullname': fullname,
            'password': generate_password_hash(password)
        }
        save_users(users)
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Main Routes ───────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    lang = request.form.get('lang', 'en').strip().lower()
    if lang not in ('en', 'od'):
        lang = 'en'
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not _allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400
    file_bytes = file.read()
    model, class_names = _load_model()
    if model is None:
        return jsonify({'error': 'Model not found. Run python train_model.py first.'}), 503
    try:
        img_array   = _preprocess_image(file_bytes)
        predictions = model.predict(img_array, verbose=0)
    except Exception as exc:
        return jsonify({'error': f'Prediction error: {str(exc)}'}), 500
    pred_idx   = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx])
    class_key  = class_names[pred_idx] if pred_idx < len(class_names) else 'Unknown'
    disease_en  = get_disease_name(class_key, 'en')
    disease_loc = get_disease_name(class_key, lang)
    remedy      = get_remedy(class_key, lang)
    healthy     = is_healthy(class_key)
    top3_indices = np.argsort(predictions[0])[::-1][:3]
    top3 = [
        {'class': class_names[i] if i < len(class_names) else 'Unknown',
         'name': get_disease_name(class_names[i] if i < len(class_names) else '', lang),
         'confidence': round(float(predictions[0][i]) * 100, 2)}
        for i in top3_indices
    ]
    return jsonify({
        'class_key': class_key, 'disease_name_en': disease_en,
        'disease_name': disease_loc, 'confidence': round(confidence * 100, 2),
        'remedy': remedy, 'is_healthy': healthy,
        'language': 'English' if lang == 'en' else 'ଓଡ଼ିଆ',
        'lang_code': lang, 'top3': top3,
    })

@app.route('/health')
def health():
    model, _ = _load_model()
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("Starting Plant Disease Detection server …")
    log.info("Open http://127.0.0.1:5000 in your browser.")
    _load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
