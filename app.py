"""
app.py — Crop Disease Detection System
Features: Login, Register, Predict, History, Dark/Light Theme, Odia Language
Run: python app.py
"""

import os, json, io, sys, logging, functools
from datetime import datetime
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
HISTORY_PATH  = os.path.join(BASE_DIR, 'data', 'history.json')
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
        log.warning("Model not found at %s.", MODEL_PATH)
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

# ── Severity logic ────────────────────────────────────────────────────────────
def get_severity(confidence, is_healthy_plant):
    if is_healthy_plant:
        return {'level':'Healthy','level_od':'ସୁସ୍ଥ','color':'green','icon':'✅',
                'advice':'Your plant is healthy. Keep maintaining good farming practices.',
                'advice_od':'ଆପଣଙ୍କ ଗଛ ସୁସ୍ଥ ଅଛି। ଉତ୍ତମ କୃଷି ଅଭ୍ୟାସ ଜାରି ରଖନ୍ତୁ।'}
    if confidence >= 90:
        return {'level':'Critical','level_od':'ଗୁରୁତର','color':'red','icon':'🔴',
                'advice':'URGENT! Disease is very severe. Immediately isolate the plant and apply treatment.',
                'advice_od':'ଜରୁରୀ! ରୋଗ ଅତ୍ୟନ୍ତ ଗୁରୁତର। ତୁରନ୍ତ ଗଛ ଅଲଗା କରି ଚିକିତ୍ସା ଲଗାନ୍ତୁ।'}
    if confidence >= 75:
        return {'level':'Severe','level_od':'ଭୀଷଣ','color':'orange','icon':'🟠',
                'advice':'Disease is spreading fast. Apply recommended treatment immediately and monitor daily.',
                'advice_od':'ରୋଗ ଦ୍ରୁତ ବ୍ୟାପୁଛି। ଏବେ ଚିକିତ୍ସା ଲଗାନ୍ତୁ ଏବଂ ପ୍ରତିଦିନ ଦେଖନ୍ତୁ।'}
    if confidence >= 60:
        return {'level':'Moderate','level_od':'ମଧ୍ୟମ','color':'yellow','icon':'🟡',
                'advice':'Disease at moderate level. Start treatment soon to prevent spreading.',
                'advice_od':'ମଧ୍ୟମ ସ୍ତରରେ ରୋଗ। ବ୍ୟାପିବା ଆଗରୁ ଚିକିତ୍ସା ଆରମ୍ଭ କରନ୍ତୁ।'}
    return {'level':'Mild','level_od':'ହାଲୁକା','color':'lime','icon':'🟢',
            'advice':'Early stage disease. Take preventive measures now before it gets worse.',
            'advice_od':'ପ୍ରାଥମିକ ଅବସ୍ଥା ରୋଗ। ଅଧିକ ଖରାପ ହେବା ଆଗରୁ ସତର୍କ ପଦକ୍ଷେପ ନିଅନ୍ତୁ।'}

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

# ── History system ────────────────────────────────────────────────────────────
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return {}
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def add_to_history(username, prediction_data):
    history = load_history()
    if username not in history:
        history[username] = []
    entry = {
        'id':           len(history[username]) + 1,
        'timestamp':    datetime.now().strftime('%d %b %Y, %I:%M %p'),
        'disease_en':   prediction_data.get('disease_name_en', ''),
        'disease_od':   prediction_data.get('disease_name', ''),
        'confidence':   prediction_data.get('confidence', 0),
        'is_healthy':   prediction_data.get('is_healthy', False),
        'remedy':       prediction_data.get('remedy', ''),
        'language':     prediction_data.get('language', 'English'),
    }
    history[username].insert(0, entry)  # newest first
    # Keep only last 50 records per user
    history[username] = history[username][:50]
    save_history(history)

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
        flash('Invalid username or password.')
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

@app.route('/history')
@login_required
def history():
    username = session['user']
    all_history = load_history()
    user_history = all_history.get(username, [])
    return render_template('history.html', history=user_history, username=username)

@app.route('/history/clear', methods=['POST'])
@login_required
def clear_history():
    username = session['user']
    all_history = load_history()
    all_history[username] = []
    save_history(all_history)
    flash('History cleared successfully!', 'success')
    return redirect(url_for('history'))

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
    conf_pct    = round(confidence * 100, 2)
    severity    = get_severity(conf_pct, healthy)
    top3_indices = np.argsort(predictions[0])[::-1][:3]
    top3 = [
        {'class': class_names[i] if i < len(class_names) else 'Unknown',
         'name': get_disease_name(class_names[i] if i < len(class_names) else '', lang),
         'confidence': round(float(predictions[0][i]) * 100, 2)}
        for i in top3_indices
    ]

    result = {
        'class_key':        class_key,
        'disease_name_en':  disease_en,
        'disease_name':     disease_loc,
        'confidence':       conf_pct,
        'remedy':           remedy,
        'is_healthy':       healthy,
        'language':         'English' if lang == 'en' else 'ଓଡ଼ିଆ',
        'lang_code':        lang,
        'top3':             top3,
        'severity_level':   severity['level'] if lang == 'en' else severity['level_od'],
        'severity_color':   severity['color'],
        'severity_icon':    severity['icon'],
        'severity_advice':  severity['advice'] if lang == 'en' else severity['advice_od'],
    }

    # Save to history
    add_to_history(session['user'], result)

    return jsonify(result)

@app.route('/health')
def health():
    model, _ = _load_model()
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("Starting Crop Disease Detection server …")
    log.info("Open http://127.0.0.1:5000 in your browser.")
    _load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
