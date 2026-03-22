"""
translator.py - Translation module for English <-> Odia language support
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load Odia translation file once at module load
_odia_translations = None

def _load_translations():
    global _odia_translations
    if _odia_translations is None:
        path = os.path.join(BASE_DIR, 'translations', 'odia.json')
        with open(path, 'r', encoding='utf-8') as f:
            _odia_translations = json.load(f)
    return _odia_translations


def get_disease_name(class_name: str, lang: str = 'en') -> str:
    """Return the disease name in the requested language."""
    if lang == 'od':
        translations = _load_translations()
        return translations['disease_names'].get(class_name, class_name)
    # English: prettify the class name
    return _prettify(class_name)


def get_ui_text(key: str, lang: str = 'en') -> str:
    """Return UI string in the requested language."""
    if lang == 'od':
        translations = _load_translations()
        return translations['ui'].get(key, key)
    # English defaults
    _en_ui = {
        'title': 'Plant Disease Detection',
        'subtitle': 'AI-powered crop protection system',
        'upload_prompt': 'Upload a leaf image for analysis',
        'upload_btn': 'Choose File',
        'analyze_btn': 'Analyze Plant',
        'disease_label': 'Disease',
        'confidence_label': 'Confidence',
        'remedy_label': 'Recommended Remedy',
        'language_label': 'Language',
        'healthy_msg': 'Your plant is healthy!',
        'error_no_file': 'Please upload an image first',
        'processing': 'Analyzing...',
    }
    return _en_ui.get(key, key)


def is_healthy(class_name: str) -> bool:
    """Return True if the class represents a healthy plant."""
    return 'healthy' in class_name.lower()


def _prettify(class_name: str) -> str:
    """Convert PlantVillage class name to a human-readable English string."""
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').replace(',', '').strip()
    if len(parts) > 1:
        condition = parts[1].replace('_', ' ').strip()
        return f"{plant} - {condition}"
    return plant
