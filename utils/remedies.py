"""
remedies.py - Remedies lookup module for plant diseases
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_remedies_data = None


def _load_remedies():
    global _remedies_data
    if _remedies_data is None:
        path = os.path.join(BASE_DIR, 'data', 'remedies.json')
        with open(path, 'r', encoding='utf-8') as f:
            _remedies_data = json.load(f)
    return _remedies_data


def get_remedy(class_name: str, lang: str = 'en') -> str:
    """Return the remedy for a given disease class in the requested language."""
    remedies = _load_remedies()
    entry = remedies.get(class_name)
    if entry is None:
        if lang == 'od':
            return "ଏହି ରୋଗ ପାଇଁ ଉପଚାର ତଥ୍ୟ ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି ଏକ ଉଦ୍ଭିଦ ବିଶେଷଜ୍ଞଙ୍କ ସହ ପରାମର୍ଶ କରନ୍ତୁ।"
        return "Remedy data not available for this disease. Please consult a plant pathologist."
    return entry.get(lang, entry.get('en', 'No remedy available.'))


def get_all_classes() -> list:
    """Return all disease class names."""
    remedies = _load_remedies()
    return list(remedies.keys())
