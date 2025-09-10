from collections import Counter
import numpy as np
import math

# Make sure CHAR_POOL is defined globally (adjust if needed)
CHAR_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")

# === Required Helper Functions ===

def calculate_entropy(text):
    if not text:
        return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)

def ascii_stats(text):
    values = [ord(c) for c in text]
    if not values:
        return {"ascii_mean": 0, "ascii_std": 0, "ascii_min": 0, "ascii_max": 0}
    return {
        "ascii_mean": np.mean(values),
        "ascii_std": np.std(values),
        "ascii_min": np.min(values),
        "ascii_max": np.max(values),
    }

def char_type_ratios(text):
    total = len(text) if text else 1
    return {
        "digit_ratio": sum(c.isdigit() for c in text) / total,
        "alpha_ratio": sum(c.isalpha() for c in text) / total,
        "symbol_ratio": sum(not c.isalnum() for c in text) / total,
    }

def base64_specifics(text):
    return {
        "equals_count": text.count('='),
        "plus_count": text.count('+'),
        "slash_count": text.count('/'),
        "equals_ratio": text.count('=') / len(text) if len(text) > 0 else 0,
    }

def num_unique_chars(text):
    return len(set(text))

# Dummy default versions (replace with your actual computation)
def get_ic(text): return 0
def get_mic(text): return 0
def get_mka(text): return 0
def get_dic(text): return 0
def get_edi(text): return 0
def get_lr(text): return 0
def get_ldi(text): return 0
def get_sdd(text): return 0
def get_nomor(text): return 0
def get_rdi(text): return 0
def chi_square_stat(text): return 0

# === ✅ MAIN EXTRACT FUNCTION ===

def extract_features(text):
    counter = Counter(text)
    features = {}

    # 1. Character frequency features
    for ch in CHAR_POOL:
        features[f'freq_{ch}'] = counter.get(ch, 0) / len(text) if text else 0

    # 2. General features
    features['length'] = len(text)
    features['unique_chars'] = num_unique_chars(text)
    features['entropy'] = calculate_entropy(text)

    # 3. ASCII statistics
    ascii = ascii_stats(text)
    features['ascii_mean'] = ascii["ascii_mean"]
    features['ascii_std'] = ascii["ascii_std"]
    features['ascii_min'] = ascii["ascii_min"]
    features['ascii_max'] = ascii["ascii_max"]

    # 4. Char types
    char_ratios = char_type_ratios(text)
    features['digit_ratio'] = char_ratios["digit_ratio"]
    features['alpha_ratio'] = char_ratios["alpha_ratio"]
    features['symbol_ratio'] = char_ratios["symbol_ratio"]

    # 5. Base64 characteristics
    b64 = base64_specifics(text)
    features['equals_count'] = b64["equals_count"]
    features['plus_count'] = b64["plus_count"]
    features['slash_count'] = b64["slash_count"]
    features['equals_ratio'] = b64["equals_ratio"]

    # 6. Cipher-based statistical features
    features['IC'] = 1000 * get_ic(text)
    features['MIC'] = 1000 * get_mic(text)
    features['MKA'] = 1000 * get_mka(text)
    features['DIC'] = 10000 * get_dic(text)
    features['EDI'] = get_edi(text)
    features['LR'] = get_lr(text)
    features['LDI'] = get_ldi(text)
    features['SDD'] = get_sdd(text)
    features['NOMOR'] = get_nomor(text)
    features['RDI'] = get_rdi(text)
    features['ChiSquare'] = chi_square_stat(text)

    return features
