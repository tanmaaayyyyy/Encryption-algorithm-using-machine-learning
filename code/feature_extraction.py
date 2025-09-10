import pandas as pd
import numpy as np
import os
import math
import string
from collections import Counter

# -----------------------------------------
# 🧩 Feature extraction dependencies
# -----------------------------------------

CHAR_POOL = string.ascii_letters + string.digits + string.punctuation

rel_freq = {
    'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
    'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
    'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
    'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
    'U': 0.07258, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974, 'Z': 0.00074
}

logdi = np.zeros((26, 26))  # Replace with real values if available
sdd = np.zeros((26, 26))    # Replace with real values if available

english_freq_list = np.array([4, 19, 0, 14, 8, 13, 18, 17, 7, 11, 3, 20,
                               2, 12, 6, 5, 24, 15, 22, 1, 21, 10, 23, 9, 25, 16]) + 1
LETTERS = list(string.ascii_uppercase)

# -----------------------------------------
# 📊 Feature functions
# -----------------------------------------

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

def get_ic(text):
    freqs = Counter(text)
    L = len(text)
    return sum(f * (f - 1) for f in freqs.values()) / (L * (L - 1)) if L > 1 else 0

def gather_letters(text, start, period):
    return ''.join(text[i] for i in range(start, len(text), period))

def get_mic(text):
    return max(
        np.mean([get_ic(gather_letters(text, s, p)) for s in range(p)])
        for p in range(1, min(16, len(text)))
    )

def get_mka(text):
    return max(
        sum(text[i] == text[i + p] for i in range(len(text) - p)) / (len(text) - p)
        if len(text) > p else 0
        for p in range(1, 16)
    )

def get_dic(text):
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    freq = Counter(bigrams)
    total = len(bigrams)
    return sum(f * (f - 1) for f in freq.values()) / (total * (total - 1)) if total > 1 else 0

def get_edi(text):
    if len(text) <= 2:
        return 0
    bigrams = [text[i:i+2] for i in range(0, len(text) - 1, 2)]
    freq = Counter(bigrams)
    denominator = len(text) * (len(text) - 2)
    if denominator == 0:
        return 0
    return 4 * sum(f * (f - 1) for f in freq.values()) / denominator


def get_lr(text):
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    freq = Counter(trigrams)
    return 1000 * math.sqrt(sum(f - 1 for f in freq.values())) / len(text)

def get_ldi(text):
    indices = [
        (ord(text[i]) - 65, ord(text[i + 1]) - 65)
        for i in range(len(text) - 1)
        if 'A' <= text[i] <= 'Z' and 'A' <= text[i + 1] <= 'Z'
    ]
    return 100 * np.mean([logdi[a][b] for a, b in indices]) if indices else 0

def get_sdd(text):
    indices = [
        (ord(text[i]) - 65, ord(text[i + 1]) - 65)
        for i in range(len(text) - 1)
        if 'A' <= text[i] <= 'Z' and 'A' <= text[i + 1] <= 'Z'
    ]
    return 100 * np.mean([sdd[a][b] for a, b in indices]) if indices else 0

def get_rdi(text):
    if len(text) < 2 or len(text) % 2 != 0:
        return 0
    indices = [
        (ord(text[i + 1]) - 65, ord(text[i]) - 65)
        for i in range(len(text) - 1)
        if 'A' <= text[i] <= 'Z' and 'A' <= text[i + 1] <= 'Z'
    ]
    return 100 * np.mean([logdi[a][b] for a, b in indices]) if indices else 0


def get_nomor(text):
    letter_counts = Counter(text)
    freq_array = np.zeros(26)
    for letter, count in letter_counts.items():
        if letter.upper() in rel_freq:
            freq_array[ord(letter.upper()) - 65] = count
    freq_order = np.argsort(-freq_array) + 1
    return sum(abs(english_freq_list[:20] - freq_order[:20]))

def chi_square_stat(text):
    N = len(text)
    if N == 0:
        return 0
    freq = Counter(text)
    chi_sq = 0
    for c in rel_freq:
        observed = freq.get(c, 0)
        expected = rel_freq[c] * N
        if expected > 0:
            chi_sq += (observed - expected) ** 2 / expected
    return chi_sq

# -----------------------------------------
# 🧠 Master Feature Extractor
# -----------------------------------------
def extract_features(text):
    counter = Counter(text)
    features = {}

    # Character frequency
    for ch in CHAR_POOL:
        features[f'freq_{ch}'] = counter.get(ch, 0) / len(text) if text else 0

    features['length'] = len(text)
    features['unique_chars'] = num_unique_chars(text)
    features['entropy'] = calculate_entropy(text)

    features.update(ascii_stats(text))
    features.update(char_type_ratios(text))
    features.update(base64_specifics(text))

    # Cipher-specific statistical features
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

# -----------------------------------------
# 📂 Load dataset and apply feature extraction
# -----------------------------------------
input_csv_path = os.path.join("dataset", r"F:\minor_project2\dataset\encryption_dataset_404k.csv")         # <- Your input dataset
output_csv_path = os.path.join("dataset", r"F:\minor_project2\dataset\encrypted_features.csv")       # <- Output with features

# Load dataset
df = pd.read_csv(input_csv_path)
if 'text' in df.columns:
    df['ciphertext'] = df['text']

# Extract features row-wise
feature_rows = []
for _, row in df.iterrows():
    feats = extract_features(str(row['ciphertext']).upper())  # convert to uppercase for consistency
    feats['label'] = row['label']  # Keep original label
    feature_rows.append(feats)

# Save result
feature_df = pd.DataFrame(feature_rows)
feature_df.to_csv(output_csv_path, index=False)

print(f"Feature extraction complete. Saved to: {output_csv_path}")