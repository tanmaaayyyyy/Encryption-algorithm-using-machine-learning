import numpy as np
import pandas as pd
from collections import Counter
import string
import math

# -----------------------------------------
# 📈 Reference tables (must be declared)
# -----------------------------------------

# Relative frequency of English letters
rel_freq = {
    'A': .082, 'B': .015, 'C': .028, 'D': .043, 'E': .13, 'F': .022,
    'G': .02, 'H': .061, 'I': .07, 'J': .0015, 'K': .0077, 'L': .04,
    'M': .024, 'N': .067, 'O': .075, 'P': .019, 'Q': .00095, 'R': .06,
    'S': .063, 'T': .091, 'U': .028, 'V': .0098, 'W': .024, 'X': .0015,
    'Y': .02, 'Z': .00074
}

# Frequency rank (1 to 26) based on standard English
english_freq_list = np.argsort([-v for v in rel_freq.values()]) + 1

# Log di-graph index matrix (26x26)
logdi = np.random.rand(26, 26)  # dummy values for now
sdd = np.random.rand(26, 26)    # dummy values for now

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


CHAR_POOL = list(string.ascii_uppercase + string.digits + string.punctuation + ' ')
CHAR_POOL = [ch for ch in CHAR_POOL if ch.strip() or ch == ' ']  # remove empty/invalid




def extract_features(ciphertext):
    text = ciphertext.upper().replace(" ", "")  # normalize
    counter = Counter(text)

    # Feature dict
    feats = {}

    # Frequency features
    for ch in CHAR_POOL:
        if ch == " ":
            continue
        feats[f'freq_{ch}'] = counter.get(ch, 0) / len(text) if len(text) > 0 else 0



    # General stats
    entropy = calculate_entropy(text)
    ascii = ascii_stats(text)
    char_ratios = char_type_ratios(text)
    b64 = base64_specifics(text)

    # Add other core features
    feats['length'] = len(text)
    feats['unique_chars'] = num_unique_chars(text)
    feats['entropy'] = entropy
    feats['ascii_mean'] = ascii["ascii_mean"]
    feats['ascii_std'] = ascii["ascii_std"]
    feats['ascii_min'] = ascii["ascii_min"]
    feats['ascii_max'] = ascii["ascii_max"]
    feats['digit_ratio'] = char_ratios["digit_ratio"]
    feats['alpha_ratio'] = char_ratios["alpha_ratio"]
    feats['symbol_ratio'] = char_ratios["symbol_ratio"]
    feats['equals_count'] = b64["equals_count"]
    feats['plus_count'] = b64["plus_count"]
    feats['slash_count'] = b64["slash_count"]
    feats['equals_ratio'] = b64["equals_ratio"]

    # Cipher-specific features
    feats['IC'] = 1000 * get_ic(text)
    feats['MIC'] = 1000 * get_mic(text)
    feats['MKA'] = 1000 * get_mka(text)
    feats['DIC'] = 10000 * get_dic(text)
    feats['EDI'] = get_edi(text)
    feats['LR'] = get_lr(text)
    feats['SDD'] = get_sdd(text)
    feats['NOMOR'] = get_nomor(text)
    feats['ChiSquare'] = chi_square_stat(text)

    return feats

# ✅ Bonus: Batch feature extractor for testing
def extract_features_df(cipher_list):
    return pd.DataFrame([extract_features(cipher) for cipher in cipher_list])

extract_features("hi")