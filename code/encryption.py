import os
import pandas as pd
import random
import base64

from Crypto.Cipher import AES, ARC4
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad  # ✅ Use correct pad function

# 📌 Caesar Cipher Function
def caesar_encrypt(text, shift=3):
    result = ''
    for char in text:
        if char.isalpha():
            shift_base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - shift_base + shift) % 26 + shift_base)
        else:
            result += char
    return result

# 📌 AES Encryption with Correct Padding
def aes_encrypt(text, key):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_text = pad(text.encode('utf-8'), AES.block_size)  # ✅ Correct use
    encrypted = cipher.encrypt(padded_text)
    return base64.b64encode(encrypted).decode('utf-8')

# 📌 RC4 Encryption
def rc4_encrypt(text, key):
    cipher = ARC4.new(key)
    encrypted = cipher.encrypt(text.encode('utf-8'))
    return base64.b64encode(encrypted).decode('utf-8')

# 📥 Load Dataset (.csv)
df_raw = pd.read_csv(r"F:\minor_project2\dataset\final_dataset.csv")  # ✅ Make sure this path is correct
sentences = df_raw["sentence"].dropna().astype(str).tolist()

# ✨ Generate Labeled Encrypted Records
records = []

for sentence in sentences:
    sentence = sentence[:100]  # 🔒 Truncate long sentences

    key16 = get_random_bytes(16)
    key_rc4 = get_random_bytes(16)

    records.append((sentence, "Plaintext"))
    records.append((caesar_encrypt(sentence), "Caesar"))
    records.append((aes_encrypt(sentence, key16), "AES"))
    records.append((rc4_encrypt(sentence, key_rc4), "RC4"))

# 📤 Save Final Dataset
df = pd.DataFrame(records, columns=["text", "label"])
df.to_csv("encryption_dataset_404k.csv", index=False)

print(" Dataset generated and saved as 'encryption_dataset_404k.csv'")
