import joblib
import pandas as pd
from model_test_feature_extraction import extract_features

# Load model and label encoder
clf = joblib.load(r"F:\minor_project2\model\trial_1\encrytion_model.pkl")
label_encoder = joblib.load(r"F:\minor_project2\model\trial_1\label_encoder.pkl")

test_cases = [
    # 🔐 AES Encrypted (Base64 Encoded)
    ("v/MtQ9qTDvQnyS4p7MdkQw==", "AES"),           # Hello World
    ("ElKdLZfVce2c+VpBAY1vQQ==", "AES"),           # Attack at Dawn
    ("xvRg6Y7SivA55fy9aWgppg==", "AES"),           # Data is Power
    ("klpbipZ+SpFejU9OkyGHDQ==", "AES"),           # Encrypt the message
    ("hdZEG8Q5FtCEU1UmyQodfA==", "AES"),           # Cryptography is fun

    # 🔐 DES Encrypted (Base64 Encoded)
#    ("o3G/RyxmhU0=", "DES"),                      # Hello World
#    ("A/HacA1DxXY=", "DES"),                      # Attack at Dawn
#    ("YrK0Ln5sHg8=", "DES"),                      # Data is Power
#    ("kk4Z7ld3JpA=", "DES"),                      # Encrypt the message
#    ("tzrE+LTx7IM=", "DES"),                      # Cryptography is fun

    # 🔒 Caesar Cipher (Shift = 3)
    ("KHOOR ZRUOG", "Caesar"),                   # Hello World
    ("FUBSWRJUDSKB LV IXQ", "Caesar"),           # Cryptography is fun
    ("DWWDFN DW GDZQ", "Caesar"),                # Attack at Dawn
    (r"HQFU\SWW WKH PHVVDJH", "Caesar"),          # Encrypt the message
    ("GDWD LV SRZHU", "Caesar"), 
                    
                    # 🔐 RC4 Encrypted (Base64 Encoded)
    ("9OQ7Tq6PBAA=", "RC4"),                      # Hello World
    ("jz+BbJqSn0A=", "RC4"),                      # Attack at Dawn
    ("rN4sYB5vJmA=", "RC4"),                      # Data is Power
    ("xqNjpcfL8PM=", "RC4"),                      # Encrypt the message
    ("GqG+Aa5Bx8U=", "RC4"),                      # Cryptography is fun
                # Data is Power

    # 🟢 Plaintext
    ("HELLO WORLD", "Plaintext"),               
    ("CRYPTOGRAPHY IS FUN", "Plaintext"),       
    ("ATTACK AT DAWN", "Plaintext"),            
    ("ENCRYPT THE MESSAGE", "Plaintext"),       
    ("DATA IS POWER", "Plaintext"),      

    ("y8gqb8BL4BzFAErMNemX6w==","AES")       
]


# Predict
for text, actual in test_cases:
    try:
        feats = extract_features(text)
        df = pd.DataFrame([feats])
        pred = clf.predict(df)
        label = label_encoder.inverse_transform(pred)[0]
        print(f"Real: {actual}, Predicted: {label}")
    except Exception as e:
        print(f"Error processing '{text}': {e}")
