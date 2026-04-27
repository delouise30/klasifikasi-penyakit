import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=== CEK FILE ===")
print(f"Folder aktif: {BASE_DIR}")
print()

# Cek model
path_model = os.path.join(BASE_DIR, 'model_final.pkl')
model = joblib.load(path_model)
print(f"model_final.pkl")
print(f"  Lokasi : {path_model}")
print(f"  Tipe   : {type(model).__name__}")
print(f"  Fitur  : {model.n_features_in_}")
print()

# Cek tfidf
path_tfidf = os.path.join(BASE_DIR, 'tfidf_final.pkl')
tfidf = joblib.load(path_tfidf)
print(f"tfidf_final.pkl")
print(f"  Lokasi : {path_tfidf}")
print(f"  Fitur  : {len(tfidf.vocabulary_)}")