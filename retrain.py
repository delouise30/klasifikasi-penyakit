import pandas as pd
import re
import joblib
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("=== MEMULAI TRAINING ULANG ===\n")

# Load dataset
df = pd.read_csv('dataset.csv')
df['Disease'] = df['Disease'].str.strip()

# Gabungkan kolom gejala
kolom_gejala = ['Symptom_1','Symptom_2','Symptom_3','Symptom_4',
                'Symptom_5','Symptom_6','Symptom_7','Symptom_8',
                'Symptom_9','Symptom_10','Symptom_11','Symptom_12',
                'Symptom_13','Symptom_14','Symptom_15','Symptom_16',
                'Symptom_17']

def gabung_gejala(baris):
    gejala = []
    for col in kolom_gejala:
        nilai = baris[col]
        if pd.notna(nilai):
            gejala.append(str(nilai).strip().replace('_', ' '))
    return ' '.join(gejala)

df['Symptoms_gabung'] = df.apply(gabung_gejala, axis=1)

# Filter 4 penyakit
penyakit_dipilih = [
    'Diabetes',
    'Hypertension',
    'Bronchial Asthma',
    'Paralysis (brain hemorrhage)'
]

df_filter = df[df['Disease'].isin(penyakit_dipilih)].copy()
df_filter = df_filter.reset_index(drop=True)

print("Jumlah data per penyakit:")
print(df_filter['Disease'].value_counts())

# Preprocessing
stop_en = set(stopwords.words('english'))
kata_penting = {'no','not','loss','gain','shortness',
                'blurred','rapid','high','low','severe',
                'weakness','numbness','difficulty'}
stop_en = stop_en - kata_penting

def preprocessing(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    token = [t for t in teks.split() if t not in stop_en]
    return ' '.join(token).strip()

df_filter['Symptoms_clean'] = df_filter['Symptoms_gabung'].apply(preprocessing)

# TF-IDF
X = df_filter['Symptoms_clean']
y = df_filter['Disease']

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(X)

print(f"\nJumlah fitur TF-IDF : {X_tfidf.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Latih SVM
print("Melatih SVM...")
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print(f"\n╔══════════════════════════════════════╗")
print(f"║   AKURASI : {akurasi*100:.2f}%                 ║")
print(f"╚══════════════════════════════════════╝")
print("\n", classification_report(y_test, y_pred))

# Verifikasi sinkronisasi
fitur_tfidf = X_tfidf.shape[1]
fitur_model = model.n_features_in_
print(f"Fitur TF-IDF  : {fitur_tfidf}")
print(f"Fitur Model   : {fitur_model}")

if fitur_tfidf == fitur_model:
    # Simpan semua file
    joblib.dump(model, 'model_final.pkl')
    joblib.dump(tfidf, 'tfidf_final.pkl')

    config = {
        'label_mapping': {
            'Diabetes'                     : 'Diabetes',
            'Hypertension'                 : 'Hipertensi',
            'Bronchial Asthma'             : 'Asma',
            'Paralysis (brain hemorrhage)' : 'Stroke'
        }
    }
    with open('config_model.json', 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n✓ SINKRON — semua file berhasil disimpan")
    print("✓ model_final.pkl")
    print("✓ tfidf_final.pkl")
    print("✓ config_model.json")
    print("\nSekarang jalankan: python app.py")
else:
    print("\n✗ TIDAK SINKRON — ada masalah")