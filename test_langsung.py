import joblib
import json
import re
import os
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model  = joblib.load(os.path.join(BASE_DIR, 'model_final.pkl'))
tfidf  = joblib.load(os.path.join(BASE_DIR, 'tfidf_final.pkl'))

with open(os.path.join(BASE_DIR, 'config_model.json'), 'r') as f:
    config = json.load(f)
mapping_id = config['label_mapping']

stop_en = set(stopwords.words('english'))
kata_penting = {'no','not','loss','gain','shortness','blurred',
                'rapid','high','low','severe','weakness','numbness'}
stop_en = stop_en - kata_penting

# Kamus gejala Indonesia → kata PERSIS di dataset
KAMUS_GEJALA = {
    # Diabetes
    'penglihatan kabur'         : 'blurred and distorted vision',
    'penglihatan buram'         : 'blurred and distorted vision',
    'lapar berlebihan'          : 'excessive hunger',
    'sering lapar'              : 'excessive hunger',
    'nafsu makan meningkat'     : 'increased appetite',
    'kelelahan'                 : 'fatigue',
    'lelah'                     : 'fatigue',
    'lemas'                     : 'lethargy',
    'lesu'                      : 'lethargy',
    'obesitas'                  : 'obesity',
    'kegemukan'                 : 'obesity',
    'sering kencing'            : 'polyuria',
    'sering buang air kecil'    : 'polyuria',
    'banyak kencing'            : 'polyuria',
    'gelisah'                   : 'restlessness',
    'berat badan turun'         : 'weight loss',
    'penurunan berat badan'     : 'weight loss',
    'gula darah tidak stabil'   : 'irregular sugar level',
    'gula darah tinggi'         : 'irregular sugar level',

    # Hypertension
    'nyeri dada'                : 'chest pain',
    'sakit dada'                : 'chest pain',
    'pusing'                    : 'dizziness',
    'kepala pusing'             : 'dizziness',
    'sakit kepala'              : 'headache',
    'kepala sakit'              : 'headache',
    'susah konsentrasi'         : 'lack of concentration',
    'sulit konsentrasi'         : 'lack of concentration',
    'kehilangan keseimbangan'   : 'loss of balance',
    'tidak seimbang'            : 'loss of balance',
    'jantung berdebar'          : 'chest pain',

    # Bronchial Asthma
    'sesak napas'               : 'breathlessness',
    'susah napas'               : 'breathlessness',
    'napas sesak'               : 'breathlessness',
    'batuk'                     : 'cough',
    'riwayat keluarga'          : 'family history',
    'demam tinggi'              : 'high fever',
    'demam'                     : 'high fever',
    'dahak'                     : 'mucoid sputum',
    'dahak berlendir'           : 'mucoid sputum',
    'mengi'                     : 'breathlessness',
    'dada berat'                : 'breathlessness',

    # Paralysis (brain hemorrhage) / Stroke
    'lumpuh'                    : 'weakness of one body side',
    'kelumpuhan'                : 'weakness of one body side',
    'satu sisi tubuh lemah'     : 'weakness of one body side',
    'mati rasa'                 : 'altered sensorium',
    'mati rasa di wajah'        : 'altered sensorium',
    'kesadaran berubah'         : 'altered sensorium',
    'bingung'                   : 'altered sensorium',
    'bicara pelo'               : 'altered sensorium',
    'sulit bicara'              : 'altered sensorium',
    'muntah'                    : 'vomiting',
    'mual muntah'               : 'vomiting',
}

def terjemahkan_kamus(teks_id):
    """Terjemahkan menggunakan kamus gejala yang tepat"""
    teks = teks_id.lower().strip()
    hasil_kata = []

    # Cek frasa panjang dulu (prioritas)
    for kata_id, kata_en in sorted(KAMUS_GEJALA.items(),
                                    key=lambda x: len(x[0]),
                                    reverse=True):
        if kata_id in teks:
            hasil_kata.append(kata_en)
            teks = teks.replace(kata_id, '')

    if not hasil_kata:
        return teks_id  # kembalikan asli jika tidak ada di kamus

    return ' '.join(hasil_kata)

def preprocessing(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    token = [t for t in teks.split() if t not in stop_en]
    return ' '.join(token).strip()

def prediksi(gejala_id):
    gejala_en = terjemahkan_kamus(gejala_id)
    bersih    = preprocessing(gejala_en)
    vektor    = tfidf.transform([bersih])
    hasil     = model.predict(vektor)[0]
    proba     = model.predict_proba(vektor)[0]

    print(f"Input    : {gejala_id}")
    print(f"Inggris  : {gejala_en}")
    print(f"Bersih   : {bersih}")
    print(f"Prediksi : {mapping_id.get(hasil, hasil)}")
    print("Probabilitas:")
    for k, p in sorted(zip(model.classes_, proba),
                        key=lambda x: x[1], reverse=True):
        bar = '█' * int(p * 30)
        print(f"  {mapping_id.get(k,k):<20} {bar} {p*100:.1f}%")
    print()

print("=== UJI PREDIKSI DENGAN KAMUS GEJALA ===\n")

print("--- UJI ASMA ---")
prediksi("sesak napas, batuk, demam, dahak")

print("--- UJI DIABETES ---")
prediksi("sering kencing, penglihatan kabur, kelelahan, berat badan turun")

print("--- UJI STROKE ---")
prediksi("lumpuh, bicara pelo, muntah, sakit kepala")

print("--- UJI HIPERTENSI ---")
prediksi("sakit kepala, pusing, nyeri dada, susah konsentrasi")