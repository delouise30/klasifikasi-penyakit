from flask import Flask, request, jsonify
import joblib, json, re, os, nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model_final.pkl'))
tfidf = joblib.load(os.path.join(BASE_DIR, 'tfidf_final.pkl'))

with open(os.path.join(BASE_DIR, 'config_model.json'), 'r') as f:
    config = json.load(f)
mapping_id = config['label_mapping']

print("Tipe model  :", type(model).__name__)
print("Fitur model :", model.n_features_in_)
print("Fitur tfidf :", len(tfidf.vocabulary_))

stop_en = set(stopwords.words('english'))
kata_penting = {'no','not','loss','gain','shortness','blurred','rapid','high','low','severe','weakness','numbness'}
stop_en = stop_en - kata_penting

KAMUS = {
    'penglihatan kabur': 'blurred and distorted vision',
    'penglihatan buram': 'blurred and distorted vision',
    'lapar berlebihan': 'excessive hunger',
    'sering lapar': 'excessive hunger',
    'nafsu makan meningkat': 'increased appetite',
    'kelelahan': 'fatigue',
    'lelah': 'fatigue',
    'lemas': 'lethargy',
    'lesu': 'lethargy',
    'obesitas': 'obesity',
    'kegemukan': 'obesity',
    'sering kencing': 'polyuria',
    'sering buang air kecil': 'polyuria',
    'banyak kencing': 'polyuria',
    'gelisah': 'restlessness',
    'berat badan turun': 'weight loss',
    'penurunan berat badan': 'weight loss',
    'gula darah tidak stabil': 'irregular sugar level',
    'gula darah tinggi': 'irregular sugar level',
    'nyeri dada': 'chest pain',
    'sakit dada': 'chest pain',
    'pusing': 'dizziness',
    'kepala pusing': 'dizziness',
    'sakit kepala': 'headache',
    'kepala sakit': 'headache',
    'susah konsentrasi': 'lack of concentration',
    'sulit konsentrasi': 'lack of concentration',
    'kehilangan keseimbangan': 'loss of balance',
    'tidak seimbang': 'loss of balance',
    'jantung berdebar': 'chest pain',
    'sesak napas': 'breathlessness',
    'susah napas': 'breathlessness',
    'napas sesak': 'breathlessness',
    'batuk': 'cough',
    'riwayat keluarga': 'family history',
    'demam tinggi': 'high fever',
    'demam': 'high fever',
    'dahak': 'mucoid sputum',
    'dahak berlendir': 'mucoid sputum',
    'mengi': 'breathlessness',
    'dada berat': 'breathlessness',
    'lumpuh': 'weakness of one body side',
    'kelumpuhan': 'weakness of one body side',
    'satu sisi tubuh lemah': 'weakness of one body side',
    'mati rasa': 'altered sensorium',
    'mati rasa di wajah': 'altered sensorium',
    'kesadaran berubah': 'altered sensorium',
    'bingung': 'altered sensorium',
    'bicara pelo': 'altered sensorium',
    'sulit bicara': 'altered sensorium',
    'muntah': 'vomiting',
    'mual muntah': 'vomiting',
    'sering haus': 'polydipsia',
    'banyak minum': 'polydipsia',
    'nyeri sendi': 'swelling joints',
    'sendi bengkak': 'swelling joints',
    'bengkak sendi': 'swelling joints',
    'sakit saat jalan': 'painful walking',
    'sulit berjalan': 'painful walking',
    'kaku sendi': 'movement stiffness',
    'sendi kaku': 'movement stiffness',
    'leher kaku': 'stiff neck',
    'otot lemah': 'muscle weakness',
}

def terjemahkan(teks_id):
    teks = teks_id.lower().strip()
    hasil = []
    for k, v in sorted(KAMUS.items(), key=lambda x: len(x[0]), reverse=True):
        if k in teks:
            hasil.append(v)
            teks = teks.replace(k, '')
    return ' '.join(hasil) if hasil else teks_id

def preprocessing(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    token = [t for t in teks.split() if t not in stop_en]
    return ' '.join(token).strip()

@app.route('/')
def index():
    return jsonify({'status': 'API berjalan', 'versi': '3.0'})

@app.route('/prediksi', methods=['POST'])
def prediksi():
    try:
        data = request.get_json(force=True)
        if not data or 'gejala' not in data:
            return jsonify({'status': 'error', 'pesan': 'Field gejala wajib diisi'}), 400
        gejala_input = str(data['gejala']).strip()
        if not gejala_input:
            return jsonify({'status': 'error', 'pesan': 'Gejala tidak boleh kosong'}), 400
        gejala_en = terjemahkan(gejala_input)
        gejala_bersih = preprocessing(gejala_en)
        print("Input   :", gejala_input)
        print("Inggris :", gejala_en)
        print("Bersih  :", gejala_bersih)
        vektor = tfidf.transform([gejala_bersih])
        hasil = model.predict(vektor)[0]
        probabilitas = model.predict_proba(vektor)[0]
        semua_prob = {}
        for kelas, prob in zip(model.classes_, probabilitas):
            nama_id = mapping_id.get(kelas, kelas)
            semua_prob[nama_id] = round(float(prob) * 100, 2)
        semua_prob = dict(sorted(semua_prob.items(), key=lambda x: x[1], reverse=True))
        print("Prediksi:", mapping_id.get(hasil, hasil))
        return jsonify({
            'status': 'success',
            'gejala_input': gejala_input,
            'gejala_inggris': gejala_en,
            'prediksi': mapping_id.get(hasil, hasil),
            'probabilitas': semua_prob
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'pesan': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)