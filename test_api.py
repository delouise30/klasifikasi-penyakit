import requests
import json

print("Memulai pengujian API...")

url = "https://klasifikasi-penyakit-production.up.railway.app/prediksi"

gejala_list = [
    "sesak napas, batuk, mengi",
    "sering kencing, sering haus, kelelahan",
]

for gejala in gejala_list:
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"gejala": gejala}),
            timeout=10
        )
        hasil = response.json()
        # Tampilkan SEMUA isi response
        print(f"Input    : {gejala}")
        print(f"Response lengkap: {json.dumps(hasil, indent=2, ensure_ascii=False)}")
        print()
    except Exception as e:
        print(f"ERROR: {e}")

print("Selesai.")