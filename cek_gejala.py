import pandas as pd

df = pd.read_csv('dataset.csv')
df['Disease'] = df['Disease'].str.strip()

penyakit_dipilih = [
    'Diabetes',
    'Hypertension',
    'Bronchial Asthma',
    'Paralysis (brain hemorrhage)'
]

kolom_gejala = ['Symptom_1','Symptom_2','Symptom_3','Symptom_4',
                'Symptom_5','Symptom_6','Symptom_7','Symptom_8',
                'Symptom_9','Symptom_10','Symptom_11','Symptom_12',
                'Symptom_13','Symptom_14','Symptom_15','Symptom_16',
                'Symptom_17']

for penyakit in penyakit_dipilih:
    df_p = df[df['Disease'] == penyakit]
    semua_gejala = set()
    for col in kolom_gejala:
        for val in df_p[col].dropna():
            semua_gejala.add(val.strip())
    print(f"\n{penyakit}:")
    for g in sorted(semua_gejala):
        print(f"  {g}")