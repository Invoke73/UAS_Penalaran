# 03_retrieval.py

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import re # Tambahkan untuk fungsi clean_text di sini juga

from transformers import AutoTokenizer, AutoModel
import torch

# Direktori dan file
DATA_PROCESSED_DIR = "../data/processed"
DATA_EVAL_DIR = "../data/eval"
CASES_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "cases.csv")
QUERIES_JSON_PATH = os.path.join(DATA_EVAL_DIR, "queries.json")

# Pastikan direktori output ada
os.makedirs(DATA_EVAL_DIR, exist_ok=True)

# Muat data kasus dari CSV yang dihasilkan Tahap 2
try:
    df_cases = pd.read_csv(CASES_CSV_PATH)
    # Pastikan kolom 'text_full' ada dan tidak kosong
    df_cases['text_full'] = df_cases['text_full'].fillna('')
    # Hilangkan baris di mana 'text_full' kosong setelah fillna
    df_cases = df_cases[df_cases['text_full'].str.strip() != '']
    if df_cases.empty:
        raise ValueError("DataFrame kasus kosong atau kolom 'text_full' kosong setelah pemrosesan.")
    print(f"[✓] {len(df_cases)} kasus dimuat dari {CASES_CSV_PATH}")
except FileNotFoundError:
    print(f"Error: File {CASES_CSV_PATH} tidak ditemukan. Pastikan Tahap 2 sudah dijalankan.")
    exit()
except ValueError as e:
    print(f"Error: {e}. Tidak ada data kasus yang valid untuk diproses.")
    exit()

# --- Implementasi BERT Embedding ---
tokenizer_bert = None
model_bert = None

def get_bert_embedding(text):
    """
    Menghasilkan embedding BERT untuk teks yang diberikan.
    Memuat model IndoBERT hanya sekali.
    """
    global tokenizer_bert, model_bert
    if tokenizer_bert is None:
        print("[+] Memuat model IndoBERT untuk embedding (hanya sekali)...")
        tokenizer_bert = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        model_bert = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        model_bert.eval() 
        print("[✓] Model IndoBERT dimuat.")
    
    inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Fungsi pembersih teks yang konsisten dengan scraper (dari 02_representation atau scraper asli)
def clean_text_for_query(text):
    """
    Membersihkan teks secara umum, konsisten dengan pembersihan dokumen kasus.
    """
    text = re.sub(r'\s+', ' ', text) # Normalisasi spasi
    # Hapus karakter non-alfanumerik kecuali yang penting untuk teks hukum (.,:()–-)
    # Sesuaikan dengan regex di scraper asli Anda jika berbeda
    text = re.sub(r'[^\w\s.,:()–-]', '', text) 
    text = text.strip().lower()
    if text.endswith(';'):
        text = text[:-1]
    return text

print("[+] Menghitung BERT embeddings untuk semua kasus...")
# Memastikan semua teks adalah string dan mengisi NaN dengan string kosong
# Pastikan juga teks sudah bersih sebelum di-embedding
case_vectors_bert = np.array([get_bert_embedding(clean_text_for_query(str(text))) for text in df_cases['text_full']])
print(f"[✓] BERT Embeddings siap. Dimensi vektor: {case_vectors_bert.shape}")

def retrieve(query: str, k: int = 5, method: str = 'bert') -> tuple[list, list]:
    """
    Mengambil top-k kasus yang paling mirip dengan query menggunakan metode BERT embedding.
    """
    # Pastikan query dibersihkan dengan cara yang sama seperti dokumen di case base
    query = clean_text_for_query(str(query))

    query_vector = None
    if method == 'bert':
        query_vector = get_bert_embedding(query).reshape(1, -1)
        similarities = cosine_similarity(query_vector, case_vectors_bert).flatten()
    else:
        raise ValueError("Metode retrieval tidak dikenal. Harap gunakan 'bert'.")

    top_k_indices = similarities.argsort()[-k:][::-1]
    
    top_k_case_ids = df_cases.iloc[top_k_indices]['case_id'].tolist()
    top_k_similarities = similarities[top_k_indices].tolist()

    return top_k_case_ids, top_k_similarities

# --- Pengujian Awal: Menghasilkan Query Uji dan Ground Truth ---
def generate_dummy_queries(num_queries=10):
    """
    Menghasilkan query uji dummy dan ground-truth untuk evaluasi.
    Akan menggunakan bagian 'solusi' atau 'ringkasan_fakta' dari kasus sebagai query
    untuk memastikan relevansi.
    """
    queries_data = []
    n_samples = min(num_queries, len(df_cases))
    if n_samples == 0:
        print("Peringatan: Tidak cukup kasus untuk membuat query dummy.")
        return []

    # Ambil sampel kasus yang memiliki solusi atau ringkasan fakta yang valid
    # Preferensi: solusi, lalu ringkasan_fakta, lalu sebagian text_full
    valid_sample_cases = df_cases[
        (df_cases['solusi'].str.strip() != "Solusi tidak dapat diekstraksi secara spesifik.") |
        (df_cases['ringkasan_fakta'].str.strip() != "Ringkasan fakta tidak dapat diekstraksi secara spesifik.")
    ]

    if len(valid_sample_cases) < n_samples:
        print(f"Peringatan: Hanya {len(valid_sample_cases)} kasus dengan solusi/fakta valid. Mengambil dari semua kasus.")
        sample_cases = df_cases.sample(n=n_samples, random_state=42, replace=True) # Pakai replace jika < n_samples
    else:
        sample_cases = valid_sample_cases.sample(n=n_samples, random_state=42)
    
    for _, row in sample_cases.iterrows():
        query_text = ""
        if row['solusi'].strip() != "Solusi tidak dapat diekstraksi secara spesifik.":
            # Gunakan solusi sebagai query jika ada dan bukan default
            query_text = row['solusi']
        elif row['ringkasan_fakta'].strip() != "Ringkasan fakta tidak dapat diekstraksi secara spesifik.":
            # Gunakan ringkasan fakta jika ada dan bukan default
            query_text = row['ringkasan_fakta']
        else:
            # Fallback ke 200-500 karakter pertama dari text_full jika tidak ada yang spesifik
            query_text = str(row['text_full'])[:500] + "..." if len(str(row['text_full'])) > 500 else str(row['text_full'])

        # Pastikan query_text dibersihkan juga
        cleaned_query_text = clean_text_for_query(query_text)
        
        queries_data.append({
            "query_id": f"q_{row['case_id']}",
            "query_text": cleaned_query_text,
            "ground_truth_case_id": row['case_id'] 
        })
    
    with open(QUERIES_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(queries_data, f, indent=4)
    print(f"[✓] Dummy queries ({len(queries_data)} buah) berhasil disimpan ke: {QUERIES_JSON_PATH}")
    return queries_data

if __name__ == "__main__":
    queries_for_testing = []
    if not os.path.exists(QUERIES_JSON_PATH):
        print(f"[!] File {QUERIES_JSON_PATH} tidak ditemukan. Menghasilkan dummy queries baru.")
        queries_for_testing = generate_dummy_queries(num_queries=10)
    else:
        try:
            with open(QUERIES_JSON_PATH, 'r', encoding='utf-8') as f:
                queries_for_testing = json.load(f)
            if not queries_for_testing or 'query_text' not in queries_for_testing[0]:
                print(f"[!] File {QUERIES_JSON_PATH} ada tetapi kosong atau formatnya tidak sesuai. Menghasilkan dummy queries baru.")
                queries_for_testing = generate_dummy_queries(num_queries=10)
            else:
                print(f"[✓] Query uji dimuat dari: {QUERIES_JSON_PATH}")
        except json.JSONDecodeError:
            print(f"[!] Error decoding JSON from {QUERIES_JSON_PATH}. File mungkin rusak. Menghasilkan dummy queries baru.")
            queries_for_testing = generate_dummy_queries(num_queries=10)


    print("\n[=] Pengujian fungsi retrieve (menggunakan BERT):")
    if queries_for_testing:
        demo_queries = queries_for_testing[:min(3, len(queries_for_testing))]
        for q_data in demo_queries:
            query_text = q_data['query_text']
            gt_case_id = q_data['ground_truth_case_id']
            
            print(f"\nQuery ID: {q_data['query_id']}")
            print(f"Query: {query_text[:100]}...")
            print(f"Ground Truth: {gt_case_id}")
            
            top_k_ids, top_k_scores = retrieve(query_text, k=5, method='bert') 
            print(f"Top 5 Retrieved Case IDs: {top_k_ids}")
            print(f"Corresponding Similarities: {[f'{s:.4f}' for s in top_k_scores]}")
            
            if gt_case_id in top_k_ids:
                print(f"-> Ground Truth '{gt_case_id}' ADA di Top-K!")
            else:
                print(f"-> Ground Truth '{gt_case_id}' TIDAK ADA di Top-K.")
    else:
        print("Tidak ada query uji untuk ditampilkan.")

