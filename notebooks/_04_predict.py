# 04_predict.py

import os
import pandas as pd
import json
from collections import Counter
from scipy.stats import mode # Diperlukan untuk majority vote
import sys

# Tambahkan path ke direktori induk untuk mengimpor retrieve dari 03_retrieval.py
# Ini diperlukan karena 04_predict.py akan memanggil fungsi retrieve()
# Pastikan jalur ini sesuai dengan struktur direktori Anda.
# Jika 04_predict.py dan 03_retrieval.py berada di folder yang sama, baris ini bisa dihapus.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Menggunakan _03_retrieval untuk menghindari konflik penamaan dan menunjukkan ini adalah file dari proyek
try:
    from _03_retrieval import retrieve
except ImportError:
    print("Error: Tidak dapat mengimpor fungsi 'retrieve' dari '03_retrieval.py'.")
    print("Pastikan '03_retrieval.py' ada dan tidak ada kesalahan impor/path.")
    sys.exit(1)


# Direktori dan file
DATA_PROCESSED_DIR = "data/processed"
DATA_RESULTS_DIR = "data/results"
CASES_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "cases.csv")
PREDICTIONS_CSV_PATH = os.path.join(DATA_RESULTS_DIR, "predictions.csv")

# Pastikan direktori output ada
os.makedirs(DATA_RESULTS_DIR, exist_ok=True)

# Muat data kasus dari CSV untuk mengakses solusi
try:
    df_cases = pd.read_csv(CASES_CSV_PATH)
    # Pastikan kolom 'solusi' ada dan tidak kosong
    df_cases['solusi'] = df_cases['solusi'].fillna('Solusi tidak tersedia.') 
    if df_cases.empty:
        raise ValueError("DataFrame kasus kosong atau kolom 'solusi' kosong.")
    print(f"[✓] {len(df_cases)} kasus dimuat untuk reuse solusi.")
except FileNotFoundError:
    print(f"Error: File {CASES_CSV_PATH} tidak ditemukan. Pastikan Tahap 2 sudah dijalankan.")
    sys.exit(1) # Keluar jika file tidak ditemukan
except ValueError as e:
    print(f"Error: {e}. Tidak ada data kasus yang valid untuk diproses.")
    sys.exit(1)

# Membuat dictionary solusi dari case_id
case_solutions = dict(zip(df_cases['case_id'], df_cases['solusi']))
print(f"[✓] {len(case_solutions)} solusi kasus tersedia.")

def predict_outcome(query: str, k: int = 5, prediction_method: str = 'weighted_similarity') -> tuple[str, list]:
    """
    Memprediksi solusi untuk kasus baru berdasarkan top-k kasus terjemirip.

    Args:
        query (str): Teks query kasus baru.
        k (int): Jumlah kasus teratas yang akan dipertimbangkan.
        prediction_method (str): Metode agregasi solusi ('majority_vote' atau 'weighted_similarity').

    Returns:
        tuple[str, list]: Prediksi solusi dan daftar case_id yang digunakan untuk prediksi.
    """
    # Dapatkan top-k kasus terjemirip dari fungsi retrieve
    top_k_ids, top_k_similarities = retrieve(query, k=k)
    
    if not top_k_ids:
        return "Tidak ada kasus serupa yang ditemukan.", []

    # Ambil solusi dari top-k kasus yang ditemukan
    solutions_from_top_k = []
    # Simpan pasangan (solusi, kemiripan)
    solutions_with_similarities = [] 

    for case_id in top_k_ids:
        solution = case_solutions.get(case_id, "Solusi tidak tersedia.")
        solutions_from_top_k.append(solution)
        # Cari skor kemiripan yang sesuai
        try:
            # top_k_ids dan top_k_similarities diurutkan sama
            idx = top_k_ids.index(case_id)
            similarity = top_k_similarities[idx]
            solutions_with_similarities.append((solution, similarity))
        except ValueError:
            # Jika case_id tidak ditemukan di top_k_ids (seharusnya tidak terjadi)
            solutions_with_similarities.append((solution, 0.0))

    predicted_solution = "Solusi tidak dapat ditentukan."

    if prediction_method == 'majority_vote':
        # Filter solusi "Solusi tidak tersedia." sebelum voting
        valid_solutions = [s for s in solutions_from_top_k if s != "Solusi tidak tersedia."]
        if valid_solutions:
            # Menggunakan scipy.stats.mode untuk menemukan mode
            # Jika ada beberapa mode, mode().mode[0] akan mengambil yang pertama
            predicted_solution = mode(valid_solutions, keepdims=False)[0] 
        else:
            predicted_solution = "Tidak ada solusi valid untuk voting."
    elif prediction_method == 'weighted_similarity':
        weighted_sums = {}
        # Iterasi melalui solusi beserta kemiripannya
        for solution, similarity in solutions_with_similarities:
            if solution != "Solusi tidak tersedia.":
                weighted_sums[solution] = weighted_sums.get(solution, 0.0) + similarity
        
        if weighted_sums:
            # Pilih solusi dengan bobot total tertinggi
            predicted_solution = max(weighted_sums, key=weighted_sums.get)
        else:
            predicted_solution = "Tidak ada solusi valid untuk bobot kemiripan."
    else:
        raise ValueError("Metode prediksi tidak dikenal. Gunakan 'majority_vote' atau 'weighted_similarity'.")
    
    return predicted_solution, top_k_ids # Mengembalikan solusi dan case_id yang digunakan

# --- Demo Manual ---
def manual_demo():
    print("\n[=] Demo Manual Prediksi Solusi:")
    
    # Contoh query baru (bisa disesuaikan dengan jenis perkara yang Anda scraping)
    # Sesuaikan contoh query dengan domain "senjata-api-2" jika itu yang Anda gunakan
    new_queries = [
        "Kasus tentang kepemilikan senjata api ilegal tanpa izin.",
        "Penyalahgunaan senjata api untuk tindakan kriminal.",
        "Pelanggaran hukum terkait amunisi dan bahan peledak.",
        "Perkara kepemilikan senjata api untuk kepentingan bela diri namun tidak terdaftar."
    ]

    predictions_data = []
    for i, query in enumerate(new_queries):
        print(f"\nQuery Baru {i+1}: {query}")
        predicted_solution, top_k_ids_used = predict_outcome(query, k=5, prediction_method='weighted_similarity')
        
        print(f"Solusi Prediksi: {predicted_solution}")
        print(f"Top {len(top_k_ids_used)} Case IDs yang digunakan: {top_k_ids_used}")
        
        predictions_data.append({
            "query_id": f"manual_q_{i+1}",
            "query_text": query,
            "predicted_solution": predicted_solution,
            "top_5_case_ids": ", ".join(top_k_ids_used) # Simpan dalam format string
        })

    # Simpan hasil prediksi ke CSV
    df_predictions = pd.DataFrame(predictions_data)
    df_predictions.to_csv(PREDICTIONS_CSV_PATH, index=False)
    print(f"\n[✓] Hasil prediksi manual disimpan ke: {PREDICTIONS_CSV_PATH}")

if __name__ == "__main__":
    manual_demo()
