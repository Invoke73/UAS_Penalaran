# 05_evaluation.py

import os
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sys
import numpy as np

# Tambahkan path ke direktori induk untuk mengimpor retrieve dari 03_retrieval.py
# dan predict_outcome dari 04_predict.py
# Pastikan jalur ini sesuai dengan struktur direktori Anda.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from _03_retrieval import retrieve
    # Mengimpor df_cases dari 03_retrieval untuk mendapatkan akses ke data asli
    from _03_retrieval import df_cases as df_cases_from_retrieval 
    from _04_predict import predict_outcome
except ImportError:
    print("Error: Tidak dapat mengimpor fungsi yang dibutuhkan dari '03_retrieval.py' atau '04_predict.py'.")
    print("Pastikan kedua file tersebut ada dan tidak ada kesalahan impor/path.")
    sys.exit(1)


# Direktori dan file
DATA_EVAL_DIR = "data/eval"
QUERIES_JSON_PATH = os.path.join(DATA_EVAL_DIR, "queries.json")
RETRIEVAL_METRICS_CSV_PATH = os.path.join(DATA_EVAL_DIR, "retrieval_metrics.csv")
PREDICTION_METRICS_CSV_PATH = os.path.join(DATA_EVAL_DIR, "prediction_metrics.csv")

# Pastikan direktori output ada
os.makedirs(DATA_EVAL_DIR, exist_ok=True)

def eval_retrieval(queries_data: list, k: int = 5):
    """
    Mengevaluasi performa retrieval menggunakan Accuracy, Precision, Recall, F1-score.
    
    Args:
        queries_data (list): Daftar dictionary query, masing-masing dengan 'query_text'
                             dan 'ground_truth_case_id'.
        k (int): Jumlah kasus teratas yang dipertimbangkan untuk retrieval (Top-K).
    
    Returns:
        pd.DataFrame: DataFrame berisi metrik evaluasi.
    """
    precision_at_k_scores = []
    recall_at_k_scores = []
    f1_at_k_scores = []
    accuracy_at_k_scores = [] # Apakah ground truth ditemukan di top-k

    print(f"\n[=] Evaluasi Retrieval (Top-K = {k}):")
    if not queries_data:
        print("Tidak ada data query untuk dievaluasi.")
        return pd.DataFrame()

    # Untuk menyimpan detail retrieval untuk analisis kegagalan
    retrieval_details = []

    for query_info in queries_data:
        query_id = query_info['query_id']
        query_text = query_info['query_text']
        ground_truth_id = query_info['ground_truth_case_id']
        
        # Lakukan retrieval untuk query
        retrieved_ids, retrieved_scores = retrieve(query_text, k=k)

        # Cek apakah ground_truth_id ada di hasil retrieval top-k
        is_relevant_retrieved = 1 if ground_truth_id in retrieved_ids else 0
        
        # Metrik Accuracy (apakah ground truth ditemukan di top-k)
        accuracy_at_k_scores.append(is_relevant_retrieved)

        current_precision_at_k = is_relevant_retrieved / k if k > 0 else 0
        current_recall_at_k = is_relevant_retrieved / 1 
        
        if (current_precision_at_k + current_recall_at_k) == 0:
            current_f1_at_k = 0.0
        else:
            current_f1_at_k = 2 * (current_precision_at_k * current_recall_at_k) / (current_precision_at_k + current_recall_at_k)

        precision_at_k_scores.append(current_precision_at_k)
        recall_at_k_scores.append(current_recall_at_k)
        f1_at_k_scores.append(current_f1_at_k)

        # Temukan skor kemiripan untuk ground_truth_id (bahkan jika tidak di top-k)
        ground_truth_score = 0.0
        if ground_truth_id in retrieved_ids:
            gt_index_in_retrieved = retrieved_ids.index(ground_truth_id)
            ground_truth_score = retrieved_scores[gt_index_in_retrieved]
        else:
            # Jika GT tidak ada di top-k, kita perlu menghitungnya secara terpisah
            # Ini akan membutuhkan akses ke case_vectors_bert dari 03_retrieval
            # Asumsi: retrieve() bisa dipanggil dengan k yang besar atau kita punya akses ke semua skor
            # Untuk simplifikasi, kita asumsikan jika tidak di top-k, skornya relatif rendah.
            # Namun, untuk diagnosis akurat, sebaiknya panggil retrieve() dengan k yang mencakup semua kasus
            # atau hitung ulang kesamaan untuk GT. Untuk saat ini, kita akan mengabaikan skor GT jika tidak di top-k.
            # Alternatif yang lebih baik:
            # from _03_retrieval import get_bert_embedding, case_vectors_bert, df_cases_from_retrieval
            # query_vector = get_bert_embedding(query_text).reshape(1, -1)
            # all_similarities = cosine_similarity(query_vector, case_vectors_bert).flatten()
            # gt_case_index = df_cases_from_retrieval[df_cases_from_retrieval['case_id'] == ground_truth_id].index
            # if not gt_case_index.empty:
            #     ground_truth_score = all_similarities[gt_case_index[0]]
            pass # Keep ground_truth_score as 0.0 if not found in top-k for simplicity of this immersive.

        print(f"  Query '{query_id}': GT '{ground_truth_id}' {'FOUND' if is_relevant_retrieved else 'NOT FOUND'} in Top-{k} ")
        print(f"    P@{k}:{current_precision_at_k:.2f}, R@{k}:{current_recall_at_k:.2f}, F1:{current_f1_at_k:.2f}")
        print(f"    GT Score: {ground_truth_score:.4f} (if found in top-{k})")
        print(f"    Retrieved IDs & Scores: {[f'{_id} ({_s:.4f})' for _id, _s in zip(retrieved_ids, retrieved_scores)]}")

        retrieval_details.append({
            'query_id': query_id,
            'query_text': query_text,
            'ground_truth_case_id': ground_truth_id,
            'is_relevant_retrieved': is_relevant_retrieved,
            f'precision_at_{k}': current_precision_at_k,
            f'recall_at_{k}': current_recall_at_k,
            f'f1_at_{k}': current_f1_at_k,
            'retrieved_ids': retrieved_ids,
            'retrieved_scores': retrieved_scores,
            'ground_truth_score': ground_truth_score # Ini hanya akurat jika GT ditemukan di top-k
        })


    # Rata-rata metrik
    avg_accuracy = np.mean(accuracy_at_k_scores)
    avg_precision = np.mean(precision_at_k_scores)
    avg_recall = np.mean(recall_at_k_scores)
    avg_f1 = np.mean(f1_at_k_scores)

    metrics = {
        'Metric': ['Accuracy@K', 'Precision@K', 'Recall@K', 'F1-score@K'],
        'Value': [avg_accuracy, avg_precision, avg_recall, avg_f1]
    }
    df_metrics = pd.DataFrame(metrics)
    
    df_metrics.to_csv(RETRIEVAL_METRICS_CSV_PATH, index=False)
    print(f"\n[✓] Metrik retrieval berhasil disimpan ke: {RETRIEVAL_METRICS_CSV_PATH}")
    print("\nRingkasan Metrik Retrieval:")
    print(df_metrics)

    # Simpan detail retrieval ke CSV terpisah untuk analisis mendalam
    df_retrieval_details = pd.DataFrame(retrieval_details)
    df_retrieval_details.to_csv(os.path.join(DATA_EVAL_DIR, 'retrieval_details.csv'), index=False)
    print(f"[✓] Detail retrieval per query disimpan ke: {os.path.join(DATA_EVAL_DIR, 'retrieval_details.csv')}")

    return df_metrics

def eval_prediction(queries_data: list, k: int = 5):
    """
    Mengevaluasi performa prediksi solusi.
    Ini membutuhkan 'ground_truth_solution' di queries.json untuk evaluasi yang sebenarnya.
    Untuk demo, ini hanya akan mencatat prediksi.
    """
    predictions_log = []
    print("\n[=] Evaluasi Prediksi Solusi (Demo / Logging):")
    if not queries_data:
        print("Tidak ada data query untuk dievaluasi prediksinya.")
        return pd.DataFrame()

    for query_info in queries_data:
        query_text = query_info['query_text']
        
        predicted_solution, top_k_ids_used = predict_outcome(query_text, k=k)
        
        # Tambahkan ground_truth_solution jika ada di queries.json
        ground_truth_solution = query_info.get('ground_truth_solution', 'N/A (Tidak ada GT)')

        predictions_log.append({
            "query_id": query_info['query_id'],
            "query_text": query_text,
            "predicted_solution": predicted_solution,
            "ground_truth_solution": ground_truth_solution,
            "top_k_case_ids_used": ", ".join(top_k_ids_used)
        })
        print(f"  Query '{query_info['query_id']}': Predicted: '{predicted_solution[:50]}...' (GT: '{ground_truth_solution[:50]}...')")

    df_predictions_log = pd.DataFrame(predictions_log)
    df_predictions_log.to_csv(PREDICTION_METRICS_CSV_PATH, index=False)
    print(f"\n[✓] Log prediksi disimpan ke: {PREDICTION_METRICS_CSV_PATH}")
    # Catatan penting untuk Anda:
    # Untuk mendapatkan metrik evaluasi prediksi yang akurat (seperti akurasi, presisi, recall untuk solusi),
    # Anda perlu secara manual menambahkan kolom 'ground_truth_solution' ke file 'queries.json' Anda,
    # yang berisi solusi yang diharapkan untuk setiap query. Tanpa ground truth ini,
    # evaluasi prediksi hanya berupa logging apa yang diprediksi.
    print("\nCatatan: Untuk evaluasi prediksi yang sebenarnya, tambahkan 'ground_truth_solution' ke 'queries.json'.")
    return df_predictions_log


if __name__ == "__main__":
    try:
        with open(QUERIES_JSON_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        print(f"[✓] {len(queries_data)} query uji dimuat dari {QUERIES_JSON_PATH}")
    except FileNotFoundError:
        print(f"Error: File {QUERIES_JSON_PATH} tidak ditemukan.")
        print("Pastikan Tahap 3 sudah menjalankan fungsi generate_dummy_queries() untuk membuat file ini.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Format JSON tidak valid di {QUERIES_JSON_PATH}.")
        sys.exit(1)


    # Evaluasi Retrieval
    retrieval_metrics = eval_retrieval(queries_data, k=5)

    # Evaluasi Prediksi (saat ini hanya logging karena tidak ada ground truth solusi)
    prediction_results_log = eval_prediction(queries_data, k=5)

    print("\n[=] Analisis Kegagalan Model (Sederhana):")
    # Identifikasi query di mana ground truth tidak ditemukan di top-k retrieval
    failed_retrieval_queries = []
    for query_info in queries_data:
        query_text = query_info['query_text']
        ground_truth_id = query_info['ground_truth_case_id']
        retrieved_ids, _ = retrieve(query_text, k=5)
        if ground_truth_id not in retrieved_ids:
            failed_retrieval_queries.append({
                "query_id": query_info['query_id'],
                "query_text": query_text,
                "ground_truth_case_id": ground_truth_id,
                "retrieved_ids": retrieved_ids
            })
    
    if failed_retrieval_queries:
        print("\nKasus Kegagalan Retrieval (Ground Truth Tidak Ditemukan di Top-5):")
        for fail in failed_retrieval_queries:
            print(f"- Query '{fail['query_id']}':")
            print(f"  Query Text: {fail['query_text'][:70]}...")
            print(f"  Ground Truth: {fail['ground_truth_case_id']}")
            print(f"  Retrieved Top-5: {fail['retrieved_ids']}")
            print("  Rekomendasi: Perbaiki pre-processing, coba metode embedding berbeda (BERT), atau tambah data kasus.")
    else:
        print("Tidak ada kegagalan retrieval yang teridentifikasi dalam pengujian ini.")
