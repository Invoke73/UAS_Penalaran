# Sistem Penalaran Berbasis Kasus (CBR) untuk Analisis Putusan Pengadilan
Proyek ini mengimplementasikan sistem Case-Based Reasoning (CBR) sederhana dalam Python untuk mendukung analisis putusan pengadilan. Sistem ini memanfaatkan data putusan yang diunduh dari Direktori Putusan Mahkamah Agung Republik Indonesia.

## Struktur Proyek

```bash
your_project_folder/
├── data/
│   ├── raw/                 # File teks putusan mentah (hasil scraping)
│   ├── processed/           # Data putusan yang sudah direpresentasikan (cases.csv)
│   └── eval/                # Data dan hasil evaluasi (queries.json, metrics.csv)
│   └── results/             # Hasil prediksi solusi (predictions.csv)
├── logs/
│   └── cleaning.log         # Log pembersihan (opsional)
├── notebooks/               # Folder untuk semua skrip Python dan Jupyter Notebooks
│   ├── _01_scraper.py # Skrip untuk scraping dan pembersihan awal (Tahap 1)
│   ├── _02_representation.py     # Skrip untuk representasi kasus (Tahap 2)
│   ├── _03_retrieval.py          # Skrip untuk retrieval kasus (Tahap 3)
│   ├── _04_predict.py            # Skrip untuk prediksi/reuse solusi (Tahap 4)
│   └── _05_evaluation.py         # Skrip untuk evaluasi model (Tahap 5)
├── requirements.txt         # Daftar dependensi Python
└── README.md                # Dokumen ini

```

## Instalasi

1. Klon Repositori (Jika berlaku)
```bash
git clone <URL_REPOSITORI_ANDA>
cd your_project_folder
```

2. Buat dan Aktifkan Virtual Environment (Direkomendasikan)
```bash
python -m venv venv
```
```bash
# Untuk Windows:
.\venv\Scripts\activate
```

```bash
# Untuk macOS/Linux:
source venv/bin/activate
```

3. Instal Dependensi
Pastikan Anda memiliki pip terinstal. Anda dapat menginstal semua dependensi yang diperlukan menggunakan file requirements.txt yang sudah disediakan:

```bash
pip install -r requirements.txt
```
Isi requirements.txt yang diharapkan:

requests
beautifulsoup4
pandas
scikit-learn
numpy
transformers
torch
scipy

Jika ada dependensi yang belum terdaftar, Anda bisa menambahkannya ke requirements.txt dan menjalankan pip install -r requirements.txt lagi.

## Cara Menjalankan Pipeline End-to-End
Untuk menjalankan seluruh sistem CBR dari awal hingga akhir, ikuti langkah-langkah di bawah ini secara berurutan. Penting: Pastikan Anda menghapus file-file output dari eksekusi sebelumnya sebelum memulai.

1. Pembersihan File Output Sebelumnya (Wajib)
Sebelum menjalankan ulang, hapus semua file yang dihasilkan dari eksekusi sebelumnya untuk memastikan konsistensi data:

```bash
del data\raw\*.txt           # Windows
del data\processed\cases.csv # Windows
del data\eval\queries.json   # Windows
del data\eval\retrieval_metrics.csv # Windows
del data\eval\retrieval_details.csv # Windows
del data\eval\prediction_metrics.csv # Windows
del data\results\predictions.csv # Windows
```

```bash
# Atau untuk Linux/macOS:
rm data/raw/*.txt
rm data/processed/cases.csv
rm data/eval/queries.json
rm data/eval/retrieval_metrics.csv
rm data/eval/retrieval_details.csv
rm data/eval/prediction_metrics.csv
rm data/results/predictions.csv
```


2. Tahap 1: Membangun Case Base (Scraping dan Pembersihan Awal)
Skrip ini akan mengunduh dokumen putusan dari Direktori MA RI dan melakukan pembersihan awal. Dokumen yang diunduh akan disimpan dalam format .txt di data/raw/.

```bash
python _01_scraper.py
```

(Catatan: Anda dapat menyesuaikan max_pages di original_scraper_update.py untuk mengunduh lebih banyak dokumen jika diperlukan, atau mengubah BASE_PAGE untuk domain perkara lain.)

3. Tahap 2: Case Representation
Skrip ini akan membaca file teks putusan mentah dari data/raw/, mengekstrak metadata penting (seperti nomor perkara, tanggal, pasal, pihak), ringkasan fakta, dan amar putusan ("solusi"). Hasilnya akan disimpan dalam format CSV di data/processed/cases.csv.

```bash
python 02_representation.py
```

4. Tahap 3: Case Retrieval
Skrip ini akan:

Memuat data kasus dari cases.csv.

Menghitung embedding BERT untuk semua dokumen kasus.

Membuat dummy query uji (untuk keperluan demonstrasi dan evaluasi) dan menyimpannya ke data/eval/queries.json.

Mendemonstrasikan fungsi retrieval yang mencari kasus terjemirip menggunakan embedding BERT.

```bash
python 03_retrieval.py
```

5. Tahap 4: Case/Solution Reuse
Skrip ini akan menggunakan hasil retrieval dari Tahap 3 untuk memprediksi "solusi" (amar putusan) untuk kasus baru berdasarkan kasus-kasus lama yang paling mirip. Hasil prediksi akan disimpan ke data/results/predictions.csv.

```bash
python 04_predict.py
```

6. Tahap 5: Evaluasi Model
Skrip ini akan mengevaluasi performa sistem retrieval Anda (menggunakan metrik seperti Accuracy, Precision, Recall, F1-score) berdasarkan query uji yang dihasilkan di Tahap 3. Hasil metrik akan disimpan ke data/eval/retrieval_metrics.csv dan data/eval/retrieval_details.csv. Ini juga akan mencatat hasil prediksi solusi.

```bash
python 05_evaluation.py
```

Troubleshooting
"Error: File ... tidak ditemukan.": Pastikan Anda telah menjalankan skrip tahap sebelumnya dengan benar dan file output yang diperlukan telah dibuat di lokasi yang benar. Pastikan juga Anda telah menghapus file-file output yang lama sebelum menjalankan kembali pipeline.

"KeyError: 'query_text'" atau Metrik Retrieval 0.0: Masalah ini biasanya disebabkan oleh ketidaksesuaian case_id atau query_text antara queries.json dan cases.csv. Pastikan Anda telah mengikuti langkah "Pembersihan File Output Sebelumnya (Wajib)" dan menjalankan seluruh pipeline secara berurutan. Verifikasi bahwa case_id di cases.csv berformat case_0xx.

Prediksi Solusi "Solusi tidak dapat diekstraksi secara spesifik.": Ini menunjukkan masalah pada ekstraksi solusi di 02_representation.py atau, lebih mungkin, bahwa skrip scraping Anda (original_scraper_update.py) tidak berhasil mengambil seluruh badan putusan utama. Pastikan original_scraper_update.py mengambil seluruh teks utama dari halaman putusan, bukan hanya metadata sidebar. Periksa beberapa file .txt di data/raw/ secara manual untuk memverifikasi isinya.

Untuk pertanyaan lebih lanjut atau bantuan debugging, silakan hubungi pengembang proyek.
