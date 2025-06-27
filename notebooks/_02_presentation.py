# 02_representation.py

import os
import re
import pandas as pd
import json

# Direktori tempat file .txt dari Tahap 1 disimpan
DATA_RAW_DIR = "../data/raw"
DATA_PROCESSED_DIR = "../data/processed"
CASES_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "cases.csv")

# Pastikan direktori output ada
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

def extract_metadata(text_content, filename_as_id):
    """
    Ekstrak metadata dan fitur kunci dari teks putusan yang sudah dibersihkan.
    Parameter filename_as_id digunakan untuk memastikan konsistensi case_id.
    """
    metadata = {
        'case_id': filename_as_id, # Langsung gunakan ID berdasarkan nama file
        'no_perkara': None,
        'tanggal': None,
        'jenis_perkara': None,
        'pasal': None,
        'pihak': None,
        'judul_putusan_bersih': "Judul putusan tidak dapat diekstraksi.", # Kolom baru untuk judul bersih
        'ringkasan_fakta': "Ringkasan fakta tidak dapat diekstraksi secara spesifik.", # Default jika tidak ditemukan
        'argumen_hukum_utama': "Argumen hukum utama tidak dapat diekstraksi secara spesifik.", # Default
        'solusi': "Solusi tidak dapat diekstraksi secara spesifik.", # Default
        'text_full': text_content # Seluruh teks bersih untuk vectorization
    }

    # Pisahkan bagian-bagian konten berdasarkan pemisah yang kita tambahkan di scraper
    parts = re.split(r'===\s*(?:JUDUL:|METADATA TABLE|MAIN JUDGMENT BODY)\s*===', text_content, flags=re.IGNORECASE)
    
    judul_raw = ""
    table_content_raw = ""
    main_body_raw = ""

    # Ekstraksi konten berdasarkan struktur yang diharapkan dari scraper
    # Asumsi: urutan parts[1] (judul), parts[2] (metadata table), parts[3] (main body)
    if len(parts) > 1:
        # Mencari judul yang diawali dengan "JUDUL:"
        judul_match = re.search(r'judul:\s*(.*?)(?=\n|$)', text_content, re.IGNORECASE | re.DOTALL)
        if judul_match:
            judul_raw = judul_match.group(1).strip()
            # Bersihkan judul dan simpan di kolom baru
            metadata['judul_putusan_bersih'] = re.sub(r'\s+', ' ', judul_raw).strip().lower()
        
        # Cari konten tabel metadata
        table_match = re.search(r'===\s*METADATA TABLE\s*===\s*(.*?)(?=\n\s*===\s*MAIN JUDGMENT BODY\s*===|\Z)', text_content, re.IGNORECASE | re.DOTALL)
        if table_match:
            table_content_raw = table_match.group(1).strip()

        # Cari badan putusan utama
        body_match = re.search(r'===\s*MAIN JUDGMENT BODY\s*===\s*(.*)', text_content, re.IGNORECASE | re.DOTALL)
        if body_match:
            main_body_raw = body_match.group(1).strip()
    else: # Fallback jika format baru tidak ditemukan, gunakan seluruh text_content
        main_body_raw = text_content # Ini akan menjadi text_content lama yang mungkin hanya metadata.
        # Jika tidak ada pemisah yang ditemukan, coba ambil judul dari baris pertama sebagai fallback
        first_line = text_content.split('\n', 1)[0].strip()
        if first_line.startswith('==='): # Jika ada format header lama dari scraper
             metadata['judul_putusan_bersih'] = re.sub(r'^===\s*(.*?)\s*===.*', r'\1', first_line).strip().lower()
        else:
             metadata['judul_putusan_bersih'] = first_line.lower()


    # --- Ekstraksi Metadata Dasar dari Table Content (atau main_body_raw jika tidak ada table) ---
    # Gunakan table_content_raw untuk metadata yang berasal dari tabel
    source_for_metadata = table_content_raw if table_content_raw else main_body_raw

    match = re.search(r'nomor:\s*([^\n]+)', source_for_metadata, re.IGNORECASE)
    if match: metadata['no_perkara'] = match.group(1).strip()
    
    match = re.search(r'tanggal register:\s*([^\n]+)', source_for_metadata, re.IGNORECASE)
    if match: metadata['tanggal'] = match.group(1).strip()
        
    match = re.search(r'jenis perkara:\s*([^\n]+)', source_for_metadata, re.IGNORECASE)
    if match: metadata['jenis_perkara'] = match.group(1).strip()
        
    match = re.search(r'pasal:\s*([^\n]+)', source_for_metadata, re.IGNORECASE)
    if match: metadata['pasal'] = match.group(1).strip()
        
    penggugat_match = re.search(r'penggugat:\s*([^\n]+)', source_for_metadata, re.IGNORECASE)
    tergugat_match = re.search(r'tergugat:\s*([^\n]+)', source_for_metadata, re.IGNORECASE)
    if penggugat_match and tergugat_match:
        metadata['pihak'] = f"{penggugat_match.group(1).strip()} vs. {tergugat_match.group(1).strip()}"
    elif penggugat_match:
        metadata['pihak'] = penggugat_match.group(1).strip()
    elif tergugat_match:
        metadata['pihak'] = tergugat_match.group(1).strip()


    # --- Peningkatan Ekstraksi Solusi (Amar Putusan) dan Argumen Hukum Utama ---
    # Prioritaskan pencarian di main_body_raw, atau fallback ke table_content_raw
    source_for_solusi_argumen = main_body_raw if main_body_raw else source_for_metadata

    # Cari kata kunci putusan di sumber yang lebih kaya
    # Pola yang lebih agresif untuk menangkap seluruh blok solusi/amar
    solusi_match = re.search(
        r'(MENGADILI|MEMUTUSKAN|MENETAPKAN)\s*:?\s*(.*?)(?=\n\s*(?:tanggal musyawarah|tanggal dibacakan|kaidah|abstrak|putusan|penuntut umum|terdakwa|\Z)|$)',
        source_for_solusi_argumen, re.IGNORECASE | re.DOTALL
    )
    
    if solusi_match:
        extracted_solusi = solusi_match.group(2).strip()
        # Batasi panjang solusi yang diekstrak jika terlalu panjang
        if len(extracted_solusi.split()) > 200: 
            extracted_solusi = ' '.join(extracted_solusi.split()[:200]) + "..."
        metadata['solusi'] = extracted_solusi
        metadata['argumen_hukum_utama'] = extracted_solusi # Asumsi solusi juga merupakan argumen hukum utama
    
    # Fallback ke ekstraksi pasal atau jenis perkara jika solusi masih belum ditemukan
    if metadata['solusi'] == "Solusi tidak dapat diekstraksi secara spesifik.":
        if metadata['pasal']:
            metadata['solusi'] = f"Putusan terkait pasal: {metadata['pasal']}"
        elif metadata['jenis_perkara']:
            metadata['solusi'] = f"Putusan terkait jenis perkara: {metadata['jenis_perkara']}"

    # --- Peningkatan Ekstraksi Ringkasan Fakta ---
    # Cari di main_body_raw, atau fallback ke source_for_metadata
    source_for_facts = main_body_raw if main_body_raw else source_for_metadata
    facts_keywords = ['bahwa', 'terdakwa', 'bukti', 'kejadian', 'perbuatan', 'saksi', 'keterangan']
    
    # Mencoba mencari section "DUDUK PERKARA" atau "FAKTA-FAKTA"
    facts_section_match = re.search(
        r'(DUDUK PERKARA|FAKTA-FAKTA|TENTANG FAKTA-FAKTA)\s*(.*?)(?=\n\s*(?:MENIMBANG|MENGADILI|MEMUTUSKAN|MENETAPKAN|\Z))',
        source_for_facts, re.IGNORECASE | re.DOTALL
    )
    
    if facts_section_match:
        metadata['ringkasan_fakta'] = facts_section_match.group(2).strip()
        if len(metadata['ringkasan_fakta'].split()) > 150:
            metadata['ringkasan_fakta'] = ' '.join(metadata['ringkasan_fakta'].split()[:150]) + "..."
    else: # Fallback ke pencarian baris dengan kata kunci jika tidak ada section header
        extracted_facts_lines = []
        for line in source_for_facts.split('\n'):
            if any(keyword in line.lower() for keyword in facts_keywords):
                extracted_facts_lines.append(line.strip())
                if len(" ".join(extracted_facts_lines).split()) > 150:
                    break
        if extracted_facts_lines:
            metadata['ringkasan_fakta'] = " ".join(extracted_facts_lines)
    
    # Hitung panjang teks dari seluruh konten gabungan
    metadata['text_length'] = len(text_content.split())

    return metadata

def create_case_representation():
    """
    Membuat representasi kasus dari file teks mentah yang dihasilkan Tahap 1.
    """
    cases_data = []
    txt_files = sorted([f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.txt')])
    
    if not txt_files:
        print(f"Peringatan: Tidak ada file .txt ditemukan di {DATA_RAW_DIR}. Pastikan Tahap 1 sudah dijalankan.")
        return

    for filename in txt_files:
        file_path = os.path.join(DATA_RAW_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text_content = f.read() # Ini sekarang akan berisi gabungan metadata dan badan utama
        
        # Pastikan case_id selalu dari nama file
        base_filename_id = os.path.basename(filename).replace(".txt", "")
        extracted_data = extract_metadata(full_text_content, base_filename_id)
        
        cases_data.append(extracted_data)
    
    # Buat DataFrame dari data yang diekstrak
    df_cases = pd.DataFrame(cases_data)
    
    # Pastikan kolom yang dibutuhkan ada, jika tidak, isi dengan string kosong
    required_cols = ['case_id', 'no_perkara', 'tanggal', 'jenis_perkara', 'pasal', 
                     'pihak', 'judul_putusan_bersih', 'ringkasan_fakta', 'argumen_hukum_utama', 'solusi', 
                     'text_full', 'text_length']
    for col in required_cols:
        if col not in df_cases.columns:
            df_cases[col] = ''
    
    # Urutkan ulang kolom sesuai contoh
    df_cases = df_cases[required_cols]

    df_cases.to_csv(CASES_CSV_PATH, index=False)
    print(f"[✓] Representasi kasus berhasil disimpan ke: {CASES_CSV_PATH}")

if __name__ == "__main__":
    create_case_representation()
