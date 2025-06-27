# original_scraper_update.py (Ini adalah file scraper awal Anda yang dimodifikasi)

import os
import requests
import re
import time
from bs4 import BeautifulSoup

BASE_PAGE = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/senjata-api-2"
SAVE_DIR = "../data/raw"
LOG_FILE = "../logs/cleaning.log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def get_links(max_pages=2):
    links = []
    for page in range(1, max_pages + 1):
        if page == 1:
            url = f"{BASE_PAGE}.html"
        else:
            url = f"{BASE_PAGE}/page/{page}.html"

        print(f"[+] Fetching page {page}: {url}")
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        entries = soup.select('div.entry-c strong a')

        for a in entries:
            href = a.get("href")
            if href and "/putusan/" in str(href):
                links.append(str(href))
    
    return links

def clean_text(text):
    # Membersihkan teks secara umum
    text = re.sub(r'\s+', ' ', text) # Normalisasi spasi
    # Hapus karakter non-alfanumerik kecuali yang penting untuk teks hukum (.,:()–-)
    text = re.sub(r'[^\w\s.,:()–-]', '', text) 
    text = text.strip().lower() # Ubah ke huruf kecil
    if text.endswith(';'): # Hapus semicolon di akhir jika ada
        text = text[:-1]
    return text

def extract_table_text(soup):
    """Mengekstrak teks dari tabel metadata sidebar."""
    sidebar = soup.select_one("#popular-post-list-sidebar")
    if not sidebar:
        return "", ""

    h2_elem = sidebar.select_one("h2")
    judul = h2_elem.get_text(separator=" ", strip=True) if h2_elem else "Putusan Tanpa Judul"
    judul = re.sub(r'\s+', ' ', judul)

    table = sidebar.select_one("table.table")
    if not table:
        return judul, ""

    texts = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 2:
            key = cells[0].get_text(strip=True)
            val = cells[1].get_text(separator="\n", strip=True)
            texts.append(f"{key}: {val}")
        elif len(cells) == 1:
            texts.append(cells[0].get_text(strip=True))

    return judul, "\n".join(texts)

def extract_main_body_text(soup):
    """
    Mengekstrak teks badan putusan utama.
    Ini seringkali berada di div konten utama, bukan sidebar.
    Mencoba mencari div dengan class 'col-md-9' yang umum untuk konten utama.
    """
    main_content_area = soup.find('div', class_='col-md-9') 
    
    if main_content_area:
        # Ekstrak semua teks paragraf di dalamnya
        paragraphs = main_content_area.find_all('p')
        if paragraphs:
            # Gabungkan paragraf, tambahkan newline ganda untuk keterbacaan
            full_text = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
            return full_text
    
    # Jika 'col-md-9' tidak ditemukan atau tidak mengandung paragraf,
    # coba cari div yang berfungsi sebagai 'box-content' (seringkali konten artikel)
    box_content = soup.find('div', class_='box-content')
    if box_content and box_content != soup.select_one("#popular-post-list-sidebar"): # Pastikan bukan sidebar
        return box_content.get_text(separator="\n\n", strip=True)

    # Fallback jika tidak ada pola yang cocok ditemukan
    return "" 

def save_case(link, idx, log):
    try:
        res = requests.get(link)
        soup = BeautifulSoup(res.text, "html.parser")

        judul, table_text = extract_table_text(soup) # Metadata dari sidebar tabel
        main_body_text = extract_main_body_text(soup) # Teks badan putusan utama

        # Gabungkan kedua teks. Teks utama putusan lebih diprioritaskan.
        # Tambahkan pemisah yang jelas untuk mempermudah debugging dan pemahaman struktur file mentah
        full_raw_content = f"=== JUDUL: {judul} ===\n\n"
        full_raw_content += f"=== METADATA TABLE ===\n{table_text}\n\n"
        full_raw_content += f"=== MAIN JUDGMENT BODY ===\n{main_body_text}\n\n" # Tambah newline untuk persiapan bersih-bersih

        # Bersihkan seluruh konten gabungan
        cleaned_combined_text = clean_text(full_raw_content)

        raw_word_count = len(full_raw_content.split())
        cleaned_word_count = len(cleaned_combined_text.split())
        ratio = cleaned_word_count / raw_word_count if raw_word_count > 0 else 0

        # Validasi bahwa teks yang dibersihkan masih memiliki proporsi yang cukup
        if ratio >= 0.8 and cleaned_word_count > 50: # Tambah cek minimal kata
            filename = os.path.join(SAVE_DIR, f"case_{idx:03}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(cleaned_combined_text) # Tulis teks yang sudah digabung dan dibersihkan

            log.write(f"case_{idx:03}.txt: OK ({cleaned_word_count}/{raw_word_count} = {ratio:.2%})\n")
            print(f"[✓] Disimpan: {filename} ({ratio:.2%})")
        else:
            log.write(f"case_{idx:03}.txt: SKIPPED ({cleaned_word_count}/{raw_word_count} = {ratio:.2%}) - terlalu sedikit teks.\n")
            print(f"[!] Lewatkan: {link} - ({ratio:.2%}) konten tersedia, terlalu sedikit teks setelah dibersihkan.")
    except Exception as e:
        log.write(f"case_{idx:03}.txt: ERROR - {str(e)}\n")
        print(f"[X] Gagal pada {link}: {e}")

if __name__ == "__main__":
    links = get_links(max_pages=2) # Atau lebih banyak halaman jika perlu
    print(f"\n[=] Total putusan ditemukan: {len(links)}\n")

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        for i, link in enumerate(links, 1):
            save_case(link, i, log)
            time.sleep(4) # Beri jeda untuk menghindari pemblokiran IP
