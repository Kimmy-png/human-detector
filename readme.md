# 🤖 Deteksi Objek dalam Zona dengan YOLOv8 dan Streamlit

Aplikasi web interaktif yang mampu mendeteksi objek (human dan lego_person) secara real-time dari kamera pengguna atau gambar yang diunggah. Aplikasi ini menggunakan model YOLOv8 yang telah di-fine-tune dan dilengkapi dengan fitur untuk mendeteksi apakah objek 'human' berada di dalam zona area yang telah ditentukan.

Aplikasi ini di-deploy menggunakan Streamlit. 

# Fitur Utama

- Deteksi Multi-Objek: Mampu mendeteksi dan membedakan antara dua kelas: human dan lego_person.

- Dua Mode Input:
    1.Upload Gambar: Pengguna dapat mengunggah file gambar (JPG, PNG) untuk dianalisis.

    2.Kamera Real-time: Mengakses webcam pengguna untuk melakukan deteksi secara langsung.
    
- Deteksi Zona (Zone Detection): Menggambar poligon area pada gambar/video dan memberikan peringatan jika objek human terdeteksi di dalamnya.

- Antarmuka Interaktif: Dibangun dengan Streamlit untuk pengalaman pengguna yang mudah dan intuitif.

- Deployment Mudah: Siap untuk di-deploy di platform cloud seperti Hugging Face Spaces.


# 🛠️ Teknologi yang Digunakan

- Model: YOLOv8 (fine-tuned)

- Framework Aplikasi Web: Streamlit

- Library Computer Vision: OpenCV, Ultralytics

- Real-time Video Streaming: streamlit-webrtc 


Cara Menjalankan & Deploy

1. Menjalankan Secara LokalUntuk menjalankan aplikasi ini di komputer Anda, ikuti langkah-langkah berikut:

Prasyarat: Pastikan Anda sudah menginstal Python 3.8+

- Clone atau unduh repositori ini.

- Buat sebuah folder proyek dan letakkan file-file berikut di dalamnya:

    1.app.py (kode utama aplikasi Streamlit)

    2.best.pt (file bobot model YOLOv8 Anda)

    3.requirements.txt (daftar library yang dibutuhkan)
    
- Buat Virtual Environment (Sangat Direkomendasikan):
```bash
python -m venv venv 

source venv/bin/activate  

# Untuk Windows: venv\Scripts\activate
```

- Instal semua library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

- Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```
Aplikasi akan terbuka secara otomatis di browser Anda.

# 📂 Struktur File.
```bash
├── best.pt           # Bobot model YOLOv8 yang sudah dilatih
├── requirements.txt    # Daftar library Python yang dibutuhkan
├── app.py              # Kode utama aplikasi Streamlit
├── train.py            # kode training model
└── README.md           # Anda sedang membacanya :)
```
# 📖 Cara Menggunakan Aplikasi

- Buka aplikasi yang sudah di-deploy.

- Gunakan sidebar di sebelah kiri untuk memilih mode input: "Upload Gambar" atau "Kamera Real-time".

- Untuk Upload Gambar:
    - Klik tombol "Browse files".

    - Pilih gambar dari komputer Anda.

    - Klik tombol "Mulai Deteksi" untuk melihat hasilnya.
    
- Untuk Kamera Real-time:
    - Izinkan browser untuk mengakses kamera Anda.

    - Klik tombol "START" untuk memulai video stream.

    - Aplikasi akan menampilkan deteksi secara langsung pada video feed.

    - Zona deteksi berwarna biru, dan bounding box akan berwarna merah jika ada orang di dalamnya.