import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoFrameCallback

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Deteksi Orang dalam Area",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Deteksi Orang dalam Area Tertentu")
st.write("Pilih mode di sidebar: unggah gambar atau gunakan kamera secara real-time.")

# --- PEMUATAN MODEL ---
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()

# --- DEFINISI ZONA DETEKSI ---
ZONE_POLYGON = np.array([
    [100, 100], [540, 100], [540, 380], [100, 380]
], np.int32)

# --- SIDEBAR UNTUK MEMILIH MODE ---
st.sidebar.title("Mode Input")
app_mode = st.sidebar.selectbox(
    "Pilih mode:",
    ["Upload Gambar", "Kamera Real-time"]
)

# --- FUNGSI UTAMA UNTUK MEMPROSES FRAME ---
def process_frame(frame: np.ndarray):
    """
    Fungsi ini mengambil frame gambar, melakukan deteksi, dan mengembalikannya dengan anotasi.
    """
    # Lakukan prediksi dengan model
    results = model.predict(frame, verbose=False)

    # Buat salinan gambar untuk digambari
    annotated_image = frame.copy()
    person_in_zone = False
    
    # Menggambar poligon zona pada gambar
    cv2.polylines(annotated_image, [ZONE_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

    # Iterasi melalui hasil deteksi
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            is_inside = cv2.pointPolygonTest(ZONE_POLYGON, center_point, False) >= 0

            color = (0, 255, 0) # Default: Hijau (di luar zona)
            
            if is_inside and class_name == 'human':
                person_in_zone = True
                color = (0, 0, 255) # Merah (di dalam zona)
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(annotated_image, center_point, 5, color, -1)
            
    return annotated_image, person_in_zone


# --- LOGIKA BERDASARKAN MODE YANG DIPILIH ---
if app_mode == "Upload Gambar":
    st.info("Anda memilih mode 'Upload Gambar'. Silakan unggah file gambar.")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Gambar Asli', use_column_width=True)
        
        if st.button('Mulai Deteksi'):
            with st.spinner('Sedang memproses...'):
                annotated_image, person_in_zone = process_frame(img_array)
                
                with col2:
                    st.image(annotated_image, caption='Hasil Deteksi', use_column_width=True)
                
                if person_in_zone:
                    st.error("🚨 TERDETEKSI: Ada 'human' di dalam area jangkauan!")
                else:
                    st.success("✅ AMAN: Tidak ada 'human' di dalam area jangkauan.")

elif app_mode == "Kamera Real-time":
    st.info("Anda memilih mode 'Kamera Real-time'. Klik 'START' untuk memulai.")
    
    class VideoProcessor(VideoFrameCallback):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Konversi frame ke format numpy array (BGR)
            img = frame.to_ndarray(format="bgr24")
            
            # Proses frame
            annotated_image, person_in_zone = process_frame(img)
            
            # Tambahkan status teks langsung pada frame video
            status_text = "TERDETEKSI: Ada 'human' di dalam area" if person_in_zone else "AMAN: Tidak ada 'human' di dalam area"
            status_color = (0, 0, 255) if person_in_zone else (0, 255, 0) # Merah jika terdeteksi, Hijau jika aman
            cv2.putText(annotated_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Konversi frame yang sudah dianotasi kembali ke format VideoFrame
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_streamer(
        key="realtime-detection",
        video_frame_callback=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

