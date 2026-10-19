import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Judul Website
st.title("Aplikasi Deteksi & Klasifikasi Makanan Otomatis 🍔")
st.write("Unggah foto makananmu di bawah ini untuk dideteksi oleh AI!")

# 2. Tombol Upload Gambar di Website
file_gambar = st.file_uploader("Pilih foto makanan (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if file_gambar is not None:
    # Menampilkan gambar yang di-upload pengguna ke layar web
    gambar = Image.open(file_gambar)
    st.image(gambar, caption='Foto yang kamu unggah', use_column_width=True)
    
    st.write("---")
    st.write("### 🤖 Hasil Analisis AI:")
    
    # 3. Menjalankan Model AI dari Google (Hugging Face)
    with st.spinner('Tunggu sebentar, AI sedang menganalisis...'):
        klasifikasi = pipeline("image-classification", model="google/vit-base-patch16-224")
        hasil = klasifikasi(gambar)
        
        # 4. Menampilkan hasil 3 tebakan teratas ke website
        for tebakan in hasil[:3]:
            persentase = tebakan['score'] * 100
            st.write(f"*{tebakan['label']}* : {persentase:.2f}%")
