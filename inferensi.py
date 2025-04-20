import streamlit as st
import pandas as pd
import joblib

st.title("Prediksi Sentimen dari File CSV")
st.subheader("by Rehand Naifisurya")
st.write("Upload model dan file vectorizer")

# Upload model & vectorizer
model_file = st.file_uploader("Upload Model (.pkl)", type=["pkl"], key="model")
vectorizer_file = st.file_uploader("Upload TF-IDF Vectorizer (.pkl)", type=["pkl"], key="vectorizer")

# Upload file CSV
uploaded_file = st.file_uploader("Upload data test (.CSV)", type=["csv"], key="csv")

# Load model dan vectorizer setelah semua file tersedia
if model_file is not None and vectorizer_file is not None and uploaded_file is not None:
    try:
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        
        # Baca file CSV
        df_test = pd.read_csv(uploaded_file)

        if 'text_stemming' not in df_test.columns:
            st.error("Kolom 'text_stemming' tidak ditemukan dalam file CSV.")
        else:
            # Transformasi dan prediksi
            X_test_transformed = vectorizer.transform(df_test['text_stemming'])
            predictions = model.predict(X_test_transformed)

            # Tambahkan hasil ke DataFrame
            df_test['Label'] = predictions

            # Tampilkan hasil
            st.subheader("Hasil Prediksi")
            st.write(df_test[['text_stemming', 'Label']].head())

            # Download hasil
            csv = df_test.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Hasil Prediksi",
                data=csv,
                file_name='hasil_prediksi.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan unggah model, vectorizer, dan file CSV untuk memulai.")
