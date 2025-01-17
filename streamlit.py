import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import time
from pymongo import MongoClient
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import silhouette_score
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
#import os
#from dotenv import load_dotenv
import os
os.system("pip install matplotlib")


# Koneksi ke MongoDB Atlas
client = MongoClient("mongodb+srv://krisna:krisna@cluster0.3mao11f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['dbkrisna']
collection_visualisasi = db['visualisasi']
collection_arima = db['arima']
collection_kmeans = db['kmeans']
collection_history = db['history']

#SECRET_KEY=your_secret_key  # Change this to a strong secret key
#ADMIN_KEY=your_admin_key     # Change this to your admin key
#MONGODB_URI=mongodb+srv://krisna:krisna@cluster0.3mao11f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
#DB_NAME=dbkrisna    

def clear_collection(collection):
    with st.spinner("Mengosongkan collection..."):
        collection.delete_many({})  # Menghapus semua dokumen
        st.success("Collection berhasil dikosongkan!")

def delete_history_by_collection_name(collection_name):
    with st.spinner(f"Menghapus semua riwayat '{collection_name}'..."):
        result = collection_history.delete_many({"collection_name": collection_name})

def load_data(collection):
    # Membuat progres bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Mengambil data dari MongoDB
    cursor = collection.find()
    data_list = []
    total = collection.count_documents({})  # Total dokumen dalam koleksi
    count = 0

    # Iterasi melalui data untuk mengisi progres bar
    for doc in cursor:
        data_list.append(doc)
        count += 1
        progress = int((count / total) * 100)  # Hitung persentase
        progress_bar.progress(progress)      # Update progres
        status_text.text(f"Memuat data... {progress}%")  # Update teks status
        time.sleep(0.01)  # Simulasi proses (hapus jika tidak diperlukan)

    # Menghapus progres bar setelah selesai
    progress_bar.empty()
    status_text.text("Memuat data selesai!")

    # Konversi ke DataFrame
    data = pd.DataFrame(data_list)
    return data

def save_data(collection, data):
    collection.delete_many({})
    # Membuat progres bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    records = data.to_dict('records')  # Konversi ke format dictionary
    total = len(records)               # Total data yang akan disimpan
    batch_size = 10                    # Simpan dalam batch kecil untuk progres

    for i in range(0, total, batch_size):
        # Ambil batch data
        batch = records[i:i+batch_size]
        collection.insert_many(batch)  # Simpan batch ke database

        # Hitung progres
        progress = int(((i + len(batch)) / total) * 100)
        progress_bar.progress(progress)  # Perbarui progres bar
        status_text.text(f"Menyimpan data... {progress}%")  # Update teks status

        time.sleep(0.01)  # Opsional: Simulasi delay untuk visualisasi progres

    # Hapus progres bar dan tampilkan status selesai
    progress_bar.empty()
    status_text.success("Data berhasil diunggah!")

# Fungsi untuk menyimpan history, mendukung satu atau dua gambar
def save_history(*args, collection_name):
    with st.spinner("Menyimpan riwayat data..."):
        history_data = {}

        # Proses setiap argumen (data1, data2, dst.) untuk figure
        for i, data in enumerate(args):
            if isinstance(data, plt.Figure):
                # Mengonversi figure ke dalam format gambar (base64)
                img_stream = io.BytesIO()
                data.savefig(img_stream, format='png')  # Menyimpan figure sebagai PNG
                img_stream.seek(0)
                img_data = base64.b64encode(img_stream.read()).decode('utf-8')

                # Menyimpan gambar ke dalam dictionary dengan nama yang sesuai
                history_data[f"fig_{i+1}"] = img_data
            else:
                # Jika data bukan figure, misalnya DataFrame atau jenis lainnya
                history_data[f"data_{i+1}"] = data.to_dict('records')

        # Menyimpan riwayat ke dalam koleksi database
        history = {
            "collection_name": collection_name,
            "data": history_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        collection_history.insert_one(history)

# Fungsi untuk memuat riwayat hasil data dari MongoDB
def load_history(collection_name=None):
    with st.spinner("Memuat riwayat data..."):
        if collection_name:
            history = pd.DataFrame(list(collection_history.find({"collection_name": collection_name})))
        else:
            history = pd.DataFrame(list(collection_history.find()))
    return history

# Fungsi untuk menghapus riwayat hasil data dari MongoDB
def delete_history(history_id):
    with st.spinner("Menghapus riwayat data..."):
        collection_history.delete_one({'_id': history_id})

# Fungsi untuk summary data
def visualisasi_data(data):
    with st.spinner("Memproses data dan membuat summary..."):
        required_columns = ['Sampah Mudah Terurai', 'Kertas', 'Plastik', 'B3 Domestik', 'Tonase', 'Kecamatan / Lokasi Kerja', 'Bulan']
        for col in required_columns:
            if col not in data.columns:
                st.error(f"Kolom '{col}' tidak ditemukan dalam data.")
                return

        # Ganti titik dengan kosong dan koma dengan titik untuk konversi ke float
        data['Tonase'] = data['Tonase'].str.replace('.', '').str.replace(',', '.').astype(float)
        data['Sampah Mudah Terurai'] = data['Sampah Mudah Terurai'].str.replace('.', '').str.replace(',', '.').astype(float)
        data['Kertas'] = data['Kertas'].str.replace('.', '').str.replace(',', '.').astype(float)
        data['Plastik'] = data['Plastik'].str.replace('.', '').str.replace(',', '.').astype(float)
        data['B3 Domestik'] = data['B3 Domestik'].str.replace('.', '').str.replace(',', '.').astype(float)

        data['Sampah Organik'] = data['Sampah Mudah Terurai'] + data['Kertas']
        data['Sampah Non-Organik'] = data['Plastik'] + data['B3 Domestik']

        summary = data.describe()
        summary = summary.transpose()

    return summary

def show_image_from_history(img_data):
    img = base64.b64decode(img_data)  # Mendekodekan base64
    st.image(img, use_container_width=True)

# Fungsi untuk prediksi ARIMA
def arima_prediction(data, years=1):
    with st.spinner("Membuat prediksi ARIMA..."):
        if 'Tonase' not in data.columns:
            st.error("Kolom 'Tonase' tidak ditemukan dalam data.")
            return

        data['Tonase'] = data['Tonase'].fillna(0).astype(str).replace('', '0')
        data['Tonase'] = data['Tonase'].str.replace('.', '').str.replace(',', '.').astype(float)

        # Data time series
        time_series = data['Tonase']

        # Uji stasioneritas dan transformasi jika perlu
        time_series = uji_adf(time_series)

        # 5. **Prediksi dan Perhitungan RMSE**
        # Tentukan data training dan testing
        train_size = int(len(time_series) * 0.8)
        train, test = time_series[:train_size], time_series[train_size:]

        # Fitting ulang pada data training
        model_train = ARIMA(train, order=(5, 1, 0))
        model_fit_train = model_train.fit()
        forecast_rmse = model_fit_train.forecast(steps=len(test))

        rmse = np.sqrt(mean_squared_error(test, forecast_rmse))
        st.write(f"\nRMSE (Root Mean Squared Error): {rmse}")

        months = years * 12

        model = ARIMA(time_series, order=(5, 1, 0))
        model_fit = model.fit()
        summary = model_fit.summary()
        forecast = model_fit.forecast(steps=months)

        fig, ax = plt.subplots()
        ax.plot(forecast, marker='o', linestyle='-', label='Forecast')
        ax.set_title('Prediksi ARIMA')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Tonase')
        ax.legend()

        st.pyplot(fig)

    return fig, summary

# Fungsi uji ADF
def uji_adf(time_series):
    # st.write("Uji ADF untuk Stasioneritas:")
    adf_test = adfuller(time_series.dropna())  # Drop NaN sebelum uji ADF
    # st.write(f"ADF Statistic: {adf_test[0]}")
    # st.write(f"p-value: {adf_test[1]}")
    # st.write("Critical Values:")
    # for key, value in adf_test[4].items():
    #     st.write(f"{key}: {value}")

    # if adf_test[1] <= 0.05:
    #     st.write("\nData stasioner (p-value <= 0.05)")
    # else:
    #     st.write("\nData tidak stasioner (p-value > 0.05). Pertimbangkan diferensiasi!")

    # 2. **Transformasi Diferensiasi Jika Tidak Stasioner**
    if adf_test[1] > 0.05:
        time_series_diff = time_series.diff().dropna()
    else:
        time_series_diff = time_series

    return time_series_diff

# Fungsi untuk prediksi KMeans
def kmeans_prediction(data, years=1):
    with st.spinner("Membuat prediksi KMeans..."):
        # Salin data asli untuk memastikan data tidak tercampur
        data = data.copy()

        # Preprocessing data
        for col in ['Sampah Mudah Terurai', 'Kertas', 'Plastik', 'Logam', 'Kaca', 'B3 Domestik']:
            data[col] = data[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

        # Feature engineering
        data['Sampah Organik'] = data['Sampah Mudah Terurai'] + data['Kertas']
        data['Sampah Non-Organik'] = data['Plastik'] + data['B3 Domestik'] + data['Logam'] + data['Kaca']

        # Scaling data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[['Sampah Organik', 'Sampah Non-Organik']])

        # KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(data_scaled)

        # Future predictions
        months = years * 12
        future_data = pd.DataFrame({
            'Sampah Organik': [data['Sampah Organik'].mean()] * months,
            'Sampah Non-Organik': [data['Sampah Non-Organik'].mean()] * months
        })
        future_data_scaled = scaler.transform(future_data)
        predictions = kmeans.predict(future_data_scaled)

        # Plotting results
        fig, ax = plt.subplots()

        # Warna cluster
        cluster_colors = ['blue', 'green', 'orange']
        for cluster in range(3):
            cluster_points = data_scaled[kmeans.labels_ == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       label=f'Cluster {cluster + 1}',
                       color=cluster_colors[cluster])

        # Menampilkan centroid
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   color='red', marker='X', s=200, label='Centroid')

        ax.set_title('Prediksi KMeans')
        ax.set_xlabel('Sampah Organik')
        ax.set_ylabel('Sampah Non-Organik')
        ax.legend()

        st.pyplot(fig)

        elbow_fig = elbow(data)
        # elbow_fig = elbow_silhouette(data)

        # Ambil centroid dari model KMeans
        centroids = kmeans.cluster_centers_
        # Denormalisasi centroid ke skala asli
        centroids_original = scaler.inverse_transform(centroids)
        # Buat DataFrame untuk centroids
        centroids_df = pd.DataFrame(
            centroids_original,
            columns=['Sampah Organik', 'Sampah Non-Organik']
        )
        # Tentukan label kluster (C1, C2, C3) berdasarkan analisis pola
        centroids_df['Cluster'] = ['C1', 'C2', 'C3']  # Sesuaikan urutan label sesuai hasil observasi Anda
        centroids_df = centroids_df.set_index('Cluster')
        st.subheader("Centroid:")
        st.write(centroids_df)

        # Tambahkan data hasil clustering ke dataframe asli
        data['Cluster'] = kmeans.labels_
        # Beri nama cluster sesuai tabel summary
        data['Cluster'] = data['Cluster'].map({0: 'C1', 1: 'C2', 2: 'C3'})
        # Pisahkan data berdasarkan cluster
        c1 = data[data['Cluster'] == 'C1']
        c2 = data[data['Cluster'] == 'C2']
        c3 = data[data['Cluster'] == 'C3']
        # Cetak data wilayah tiap cluster
        st.subheader("Wilayah Cluster C1:")
        st.write(c1[['Kecamatan / Lokasi Kerja', 'Kota / Kabupaten']])

        st.subheader("\nWilayah Cluster C2:")
        st.write(c2[['Kecamatan / Lokasi Kerja', 'Kota / Kabupaten']])

        st.subheader("\nWilayah Cluster C3:")
        st.write(c3[['Kecamatan / Lokasi Kerja', 'Kota / Kabupaten']])

        # Mapping deskripsi cluster
        cluster_description = {
            'C1': 'Wilayah dengan volume sampah kecil',
            'C2': 'Wilayah dengan volume sampah sangat besar',
            'C3': 'Wilayah dengan volume sampah sedang'
        }
        # Tambahkan deskripsi ke data
        data['Deskripsi Cluster'] = data['Cluster'].map(cluster_description)
        # Pisahkan data berdasarkan cluster
        c1 = data[data['Cluster'] == 'C1']
        c2 = data[data['Cluster'] == 'C2']
        c3 = data[data['Cluster'] == 'C3']

        # Tampilkan data wilayah setiap cluster
        st.subheader("Cluster C1 - Wilayah dengan volume sampah kecil:")
        st.write(c1[['Kecamatan / Lokasi Kerja', 'Kota / Kabupaten', 'Deskripsi Cluster']])

        st.subheader("\nCluster C2 - Wilayah dengan volume sampah sangat besar:")
        st.write(c2[['Kecamatan / Lokasi Kerja', 'Kota / Kabupaten', 'Deskripsi Cluster']])

        st.subheader("\nCluster C3 - Wilayah dengan volume sampah sedang:")
        st.write(c3[['Kecamatan / Lokasi Kerja', 'Kota / Kabupaten',  'Deskripsi Cluster']])

    return fig, elbow_fig

# Fungsi untuk menghitung Elbow dan Silhouette Score
# def elbow_silhouette(data):
#     with st.spinner("Membuat Elbow dan Silhouette Score..."):
#         scaler = StandardScaler()
#         data_scaled = scaler.fit_transform(data[['Sampah Organik', 'Sampah Non-Organik']])

#         distortions = []
#         silhouette_scores = []
#         K = range(2, 11)
#         for k in K:
#             kmeans = KMeans(n_clusters=k)
#             kmeans.fit(data_scaled)
#             distortions.append(kmeans.inertia_)
#             silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

#         fig, ax = plt.subplots(1, 2, figsize=(15, 5))

#         # Plot Elbow Method
#         ax[0].plot(K, distortions, 'bx-')
#         ax[0].set_title('Metode Elbow')
#         ax[0].set_xlabel('Jumlah Cluster')
#         ax[0].set_ylabel('Distorsi')

#         # Plot Silhouette Score
#         ax[1].plot(K, silhouette_scores, 'bx-')
#         ax[1].set_title('Silhouette Score')
#         ax[1].set_xlabel('Jumlah Cluster')
#         ax[1].set_ylabel('Silhouette Score')

#         st.pyplot(fig)

#     return fig

def elbow(data):
    with st.spinner("Membuat Elbow..."):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[['Sampah Organik', 'Sampah Non-Organik']])

        distortions = []
        K = range(2, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data_scaled)
            distortions.append(kmeans.inertia_)

        # Plot Elbow Method
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(K, distortions, 'bx-')
        ax.set_title('Metode Elbow')
        ax.set_xlabel('Jumlah Cluster')
        ax.set_ylabel('Distorsi')

        st.pyplot(fig)

    return fig

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Username atau password salah")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'save_data' not in st.session_state:
        st.session_state['save_data'] = False
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        st.sidebar.title("Navigasi")
        page = st.sidebar.radio("Pilih Halaman", ["Data", "Time", "Area", "Log Out"])

        data_imported = collection_visualisasi.count_documents({}) > 0

        if page == "Data":
            st.title("Data CSV")

            # Cek jika file sudah di-upload sebelumnya
            if 'save_data' not in st.session_state:
                st.session_state['save_data'] = False

            uploaded_file = st.file_uploader("Upload CSV", type="csv")

            # Cek jika file di-upload dan belum ada data yang disimpan
            if uploaded_file is not None and not st.session_state['save_data']:
                data = pd.read_csv(uploaded_file, sep=';')
                data.dropna(inplace=True)

                # Simpan data
                save_data(collection_visualisasi, data)

                # Set status sudah disimpan
                st.session_state['save_data'] = True

                # Reset uploader untuk mencegah upload ulang di halaman ini
                uploaded_file = None

                st.subheader("Summary Data")
                summary = visualisasi_data(data)
                if summary is not None:
                    st.write(summary)
                    save_history(summary, collection_name="visualisasi")

            # Jika data sudah disimpan, tampilkan opsi untuk reset jika perlu
            elif st.session_state['save_data']:
                if st.button("Reset Upload"):
                    st.session_state['save_data'] = False
                    st.info("Upload data telah direset.")
                    st.rerun()

            st.title("Data Collection")
            total_rows = collection_visualisasi.count_documents({})
            st.write(f"Total data saat ini: **{total_rows} baris**")
            if st.button("Kosongkan Collection Visualisasi"):
                clear_collection(collection_visualisasi)
                st.rerun()

            st.title("Riwayat Visualisasi")
            history = load_history('visualisasi')
            if history.empty:
                st.warning("Belum ada riwayat visualisasi.")
            else:
                if st.button("Kosongkan Riwayat"):
                    delete_history_by_collection_name("visualisasi")
                    st.rerun()

                for index, row in history.iterrows():
                    st.write(f"{row['timestamp']} - {row['collection_name']}")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if st.button(f"Lihat Hasil {row['_id']}"):
                            st.dataframe(row['data']['data_1'])
                    with col2:
                        if st.button(f"Hapus {row['_id']}"):
                            delete_history(row['_id'])
                            st.success(f"Riwayat {row['_id']} berhasil dihapus")
                            time.sleep(1)
                            st.rerun()

        elif page == "Time":
            if not data_imported:
                st.warning("Harap impor data CSV terlebih dahulu di halaman Data.")
            else:
                st.title("Prediksi ARIMA")

                # Gunakan data yang sudah dimuat sebelumnya jika ada
                if 'data' not in st.session_state or not st.session_state['data_loaded']:
                    # Memuat data jika belum dimuat
                    data = load_data(collection_visualisasi)
                    st.session_state['data'] = data  # Simpan data di session_state
                    st.session_state['data_loaded'] = True  # Set flag data loaded

                # Mengambil data dari session_state jika sudah dimuat
                data = st.session_state['data'].copy()

                years = st.slider("Pilih jumlah tahun prediksi", 1, 2, 1)

                if st.button("Lihat Hasil"):
                    fig, summary = arima_prediction(data, years=years)
                    st.write(summary)
                    save_history(fig, collection_name="arima")

                st.title("Riwayat Prediksi ARIMA")
                history = load_history('arima')
                if history.empty:
                    st.warning("Belum ada riwayat prediksi.")
                else:
                    if st.button("Kosongkan Riwayat"):
                        delete_history_by_collection_name("arima")
                        st.rerun()

                    for index, row in history.iterrows():
                        st.write(f"{row['timestamp']} - {row['collection_name']}")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button(f"Lihat Hasil {row['_id']}"):
                                if 'fig_1' in row['data']:
                                    fig_base64 = row['data']['fig_1']
                                    show_image_from_history(fig_base64)
                        with col2:
                            if st.button(f"Hapus {row['_id']}"):
                                delete_history(row['_id'])
                                st.success(f"Riwayat {row['_id']} berhasil dihapus")
                                time.sleep(1)
                                st.rerun()

        elif page == "Area":
            if not data_imported:
                st.warning("Harap impor data CSV terlebih dahulu di halaman Data.")
            else:
                st.title("Prediksi KMeans")

                # Gunakan data yang sudah dimuat sebelumnya jika ada
                if 'data' not in st.session_state or not st.session_state['data_loaded']:
                    # Memuat data jika belum dimuat
                    data = load_data(collection_visualisasi)
                    st.session_state['data'] = data  # Simpan data di session_state
                    st.session_state['data_loaded'] = True  # Set flag data loaded

                # Mengambil data dari session_state jika sudah dimuat
                data = st.session_state['data'].copy()

                years = st.slider("Pilih jumlah tahun prediksi", 1, 2, 1)

                if st.button("Lihat Hasil"):
                    fig, fig_elbow = kmeans_prediction(data, years=years)
                    save_history(fig, fig_elbow, collection_name="kmeans")

                st.title("Riwayat Prediksi KMeans")
                history = load_history('kmeans')
                if history.empty:
                    st.warning("Belum ada riwayat prediksi KMeans.")
                else:
                    if st.button("Kosongkan Riwayat"):
                        delete_history_by_collection_name("kmeans")
                        st.rerun()

                    for index, row in history.iterrows():
                        st.write(f"{row['timestamp']} - {row['collection_name']}")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button(f"Lihat Hasil {row['_id']}"):
                                # Mengambil gambar dari riwayat yang disimpan
                                if 'fig_1' in row['data']:
                                    fig_base64 = row['data']['fig_1']
                                    show_image_from_history(fig_base64)
                                if 'fig_2' in row['data']:
                                    elbow_fig_base64 = row['data']['fig_2']
                                    show_image_from_history(elbow_fig_base64)
                        with col2:
                            if st.button(f"Hapus {row['_id']}"):
                                delete_history(row['_id'])
                                st.success(f"Riwayat {row['_id']} berhasil dihapus")
                                time.sleep(1)
                                st.rerun()

        elif page == "Log Out":
            st.session_state['logged_in'] = False
            st.rerun()

if __name__ == "__main__":
	main()
