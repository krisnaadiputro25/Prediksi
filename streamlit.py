import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import time
from pymongo import MongoClient
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import itertools

# Koneksi ke MongoDB Atlas
client = MongoClient("mongodb+srv://krisna:krisna@cluster0.3mao11f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['dbkrisna']
collection_visualisasi = db['visualisasi']
collection_history = db['history']

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

def show_image_from_history(img_data):
    img = base64.b64decode(img_data)  # Mendekodekan base64
    st.image(img, use_container_width=True)

# Fungsi untuk prediksi ARIMA
def arima_prediction(data):
    with st.spinner("Membuat prediksi ARIMA..."):
        # time series conversion
        month_mapping = {
            "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
            "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
            "September": 9, "Oktober": 10, "November": 11, "Desember": 12
        }

        df_arima = data.copy()
        df_arima['month'] = df_arima['Bulan'].map(month_mapping)
        df_arima['year'] = df_arima['Tahun'].astype(int)
        df_arima['date'] = pd.to_datetime(
            df_arima[['year', 'month']].assign(Day=1), errors='coerce'
        )

        df_arima = pd.DataFrame(df_arima.groupby('date')['Tonase'].sum())
        ts_data = df_arima['Tonase']

        st.write("Hasil ADF Test sebelum differencing:")
        adf_test(ts_data)

        if adfuller(ts_data)[1] > 0.05:
            ts_data = ts_data.diff().dropna()
            st.write("\nHasil ADF Test setelah differencing:")
            adf_test(ts_data)
        else:
            pass

        # Train-test split
        train_size = int(len(ts_data) * 0.95)
        train, test = ts_data[:train_size], ts_data[train_size:]

        # mencari best ARIMA parameter
        # p_values = range(0, 10)
        # d_values = range(0, 10)
        # q_values = range(0, 10)

        # # iterate combinations
        # pdq_combinations = list(itertools.product(p_values, d_values, q_values))
        # best_score, best_params = float("inf"), None

        # for params in pdq_combinations:
        #     try:

        #         model = ARIMA(train, order=params)
        #         model_fit = model.fit()
        #         forecast = model_fit.forecast(steps=len(test))
        #         forecast_series = pd.Series(forecast, index=test.index)

        #         # MAPE
        #         mape = round(np.mean(np.abs((test - forecast_series) / test)) * 100, 3)

        #         # Cari nilai MAPE terendah
        #         if mape < best_score:
        #             best_score, best_params = mape, params

        #     except:
        #         continue

        # ARIMA model
        p, d, q = 9, 5, 4
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()

        # forecast
        forecast = model_fit.forecast(steps=len(test))
        forecast_series = pd.Series(forecast, index=test.index)

        # evaluation
        mae = mean_absolute_error(test, forecast_series)
        mape = np.mean(np.abs((test - forecast_series) / test)) * 100
        rmse = np.sqrt(mean_squared_error(test, forecast_series))

        st.write("\nEvaluasi Model:")
        st.write(f"MAE: {mae}")
        st.write(f"MAPE: {mape}%")
        st.write(f"RMSE: {rmse}")

        # visualization
        full_series = pd.concat([train, test])
        forecast_series_full = pd.concat([train, forecast_series])

        # Membuat plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast_series_full, label="Forecast", color='red')
        ax.plot(train, label="Actual", color='blue')
        ax.set_title("ARIMA Model - Actual vs Forecast")
        ax.set_xlabel(None)
        ax.set_ylabel("Tonase")
        ax.legend()
        ax.grid()

        # Menampilkan plot di Streamlit
        st.pyplot(fig)

    return fig

# ADF Test
def adf_test(series):
    result = adfuller(series)
    st.write("ADF Test Statistic:", result[0])
    st.write("p-value:", result[1])
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write(f"   {key}: {value}")
    if result[1] <= 0.05:
        st.write("Data stasioner (H0 ditolak).")
    else:
        st.write("Data tidak stasioner (H0 diterima).")

# Fungsi untuk prediksi KMeans
def kmeans_prediction(data):
    with st.spinner("Membuat prediksi KMeans..."):
        # Salin data asli untuk memastikan data tidak tercampur
        data = data.copy()

        # scaling
        features = ['kota_kabupaten_code', 'tahun_code', 'bulan_code', 'Tonase']
        X = data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow Method
        inertia = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        fig_elbow, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_range, inertia, marker='o')
        ax.set_title("Elbow Method")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia")
        ax.set_xticks(k_range)
        ax.grid()
        st.pyplot(fig_elbow)

        # Silhouette Score
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            labels = kmeans.labels_
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)

        fig_silhoute, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_range, silhouette_scores, marker='o', color='orange')
        ax.set_title("Silhouette Score")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_xticks(k_range)
        ax.grid()
        st.pyplot(fig_silhoute)

        # optimal silhouette score
        optimal_k_1 = k_range[np.argmax(silhouette_scores)]
        st.write(f'Jumlah cluster optimal berdasarkan Silhouette Score: {optimal_k_1}')

        # optimal inertia score
        optimal_k_2 = np.argmax(np.diff(np.diff(inertia))) + 2
        st.write(f'Jumlah cluster optimal berdasarkan Inertia Score: {optimal_k_2}')

        # K-Means (silhouette score)
        optimal_k = optimal_k_1
        kmeans_optimal_silhoutte = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_optimal_silhoutte.fit(X_scaled)
        data['cluster_silhouette'] = kmeans_optimal_silhoutte.labels_

        # K-Means (inertia score)
        optimal_k = optimal_k_2
        kmeans_optimal_inertia = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_optimal_inertia.fit(X_scaled)
        data['cluster_inertia'] = kmeans_optimal_inertia.labels_

        # Calinski-Harabasz Index
        score = calinski_harabasz_score(X_scaled, kmeans_optimal_silhoutte.labels_)
        st.write(f"CHI Silhouette: {score}")

        score = calinski_harabasz_score(X_scaled, kmeans_optimal_inertia.labels_)
        st.write(f"CHI Inertia: {score}")

        # Visualisasi dengan PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Scatter Plot - Clustering (Silhouette)
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_optimal_silhoutte.labels_, cmap='viridis')
        axes[0].set_title('Clustering (Silhouette)')
        axes[0].set_xlabel('PCA Component 1')
        axes[0].set_ylabel('PCA Component 2')

        # Scatter Plot - Clustering (Inertia)
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_optimal_inertia.labels_, cmap='viridis')
        axes[1].set_title('Clustering (Inertia)')
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')

        # Menyesuaikan layout
        plt.tight_layout()

        # Menampilkan plot di Streamlit
        st.pyplot(fig)

        # Statistik deskriptif untuk setiap klaster
        cluster_summary = data.groupby('cluster_inertia').agg(
            Kota_Kabupaten=('Kota / Kabupaten', lambda x: x.mode()[0]),
            Tahun_Min=('Tahun', 'min'),
            Tahun_Max=('Tahun', 'max'),
            Tahun_Avg=('Tahun', 'mean'),
            Bulan_Mode=('Bulan', lambda x: x.mode()[0]),
            Tonase_Mean=('Tonase', 'mean'),
            Tonase_Median=('Tonase', 'median'),
            Tonase_Std=('Tonase', 'std')
        ).reset_index()

        cluster_summary

    return fig_elbow, fig_silhoute, fig, cluster_summary

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
            st.title("Data XLS")

            # Cek jika file sudah di-upload sebelumnya
            if 'save_data' not in st.session_state:
                st.session_state['save_data'] = False

            uploaded_file = st.file_uploader("Upload XLS", type="xlsx")

            # Cek jika file di-upload dan belum ada data yang disimpan
            if uploaded_file is not None and not st.session_state['save_data']:
                data = pd.read_excel(uploaded_file)

                # processing
                df = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                df = df.applymap(lambda x: x.capitalize() if isinstance(x, str) else x)
                df = df[['Kota / Kabupaten', 'Tahun', 'Bulan', 'Tonase']]
                df = df[(df['Kota / Kabupaten'].isna() == False) & (df['Kota / Kabupaten'] != 'Lembaga')]
                df = df[(df['Tonase'] >= 100) & (df['Tonase'] <= 10000)]

                # encode
                encoder = LabelEncoder()
                df['kota_kabupaten_code'] = encoder.fit_transform(df['Kota / Kabupaten'])
                df['tahun_code'] = encoder.fit_transform(df['Tahun'])
                df['bulan_code'] = encoder.fit_transform(df['Bulan'])

                # Simpan data
                save_data(collection_visualisasi, df)

                # Set status sudah disimpan
                st.session_state['save_data'] = True

                # Reset uploader untuk mencegah upload ulang di halaman ini
                uploaded_file = None

                st.subheader("Dataset Head")
                st.write(df.head())
                save_history(df.head(), collection_name="visualisasi")

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
                st.warning("Harap impor data XLS terlebih dahulu di halaman Data.")
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

                if st.button("Tampilkan Prediksi ARIMA"):
                    plt = arima_prediction(data)
                    save_history(plt, collection_name="arima")

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

                if st.button("Tampilkan Prediksi KMeans"):
                    fig_elbow, fig_silhoute, fig, cluster_summary = kmeans_prediction(data)
                    save_history(fig, fig_elbow, fig_silhoute, cluster_summary, collection_name="kmeans")

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
                                if 'fig_3' in row['data']:
                                    elbow_fig_base64 = row['data']['fig_3']
                                    show_image_from_history(elbow_fig_base64)
                                if 'data_4' in row['data']:
                                    df = pd.DataFrame(row['data']['data_4'])  # Konversi dictionary kembali ke DataFrame
                                    st.dataframe(df)
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