import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

RAW_COLUMNS = [
    'Timestamp', 'Age', 'Gender', 'Study_Hours', 'Class_Attendance',
    'Tuition', 'Exam_Frequency', 'Assignment_Load', 'Sleep_Hours',
    'Physical_Exercise', 'Social_Media_Use', 'Screen_Time',
    'Family_Income_Level', 'Peer_Pressure', 'Family_Support',
    'Anxiety_Level', 'University_Type', 'Stress_Score', 'Stress_Level'
]

NUMERIC_COLS = [
    'Age', 'Study_Hours', 'Class_Attendance', 'Exam_Frequency',
    'Assignment_Load', 'Sleep_Hours', 'Social_Media_Use', 'Screen_Time',
    'Peer_Pressure', 'Family_Support', 'Anxiety_Level'
]

NOMINAL_COLS = ['Gender', 'Tuition', 'Physical_Exercise', 'University_Type']

ORDINAL_MAPS = {
    'Family_Income_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Stress_Level': {'Low': 0, 'Medium': 1, 'High': 2}
}

LEVEL_ORDER = ['Low', 'Medium', 'High']

VAR_INFO = {
    "Age": "Usia mahasiswa dalam satuan tahun. Variabel ini digunakan untuk melihat apakah kelompok usia tertentu lebih rentan terhadap tingkat stres yang lebih tinggi dibandingkan kelompok lainnya.",
    "Study_Hours": "Rata-rata jumlah jam belajar mahasiswa per harinya. Durasi belajar yang terlalu tinggi atau terlalu rendah sama-sama berpotensi meningkatkan tingkat stres akademik.",
    "Class_Attendance": "Persentase kehadiran mahasiswa di kelas. Kehadiran yang rendah sering kali mencerminkan motivasi belajar yang menurun dan dapat menjadi indikator awal munculnya stres.",
    "Exam_Frequency": "Frekuensi ujian yang dihadapi mahasiswa dalam satu semester. Semakin sering ujian, semakin besar tekanan akademik yang dirasakan.",
    "Assignment_Load": "Tingkat beban tugas akademik yang diterima mahasiswa, diukur dalam skala 1 sampai 9. Beban tugas yang berlebihan berkaitan langsung dengan peningkatan stres.",
    "Tuition": "Status apakah mahasiswa mengikuti bimbingan belajar atau les tambahan di luar perkuliahan reguler. Variabel ini mencerminkan adanya beban akademik tambahan.",
    "Sleep_Hours": "Rata-rata jam tidur mahasiswa per hari. Durasi tidur yang kurang dari enam jam terbukti menurunkan kemampuan kognitif dan meningkatkan kerentanan terhadap stres.",
    "Physical_Exercise": "Status apakah mahasiswa melakukan aktivitas fisik secara rutin. Olahraga diketahui sebagai salah satu mekanisme efektif untuk mengurangi tingkat stres.",
    "Social_Media_Use": "Rata-rata durasi penggunaan media sosial per hari dalam satuan jam. Penggunaan berlebihan dikaitkan dengan peningkatan kecemasan dan gangguan konsentrasi belajar.",
    "Screen_Time": "Total durasi penggunaan perangkat layar per hari, mencakup ponsel, laptop, dan televisi. Merupakan faktor dengan pengaruh terbesar terhadap tingkat stres berdasarkan hasil feature importance.",
    "Family_Income_Level": "Tingkat pendapatan keluarga mahasiswa, dikategorikan menjadi Low, Medium, dan High. Kondisi ekonomi keluarga berpengaruh terhadap tekanan psikologis yang dirasakan mahasiswa di luar konteks akademik.",
    "Peer_Pressure": "Tingkat tekanan yang dirasakan mahasiswa dari lingkungan pertemanan, diukur dalam skala 1 sampai 9. Kompetisi sosial yang tidak sehat dapat memperburuk kondisi psikologis mahasiswa.",
    "Family_Support": "Tingkat dukungan yang diterima mahasiswa dari keluarga, diukur dalam skala 1 sampai 9. Semakin tinggi dukungan keluarga, semakin rendah tingkat stres yang dirasakan.",
    "Anxiety_Level": "Tingkat kecemasan yang dirasakan mahasiswa, diukur dalam skala 1 sampai 9. Variabel ini berkorelasi kuat dengan Screen_Time dan Assignment_Load.",
    "Gender": "Jenis kelamin mahasiswa. Laki-laki dan perempuan menunjukkan pola respons stres yang berbeda secara psikologis.",
    "University_Type": "Jenis universitas tempat mahasiswa berkuliah, apakah negeri, swasta, atau nasional. Setiap tipe memiliki tingkat kompetisi akademik yang berbeda.",
}


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), encoding='latin1')

    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"Jumlah kolom pada file tidak sesuai. Diharapkan {len(RAW_COLUMNS)} kolom, "
            f"ditemukan {df.shape[1]} kolom."
        )

    df.columns = RAW_COLUMNS
    df = df.drop(columns=['Timestamp', 'Stress_Score'])

    for col, mapping in ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping)

    if df[list(ORDINAL_MAPS.keys())].isnull().values.any():
        raise ValueError(
            "Terdapat nilai pada kolom Stress_Level atau Family_Income_Level yang tidak "
            "sesuai kategori Low, Medium, atau High."
        )

    encoders = {}
    for col in NOMINAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    jumlah_sebelum = len(df)
    mask = pd.Series(True, index=df.index)
    for level in df['Stress_Level'].unique():
        subset = df[df['Stress_Level'] == level]
        for col in NUMERIC_COLS:
            q1 = subset[col].quantile(0.25)
            q3 = subset[col].quantile(0.75)
            iqr = q3 - q1
            batas_bawah = q1 - 1.5 * iqr
            batas_atas = q3 + 1.5 * iqr
            idx_outlier = subset[(subset[col] < batas_bawah) | (subset[col] > batas_atas)].index
            mask.loc[idx_outlier] = False
    df = df[mask].reset_index(drop=True)
    jumlah_sesudah = len(df)

    corr = df.corr()

    X = df.drop(columns=['Stress_Level'])
    y = df['Stress_Level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        "model": model,
        "encoders": encoders,
        "feature_cols": X.columns.tolist(),
        "accuracy": accuracy,
        "y_test": y_test,
        "y_pred": y_pred,
        "fi": fi,
        "corr": corr,
        "jumlah_sebelum": jumlah_sebelum,
        "jumlah_sesudah": jumlah_sesudah,
    }


DEFAULT_MODEL_PATH = "rf_model.pkl"
DEFAULT_META_PATH = "rf_meta.pkl"


def normalize_meta(model, meta):
    """
    Menyesuaikan isi meta dari berbagai kemungkinan sumber (misalnya hasil
    export notebook yang memakai nama key 'feature_importance', bukan 'fi',
    dan tidak menyimpan matriks korelasi atau jumlah data). Key yang tidak
    tersedia diisi None dan ditangani secara terpisah di tampilan.
    """
    return {
        "model": model,
        "encoders": meta.get("encoders", {}),
        "feature_cols": meta.get("feature_cols", []),
        "accuracy": meta.get("accuracy"),
        "y_test": meta.get("y_test"),
        "y_pred": meta.get("y_pred"),
        "fi": meta.get("fi", meta.get("feature_importance")),
        "corr": meta.get("corr"),
        "jumlah_sebelum": meta.get("jumlah_sebelum"),
        "jumlah_sesudah": meta.get("jumlah_sesudah"),
    }


@st.cache_resource(show_spinner=False)
def load_default_model():
    model = joblib.load(DEFAULT_MODEL_PATH)
    meta = joblib.load(DEFAULT_META_PATH)
    return normalize_meta(model, meta)


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] * {
        color: #0f172a;
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.3;
        margin-bottom: 4px;
    }

    .sidebar-caption {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 16px;
    }

    .section-label {
        font-size: 12px;
        font-weight: 600;
        color: #9ca3af;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 16px 0 8px 0;
    }

    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .main-subtitle {
        font-size: 15px;
        color: #64748b;
        margin-top: 0;
        margin-bottom: 20px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 6px;
    }

    .metric-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    .metric-value-blue {
        font-size: 38px;
        font-weight: 700;
        color: #2563eb;
        line-height: 1.1;
    }

    .metric-value-dark {
        font-size: 38px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }

    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    .chart-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        min-height: 320px;
    }

    .chart-title {
        font-size: 17px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 16px;
    }

    .chart-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 48px 12px;
        color: #94a3b8;
        font-size: 13px;
        text-align: center;
    }

    .form-section-title {
        font-size: 15px;
        font-weight: 700;
        color: #2563eb;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    .predict-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 28px 32px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .predict-subtitle {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 24px;
    }

    .about-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
        font-size: 13px;
        color: #374151;
    }

    .stSlider label, .stSelectbox label, .stNumberInput label {
        font-size: 13px !important;
        color: #374151 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 0;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #1d4ed8;
    }

    .stRadio > div {
        gap: 4px;
    }

    .stRadio label {
        font-size: 14px !important;
    }

    hr {
        border-color: #e2e8f0;
        margin: 12px 0;
    }

    .st-key-upload_dropzone {
        position: relative;
    }

    .st-key-upload_dropzone .stElementContainer {
        position: static !important;
    }

    .st-key-upload_dropzone section[data-testid="stFileUploaderDropzone"] {
        position: absolute !important;
        inset: 0 !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        opacity: 0 !important;
        z-index: 5 !important;
        cursor: pointer !important;
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .st-key-upload_dropzone div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    .custom-upload-visual {
        position: relative;
        z-index: 1;
        border: 2px dashed #3b82f6;
        border-radius: 16px;
        background: #f8fafc;
        padding: 56px 24px;
        text-align: center;
    }

    .custom-upload-icon {
        font-size: 34px;
        color: #3b82f6;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .custom-upload-title {
        font-size: 17px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 4px;
    }

    .custom-upload-caption {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 10px;
    }

    .custom-upload-limit {
        font-size: 12px;
        color: #94a3b8;
    }

    .upload-success {
        margin-top: 12px;
        padding: 10px 16px;
        background: #ecfdf5;
        border: 1px solid #6ee7b7;
        color: #047857;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 600;
        text-align: center;
    }

    .upload-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .upload-heading {
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .upload-caption {
        text-align: center;
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 18px;
    }

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 20px;
        font-size: 14px;
        color: #1e3a8a;
    }

    .info-box b {
        color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">Prediksi Tingkat<br>Stres Mahasiswa</div>', unsafe_allow_html=True)

    menu = st.radio("Menu", ["Beranda", "Prediksi Manual"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Penjelasan Variabel</div>', unsafe_allow_html=True)

    for var, desc in VAR_INFO.items():
        with st.expander(var):
            st.markdown(f'<span style="font-size:13px;color:#374151">{desc}</span>', unsafe_allow_html=True)

    st.markdown(
        '<div class="about-card"><b>Tentang Aplikasi</b><br><br>'
        'Aplikasi ini memprediksi tingkat stres mahasiswa menggunakan algoritma Random Forest. '
        'Upload dataset pada menu Beranda untuk melatih model.</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="main-title">Prediksi Tingkat Stres Mahasiswa</div>', unsafe_allow_html=True)
st.markdown("---")

if "default_result" not in st.session_state:
    try:
        st.session_state.default_result = load_default_model()
    except FileNotFoundError:
        st.session_state.default_result = None

if "uploaded_result" not in st.session_state:
    st.session_state.uploaded_result = None

if menu == "Beranda":
    with st.container(key="upload_dropzone"):
        st.markdown(
            '<div class="custom-upload-visual">'
            '<div class="custom-upload-icon">&#8593;</div>'
            '<div class="custom-upload-title">Upload Dataset</div>'
            '<div class="custom-upload-caption">Klik atau drag file CSV di sini</div>'
            '<div class="custom-upload-limit">200 MB &bull; CSV</div>'
            '</div>',
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                with st.spinner("Memproses dataset dan melatih model..."):
                    st.session_state.uploaded_result = run_pipeline(uploaded_file.getvalue())
            except ValueError as e:
                st.session_state.uploaded_result = None
                st.error(str(e))
            except Exception:
                st.session_state.uploaded_result = None
                st.error("Gagal memproses file. Pastikan format CSV sesuai dengan template yang digunakan.")

    if uploaded_file is not None and st.session_state.uploaded_result is not None:
        st.markdown(
            f'<div class="upload-success">&#10003; {uploaded_file.name} berhasil diupload</div>',
            unsafe_allow_html=True
        )

    result = st.session_state.uploaded_result

    st.markdown('<div class="section-header">Ringkasan Hasil Training & Testing</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        akurasi_val = f"{result['accuracy']*100:.2f}%" if result else "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Akurasi Test</div>
            <div class="metric-value-blue">{akurasi_val}</div>
            <div class="metric-sub">Akurasi model pada data test</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        jumlah_data = result.get('jumlah_sesudah') if result else None
        if jumlah_data is not None:
            total_val = f"{jumlah_data:,}".replace(",", ".")
            total_sub = "Jumlah data setelah cleaning"
        elif result and result.get("y_test") is not None:
            total_val = f"{len(result['y_test']):,}".replace(",", ".")
            total_sub = "Jumlah data uji (data latih tidak tercatat pada model ini)"
        else:
            total_val = "-"
            total_sub = "Jumlah data setelah cleaning"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Data</div>
            <div class="metric-value-blue">{total_val}</div>
            <div class="metric-sub">{total_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        algo_val = "Random Forest" if result else "-"
        algo_size = "28px" if result else "38px"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Algoritma</div>
            <div class="metric-value-dark" style="font-size:{algo_size};color:#2563eb">{algo_val}</div>
            <div class="metric-sub">Model yang digunakan</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Confusion Matrix (Data Test)</div>', unsafe_allow_html=True)

            if result:
                cm = confusion_matrix(result["y_test"], result["y_pred"])
                classes = LEVEL_ORDER

                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("white")

                blues = plt.cm.Blues
                ax.imshow(cm, cmap=blues, aspect="auto")

                ax.set_xticks(np.arange(len(classes)))
                ax.set_yticks(np.arange(len(classes)))
                ax.set_xticklabels(classes, fontsize=11)
                ax.set_yticklabels(classes, fontsize=11)
                ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
                ax.set_ylabel("Actual", fontsize=12, labelpad=10)

                ax.xaxis.set_label_position("top")
                ax.xaxis.tick_top()

                norm = cm.max()
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        val = cm[i, j]
                        text_color = "white" if val > norm * 0.6 else "black"
                        ax.text(j, i, str(val), ha="center", va="center", fontsize=13, fontweight="600", color=text_color)

                colorbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
                sm = plt.cm.ScalarMappable(cmap=blues, norm=plt.Normalize(vmin=0, vmax=norm))
                sm.set_array([])
                fig.colorbar(sm, cax=colorbar_ax)

                plt.tight_layout(rect=[0, 0, 0.9, 1])
                st.pyplot(fig)
                st.download_button(
                    "Unduh Confusion Matrix", data=fig_to_bytes(fig),
                    file_name="confusion_matrix.png", mime="image/png"
                )
            else:
                st.markdown(
                    '<div class="chart-placeholder">Confusion matrix akan tampil di sini '
                    'setelah dataset diupload.</div>',
                    unsafe_allow_html=True
                )

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Feature Importance (Top 10)</div>', unsafe_allow_html=True)

            if result:
                fi_df = result["fi"].sort_values(ascending=False).head(10)

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                fig2.patch.set_facecolor("white")

                bars = ax2.barh(
                    fi_df.index[::-1],
                    fi_df.values[::-1],
                    color="#2563eb",
                    height=0.55,
                    edgecolor="none"
                )

                for bar, val in zip(bars, fi_df.values[::-1]):
                    ax2.text(
                        val + 0.002,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}",
                        va="center",
                        fontsize=9,
                        color="#374151"
                    )

                ax2.set_xlabel("Importance", fontsize=11)
                ax2.set_xlim(0, fi_df.values.max() * 1.22)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                ax2.spines["left"].set_visible(False)
                ax2.tick_params(axis="y", labelsize=10)
                ax2.tick_params(axis="x", labelsize=9)
                ax2.xaxis.grid(True, linestyle="--", alpha=0.5)
                ax2.set_axisbelow(True)

                plt.tight_layout()
                st.pyplot(fig2)
                st.download_button(
                    "Unduh Feature Importance", data=fig_to_bytes(fig2),
                    file_name="feature_importance.png", mime="image/png"
                )
            else:
                st.markdown(
                    '<div class="chart-placeholder">Feature importance akan tampil di sini '
                    'setelah dataset diupload.</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="chart-title">Heatmap Korelasi</div>', unsafe_allow_html=True)

        if result and result.get("corr") is not None:
            corr = result["corr"]

            fig3, ax3 = plt.subplots(figsize=(11, 8))
            fig3.patch.set_facecolor("white")
            im = ax3.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

            ax3.set_xticks(np.arange(len(corr.columns)))
            ax3.set_yticks(np.arange(len(corr.columns)))
            ax3.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
            ax3.set_yticklabels(corr.columns, fontsize=9)

            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    val = corr.values[i, j]
                    text_color = "white" if abs(val) > 0.6 else "black"
                    ax3.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)

            fig3.colorbar(im, ax=ax3, fraction=0.03, pad=0.02)
            plt.tight_layout()
            st.pyplot(fig3)
            st.download_button(
                "Unduh Heatmap Korelasi", data=fig_to_bytes(fig3),
                file_name="heatmap_korelasi.png", mime="image/png"
            )
        else:
            st.markdown(
                '<div class="chart-placeholder">Heatmap korelasi akan tampil disini setelah dataset diupload. </div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="info-box"><b>Cara Penggunaan</b><br><br>'
        '1. Upload dataset atau file CSV Anda pada bagian atas untuk melatih model.<br>'
        '2. Setelah dataset diupload, ringkasan hasil training & testing serta visualisasi Confusion '
        'Matrix, Feature Importance, dan Heatmap Korelasi akan tampil di bawah ini.<br>'
        '3. Menu Prediksi Manual tetap bisa langsung dipakai kapan saja, meski dataset belum diupload '
        'di halaman ini.</div>',
        unsafe_allow_html=True
    )

else:
    result = st.session_state.uploaded_result or st.session_state.default_result

    st.markdown('<div class="section-header">Prediksi Manual</div>', unsafe_allow_html=True)

    if result is None:
        st.markdown(
            '<div class="predict-subtitle">Model belum tersedia. Silakan upload dataset '
            'terlebih dahulu pada menu Beranda sebelum melakukan prediksi manual.</div>',
            unsafe_allow_html=True
        )
    else:
        model = result["model"]
        encoders = result["encoders"]
        feature_cols = result["feature_cols"]

        with st.container(border=True):
            st.markdown(
                '<div class="predict-subtitle">Isi semua data mahasiswa di bawah, lalu klik Prediksi Sekarang. '
                'Penjelasan setiap variabel tersedia di menu sebelah kiri.</div>',
                unsafe_allow_html=True
            )

            with st.form("form_prediksi"):
                st.markdown('<div class="form-section-title">Data Akademik</div>', unsafe_allow_html=True)

                row1a, row1b, row1c = st.columns(3)
                with row1a:
                    age = st.slider("Usia (tahun)", 19, 24, 21)
                    study_hours = st.slider("Jam Belajar / Hari", 0, 12, 3)
                with row1b:
                    attendance = st.slider("Kehadiran Kelas (%)", 40, 99, 80)
                    exam_frequency = st.slider("Frekuensi Ujian (1-9)", 1, 9, 5)
                with row1c:
                    assignment_load = st.slider("Beban Tugas (1-9)", 1, 9, 5)
                    tuition = st.selectbox("Ikut Les / Bimbel?", list(encoders["Tuition"].classes_))

                st.markdown('<div class="form-section-title" style="margin-top:20px">Gaya Hidup</div>', unsafe_allow_html=True)

                row2a, row2b, row2c = st.columns(3)
                with row2a:
                    sleep_hours = st.slider("Jam Tidur / Hari (jam)", 3, 12, 7)
                    screen_time = st.slider("Screen Time / Hari (jam)", 1, 12, 4)
                with row2b:
                    social_media = st.slider("Media Sosial / Hari (jam)", 0, 12, 2)
                    physical_exercise = st.selectbox("Olahraga Rutin?", list(encoders["Physical_Exercise"].classes_))
                with row2c:
                    gender = st.selectbox("Jenis Kelamin", list(encoders["Gender"].classes_))
                    university_type = st.selectbox("Tipe Universitas", list(encoders["University_Type"].classes_))

                st.markdown('<div class="form-section-title" style="margin-top:20px">Kondisi Sosial & Psikologis</div>', unsafe_allow_html=True)

                row3a, row3b, row3c = st.columns(3)
                with row3a:
                    family_income = st.selectbox("Pendapatan Keluarga", list(ORDINAL_MAPS["Family_Income_Level"].keys()))
                with row3b:
                    peer_pressure = st.slider("Tekanan Teman Sebaya (1-9)", 1, 9, 4)
                    family_support = st.slider("Dukungan Keluarga (1-9)", 1, 9, 6)
                with row3c:
                    anxiety_level = st.slider("Tingkat Kecemasan (1-9)", 1, 9, 4)

                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Prediksi Sekarang", width='stretch')

        if submitted:
            row = {
                "Age": age,
                "Gender": encoders["Gender"].transform([gender])[0],
                "Study_Hours": study_hours,
                "Class_Attendance": attendance,
                "Tuition": encoders["Tuition"].transform([tuition])[0],
                "Exam_Frequency": exam_frequency,
                "Assignment_Load": assignment_load,
                "Sleep_Hours": sleep_hours,
                "Physical_Exercise": encoders["Physical_Exercise"].transform([physical_exercise])[0],
                "Social_Media_Use": social_media,
                "Screen_Time": screen_time,
                "Family_Income_Level": ORDINAL_MAPS["Family_Income_Level"][family_income],
                "Peer_Pressure": peer_pressure,
                "Family_Support": family_support,
                "Anxiety_Level": anxiety_level,
                "University_Type": encoders["University_Type"].transform([university_type])[0],
            }

            input_data = pd.DataFrame([row])[feature_cols]
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            inverse_level = {v: k for k, v in ORDINAL_MAPS["Stress_Level"].items()}
            label = inverse_level[prediction]
            confidence = np.max(probabilities) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Hasil Prediksi</div>', unsafe_allow_html=True)

            label_lower = str(label).lower()
            if label_lower == "low":
                bg = "#dcfce7"
                text_color = "#166534"
            elif label_lower == "medium":
                bg = "#fef3c7"
                text_color = "#92400e"
            else:
                bg = "#fee2e2"
                text_color = "#991b1b"

            st.markdown(f"""
            <div style="background:{bg};padding:32px;border-radius:16px;text-align:center;margin-bottom:16px;">
                <div style="font-size:42px;font-weight:700;color:{text_color};margin-bottom:6px;">{label}</div>
                <div style="font-size:16px;color:{text_color};">Confidence: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Stress Level": [inverse_level[0], inverse_level[1], inverse_level[2]],
                "Probability": [f"{probabilities[i]*100:.2f}%" for i in range(3)]
            })

            st.markdown("**Detail Probabilitas**")
            st.dataframe(prob_df, width='stretch', hide_index=True)
