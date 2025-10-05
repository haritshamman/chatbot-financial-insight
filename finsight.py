# --------------------------------------------------------------------------
# --- 1. IMPORT LIBRARY YANG DIBUTUHKAN ---
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# --------------------------------------------------------------------------
# --- 2. KONFIGURASI HALAMAN UTAMA ---
# --------------------------------------------------------------------------
# Set konfigurasi halaman (harus menjadi perintah pertama Streamlit)
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Insight AI: Your Financial Portfolio Analyst by Mister H")
st.markdown("Unggah data portofolio Anda, dan biarkan AI menganalisis, memberikan ringkasan, dan menemukan wawasan tersembunyi.")

# --------------------------------------------------------------------------
# --- 3. SIDEBAR UNTUK INPUT & UPLOAD ---
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Konfigurasi")
    # Input Google AI API Key
    google_api_key = st.text_input("Masukkan Google AI API Key Anda", type="password")
    
    st.divider()

    # File Uploader untuk data portofolio
    uploaded_file = st.file_uploader("Unggah file Excel atau CSV", type=["csv", "xlsx"])
    
    st.divider()
    st.info("Pastikan file Anda memiliki header kolom yang jelas untuk analisis yang akurat.")
    st.info("Contoh kolom: NamaAset, TipeInvestasi, Nilai, TanggalBeli, Sektor, RatingRisiko, ReturnYTD, dll.")


# --------------------------------------------------------------------------
# --- 4. INISIALISASI AGENT & MANAJEMEN STATE ---
# --------------------------------------------------------------------------
# Inisialisasi session state untuk menyimpan pesan dan data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

# Fungsi untuk membuat agent, akan dipanggil saat file diunggah
def initialize_agent(df, api_key):
    """Membuat dan mengembalikan Pandas DataFrame Agent."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            google_api_key=api_key, 
            temperature=0.0 # Suhu 0 untuk jawaban yang lebih deterministik & faktual
        )
        # Membuat agent yang dirancang khusus untuk bekerja dengan Pandas
        return create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            # handle_parsing_errors may not be supported in all versions; keep simple
            allow_dangerous_code=True
        )
    except Exception as e:
        st.error(f"Gagal menginisialisasi agent. Cek API Key Anda. Error: {e}")
        return None

# Cek jika file baru diunggah, lalu buat agent baru
@st.cache_data(show_spinner=False)
def _read_file(uploaded_file):
    """Membaca file upload menjadi DataFrame (cacheable)."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


if uploaded_file is not None and google_api_key:
    try:
        dataframe = _read_file(uploaded_file)

        # Simpan dataframe dan buat agent baru
        st.session_state.dataframe = dataframe
        st.session_state.agent = initialize_agent(dataframe, google_api_key)

        # Hapus riwayat chat lama saat file baru diunggah
        st.session_state.messages = []
        st.success("File berhasil diunggah dan agent analisis siap!")
        
    except Exception as e:
        st.error(f"Gagal memproses file: {e}")
        st.session_state.agent = None

# --------------------------------------------------------------------------
# --- 5. TAMPILAN UTAMA & INTERAKSI CHAT ---
# --------------------------------------------------------------------------
if st.session_state.agent is None:
    st.info("Silakan masukkan API Key dan unggah file data Anda di sidebar untuk memulai analisis.")
else:
    # Menampilkan preview data jika agent sudah siap
    st.subheader("Preview Data Portofolio Anda")
    if st.session_state.dataframe is not None:
        st.dataframe(st.session_state.dataframe.head(10))

        # Ringkasan metrik singkat otomatis
        numeric_cols = st.session_state.dataframe.select_dtypes(include=['number']).columns.tolist()
        col1, col2, col3 = st.columns(3)
        with col1:
            n_assets = len(st.session_state.dataframe)
            st.metric("Jumlah Aset", f"{n_assets}")
        with col2:
            if numeric_cols:
                total_val = st.session_state.dataframe[numeric_cols[0]].sum()
                st.metric("Total (perkiraan)", f"{total_val:,.2f}", delta=None)
            else:
                st.metric("Total (perkiraan)", "N/A")
        with col3:
            if 'ReturnYTD' in st.session_state.dataframe.columns:
                avg_return = st.session_state.dataframe['ReturnYTD'].mean()
                st.metric("Rata-rata Return YTD", f"{avg_return:.2f}%")
            else:
                st.metric("Rata-rata Return YTD", "N/A")

    st.subheader("Atau, mulai dengan salah satu analisis ini:")

# Fungsi untuk menangani klik tombol
def _extract_agent_answer(response):
    """Ekstrak teks jawaban dari response agent dengan beberapa fallback."""
    try:
        if response is None:
            return "Maaf, tidak ada respons dari agent."
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("output", "output_text", "content", "result"):
                if key in response:
                    return response.get(key)
            return str(response)
        # Generic object fallback
        return getattr(response, 'content', str(response))
    except Exception:
        return str(response)


def handle_button_click(prompt_text):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    # Langsung proses prompt setelah tombol diklik
    with st.chat_message("user"):
        st.markdown(prompt_text)
    with st.chat_message("assistant"):
        with st.spinner("🧠 Menganalisis data..."):
            try:
                response = st.session_state.agent.invoke(prompt_text)
                answer = _extract_agent_answer(response)
            except Exception as e:
                answer = f"Terjadi kesalahan saat memanggil agent: {e}"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📈 Ringkasan Eksekutif Portofolio"):
        handle_button_click("Berikan ringkasan eksekutif dari portofolio ini, termasuk total nilai, jumlah aset, dan statistik utama.")

with col2:
    if st.button("🏆 Aset Kinerja Terbaik & Terburuk"):
        handle_button_click("Tunjukkan 5 aset dengan return tertinggi dan 5 aset dengan return terendah.")

with col3:
    if st.button("🔬 Analisis Konsentrasi Risiko"):
        handle_button_click("Analisis konsentrasi risiko berdasarkan sektor. Tampilkan total nilai dan persentase untuk setiap sektor.")

st.divider()

# Area chat: tampilkan riwayat dan input pengguna
st.subheader("Ajukan Pertanyaan Analisis")

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  

# Input dari pengguna
if prompt := st.chat_input("Contoh: 'Berapa total nilai portofolio?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Proses prompt dengan agent
    with st.chat_message("assistant"):
        with st.spinner("🧠 Menganalisis data..."):
            try:
                response = st.session_state.agent.invoke(prompt)
                answer = _extract_agent_answer(response)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_message = f"Terjadi kesalahan saat berkomunikasi dengan agent: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})


