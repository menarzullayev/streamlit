import streamlit as st
import pandas as pd
import numpy as np
import logging
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# Import custom files and modules
from config import CAMERAS, TRUE_FILENAME, TRUE_AGE, COLOR_MAP_MAIN
from modules.metrics_calculator import aggregate_all_metrics
from modules.analytics import calculate_age_segmentation, calculate_fps_metrics, calculate_bias_metrics
from modules.plotting import plot_bar_chart, plot_confusion_matrix, plot_metrics_table, plot_age_segment_bias, plot_individual_detection_comparison_chart, plot_worker_score_comparison
from modules.data_cleaner import load_and_clean_data, TRUE_GENDER

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app_main')

# --- 1. CONFIGURATION AND CSS INJECTION ---
st.set_page_config(
    page_title="Camera Performance Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_full_width_css():
    st.markdown(
        """
        <style>
        .main { max-width: 1600px; padding-right: 2rem; padding-left: 2rem; }
        .st-emotion-cache-1g83s45, .st-emotion-cache-1jm692n, .st-emotion-cache-1v060j8 { 
            max-width: 1600px !important; width: 100% !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def analyze_and_get_metrics(df_raw: pd.DataFrame) -> tuple:
    try:
        from modules.data_cleaner import load_and_clean_data 
        df_cleaned = load_and_clean_data(df_raw.copy())
        
        metrics = aggregate_all_metrics(df_cleaned)
        
        df_segment = calculate_age_segmentation(df_cleaned.copy())
        df_fps = calculate_fps_metrics(df_cleaned.copy())
        df_bias_global = calculate_bias_metrics(df_cleaned.copy())

        return metrics, df_cleaned, df_segment, df_fps, df_bias_global
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

@st.cache_data(ttl=600)
def load_from_google_sheet(url):
    try:
        logger.info(f"Attempting to load data from Google Sheet URL: {url}")
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logger.error(f"Failed to load Google Sheet: {e}")
        raise Exception(f"Failed to load Google Sheet: {e}. Check if the GID is correct and the sheet is shared publicly for reading.")


def main():
    inject_full_width_css() 
    st.title("Camera Performance Comparison Dashboard 📸")

    # --- GOOGLE SHEET CONFIGURATION ---
    # GID va SHEET ID uchun so'ralgan input
    DEFAULT_FILE_PATH = 'data/data.xlsx'
    df_raw = None
    
    # Session state'ni boshlash
    if 'data_source' not in st.session_state:
         st.session_state['data_source'] = 'local'

    # --- Sidebar Interaktiv Qismi ---
    st.sidebar.header("1. Data Upload (CSV/XLSX)")

    # 1. LOCAL FILE UPLOAD
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel file (Ensure correct column names)",
        type=['csv', 'xlsx']
    )
    
    # 2. GOOGLE SHEET INPUT (Manual Import)
    st.sidebar.markdown('---')
    st.sidebar.subheader("Manual Google Sheet Load")
    
    # DEFAULT QIymatlar
    DEFAULT_SHEET_ID = "1nEGhe-9xo2xpELhW1zvufvyN2vlEDnuXv9VlMPZmk4U"
    DEFAULT_FHD_GID = "1135908356" # FHD bo'limi uchun berilgan GID

    sheet_id_input = st.sidebar.text_input("Enter Google Sheet ID", DEFAULT_SHEET_ID)
    gid_input = st.sidebar.text_input("Enter GID (Sheet Tab ID)", DEFAULT_FHD_GID)
    
    GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{sheet_id_input}/export?format=csv&gid={gid_input}"

    if st.sidebar.button("Load Data from Google Sheet"):
        st.session_state['data_source'] = 'google'
        # Google Sheet yuklanayotganda keshni tozalash
        analyze_and_get_metrics.clear()
        time.sleep(0.5)
        
    # --- Qaysi manbadan yuklashni aniqlash ---
    
    if st.session_state['data_source'] == 'google':
        try:
            df_raw = load_from_google_sheet(GOOGLE_SHEET_URL)
            st.sidebar.success(f"Data loaded from Google Sheet (GID: {gid_input})")
        except Exception as e:
            st.error(f"Failed to load data from Google Sheet: {e}")
            st.info(f"Please ensure the sheet is publicly shared and the GID is correct.")
            st.session_state['data_source'] = 'local'
            
    elif uploaded_file is None:
        # Avtomatik yuklash mantiği (Local file)
        if os.path.exists(DEFAULT_FILE_PATH):
            try:
                if DEFAULT_FILE_PATH.endswith(('.xlsx', '.xls')):
                    df_raw = pd.read_excel(DEFAULT_FILE_PATH)
                else:
                    df_raw = pd.read_csv(DEFAULT_FILE_PATH)
                    
                st.sidebar.success(f"Default file loaded automatically from {DEFAULT_FILE_PATH}")
            except Exception as e:
                st.error(f"Error loading default file: {e}. Check file type.")
                st.exception(e)
        else:
             st.info(f"Please ensure your 'data/' folder contains the default file ({DEFAULT_FILE_PATH}) or upload a new file.")
        
    elif uploaded_file is not None:
        # User yuklagan fayl
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(uploaded_file)
            
    # --- Data Processing and Metrics Calculation ---
    if df_raw is None:
        st.stop()
        
    analyze_and_get_metrics.clear()
    
    try:
        with st.spinner("Analyzing data and calculating metrics..."):
            metrics, df_cleaned, df_segment, df_fps, df_bias_global = analyze_and_get_metrics(df_raw)
    except Exception as e:
        st.error(f"Analysis failed during processing. Check column names and data types. Error: {e}")
        st.exception(e)
        st.stop()


    if not metrics:
        st.stop()
        
    # --- Dashboard Elements ---
    st.sidebar.header("2. Global Filters")
    
    selected_camera_filter = st.sidebar.multiselect(
        'Filter Data Display by Camera Type',
        options=CAMERAS,
        default=CAMERAS,
        format_func=lambda x: x.upper()
    )
    
    filtered_filenames = df_cleaned[df_cleaned[TRUE_FILENAME].notna()][TRUE_FILENAME].unique()
    st.sidebar.markdown('*(Note: Metrics are static based on full dataset)*')

    # Qolgan Dashboard qismlari (o'zgartirishsiz)
    st.header("1. Summary Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    mae_values = {k: v for k, v in metrics.items() if '_mae_age' in k and not np.isnan(v)}
    best_mae_camera = min(mae_values, key=mae_values.get).split('_')[0].upper() if mae_values else 'N/A'
    
    count_acc_values = {k: v for k, v in metrics.items() if '_count_accuracy' in k and not np.isnan(v)}
    best_count_acc = max(count_acc_values, key=count_acc_values.get).split('_')[0].upper() if count_acc_values else 'N/A'
    
    with col1:
        st.metric(label="Total Records Analyzed", value=metrics.get('total_records', 'N/A'))
    with col2:
        st.metric(label="Best Age MAE Camera", value=best_mae_camera, delta=f"Min MAE: {min(mae_values.values()):.2f}" if mae_values else None)
    with col3:
        st.metric(label="Best Detection Accuracy", value=best_count_acc, delta=f"Acc: {max(count_acc_values.values()):.2f}" if count_acc_values else 'N/A')
    with col4:
        st.metric(label="Total Worker Records (GT)", value=df_cleaned['worker_gt'].sum() if 'worker_gt' in df_cleaned.columns else 'N/A')

    st.markdown("---") 

    st.header("2. Age, FPS and Bias Analysis")
    
    col_mae, col_bias, col_fps = st.columns(3)

    mae_data = pd.DataFrame([
        {'Camera': prefix.upper(), 'Value': metrics.get(f'{prefix}_mae_age', np.nan)}
        for prefix in CAMERAS
    ]).dropna(subset=['Value']).set_index('Camera')
    with col_mae:
        plot_bar_chart(mae_data, 'Value', "Mean Absolute Error (MAE) [Years]")

    df_bias_global = df_bias_global.set_index('Camera').rename(columns={'Global Bias (Years)': 'Bias'}).sort_values(by='Bias')
    with col_bias:
        plot_bar_chart(df_bias_global, 'Bias', "Global Systematic Error (Bias) [Years]")
        
    df_fps = df_fps.set_index('Camera').rename(columns={'Average FPS': 'FPS'}).sort_values(by='FPS', ascending=False)
    with col_fps:
        plot_bar_chart(df_fps, 'FPS', "Average Frame Rate (FPS)")

    st.markdown("---")
    
    st.subheader("2.1 Detailed Age Segmentation")
    if not df_segment.empty:
        col_seg_mae, col_seg_bias = st.columns(2)
        with col_seg_mae:
            plot_age_segment_bias(df_segment, chart_type='MAE')
        with col_seg_bias:
            plot_age_segment_bias(df_segment, chart_type='Bias (Mean Error)')
    
    st.markdown("---")

    st.header("3. Gender Classification Metrics")
    
    df_gender_metrics = pd.DataFrame([
        {'Camera': prefix.upper(), 'Precision': metrics.get(f'{prefix}_gender_male_precision'), 'Recall': metrics.get(f'{prefix}_gender_male_recall'), 'F1 Score': metrics.get(f'{prefix}_gender_male_f1_score'), 'Accuracy': metrics.get(f'{prefix}_gender_male_accuracy')}
        for prefix in CAMERAS
    ]).set_index('Camera')
    
    plot_metrics_table(df_gender_metrics, "Gender Classification: Precision, Recall, F1 Score")
    
    gender_acc_data = pd.DataFrame([
        {'Camera': prefix.upper(), 'Value': metrics.get(f'{prefix}_gender_male_accuracy', np.nan)}
        for prefix in CAMERAS
    ]).dropna(subset=['Value']).set_index('Camera').rename(columns={'Value': 'Accuracy'})
    plot_bar_chart(gender_acc_data, 'Accuracy', "Gender Classification Accuracy (Overall)")
    
    st.markdown("---")
    
    st.header("4. Detection (Count) and Worker Analysis")
    
    col_count, col_worker = st.columns(2)
    
    df_count_metrics = pd.DataFrame([
        {'Camera': prefix.upper(), 'TP': metrics.get(f'{prefix}_count_tp'), 'FP': metrics.get(f'{prefix}_count_fp'), 'FN': metrics.get(f'{prefix}_count_fn'), 'Accuracy': metrics.get(f'{prefix}_count_accuracy')}
        for prefix in CAMERAS
    ]).set_index('Camera')
    with col_count:
        plot_metrics_table(df_count_metrics, "Detection Accuracy (Count) Metrics")
    
    df_worker_metrics = pd.DataFrame([
        {'Camera': prefix.upper(), 'Precision': metrics.get(f'{prefix}_worker_precision'), 'Recall': metrics.get(f'{prefix}_worker_recall'), 'F1 Score': metrics.get(f'{prefix}_worker_f1_score'), 'Accuracy': metrics.get(f'{prefix}_worker_accuracy')}
        for prefix in CAMERAS
    ]).set_index('Camera')
    with col_worker:
        plot_metrics_table(df_worker_metrics, "Worker Recognition Metrics (vs GT Worker)")
        
    st.subheader("Confusion Matrices (Count)")
    col_cm1, col_cm2, col_cm3 = st.columns(3)
    
    with col_cm1:
        plot_confusion_matrix(metrics, 'v201', 'count')
    with col_cm2:
        plot_confusion_matrix(metrics, 'hd', 'count')
    with col_cm3:
        plot_confusion_matrix(metrics, 'fhd', 'count')
        
    st.markdown("---")

    st.header("5. Individual Detection Deep Dive 🔬")
    plot_individual_detection_comparison_chart(df_cleaned)
    
    st.markdown("---")

    st.header("6. Raw Data Table (Filtered)")
    
    names_to_filter = st.multiselect(
        'Filter Data by Filename',
        options=filtered_filenames,
        default=[]
    )
    
    df_display = df_cleaned.copy()
    
    if names_to_filter:
        df_display = df_display[df_display[TRUE_FILENAME].isin(names_to_filter)]
    
    if selected_camera_filter:
        camera_cols_to_hide = [cam for cam in CAMERAS if cam not in selected_camera_filter]
        
        for cam in camera_cols_to_hide:
            cols_to_hide = [f'{cam}_id', f'{cam}_age', f'{cam}_gender', f'{cam}_fps', f'{cam}_score_mansb', f'{cam}_score_saidak', f'{cam}_time', f'{cam}_count']
            
            valid_cols_to_hide = [col for col in cols_to_hide if col in df_display.columns]
            df_display[valid_cols_to_hide] = np.nan
            
    st.dataframe(df_display, width='stretch')
    st.markdown("*(Note: Columns not selected in the sidebar filter may show NaN values for visualization purposes)*")
        

if __name__ == '__main__':
    logging.getLogger('data_cleaner').setLevel(logging.INFO)
    logging.getLogger('metrics_calculator').setLevel(logging.INFO)
    logging.getLogger('analytics').setLevel(logging.INFO)
    
    main()