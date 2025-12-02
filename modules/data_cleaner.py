import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import TRUE_AGE, TRUE_GENDER, TRUE_FILENAME, CAMERAS

logger = logging.getLogger('data_cleaner')
logger.setLevel(logging.INFO)

# Old names derived from CSV, used for renaming
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    rename_map = {
        'v.2.01 - age': 'v201_age', 'v.2.01 - gender': 'v201_gender', 'FPS': 'v201_fps',
        'v.2.01 - time': 'v201_time', 'v.2.01-ID': 'v201_id',
        
        'HD-age': 'hd_age', 'HD-gender': 'hd_gender', 'FPS.1': 'hd_fps',
        'HD': 'hd_time', 'HD-ID': 'hd_id',
        
        'FHD-age': 'fhd_age', 'FHD-gender': 'fhd_gender', 'FPS.2': 'fhd_fps',
        'FHD': 'fhd_time', 'FHD-ID': 'fhd_id',
        
        'Mansurbek': 'v201_score_mansb', 'Saidakbar': 'v201_score_saidak',
        'Mansurbek.1': 'hd_score_mansb', 'Saidakbar.1': 'hd_score_saidak',
        'Mansurbek.2': 'fhd_score_mansb', 'Saidakbar.2': 'fhd_score_saidak',
    }
    
    valid_rename_map = {old: new for old, new in rename_map.items() if old in df.columns}
    df = df.rename(columns=valid_rename_map)
    logger.info(f"Renamed {len(valid_rename_map)} columns to standardized format.")
    
    return df


def clean_fps_and_worker_scores(df: pd.DataFrame) -> pd.DataFrame:
    
    # Clean FPS
    fps_cols = ['v201_fps', 'hd_fps', 'fhd_fps']
    for col in fps_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean Worker Scores
    score_cols = ['v201_score_mansb', 'v201_score_saidak', 'hd_score_mansb', 'hd_score_saidak', 'fhd_score_mansb', 'fhd_score_saidak']
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info("Cleaned FPS and Worker Score columns to float.")
    return df


def normalize_gender_worker(df: pd.DataFrame) -> pd.DataFrame:
    
    df[TRUE_GENDER] = df[TRUE_GENDER].replace({'Man': 1, 'Woman': 0})
    if TRUE_GENDER in df.columns:
        df[TRUE_GENDER] = pd.to_numeric(df[TRUE_GENDER], errors='coerce').fillna(0).astype(int)
    
    for col in ['v201_gender', 'hd_gender', 'fhd_gender']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    for prefix in CAMERAS:
        score_cols = [f'{prefix}_score_mansb', f'{prefix}_score_saidak']
        
        df[f'{prefix}_worker_detected'] = df[score_cols].max(axis=1).apply(lambda x: 1 if x > 0 else 0).astype(int)
        
    df['worker_gt'] = df[TRUE_FILENAME].apply(lambda x: 1 if x in ['Mansurbek', 'Saidakbar'] else 0)
    
    logger.info("Normalized gender and created binary worker detection flags.")
    return df


def create_detection_flags(df: pd.DataFrame) -> pd.DataFrame:
    
    df['v201_count'] = df['v201_id'].notna().astype(int)
    df['hd_count'] = df['hd_id'].notna().astype(int)
    df['fhd_count'] = df['fhd_id'].notna().astype(int)
    
    df[TRUE_AGE] = pd.to_numeric(df[TRUE_AGE], errors='coerce')
    df['true_count'] = df[TRUE_AGE].notna().astype(int)
    
    logger.info("Created binary detection flags (count metrics) for all cameras.")
    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """Tekshirish funksiyasi (Tavsiya #17)."""
    required_cols = [TRUE_AGE, TRUE_GENDER, TRUE_FILENAME, 'v201_id', 'hd_id', 'fhd_id', 'v201_age', 'hd_age', 'fhd_age']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"CRITICAL ERROR: The uploaded file is missing required columns after renaming: {missing_cols}")


def load_and_clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    try:
        df = rename_columns(df_raw.copy())
        
        # Check for required columns after renaming (Tavsiya #17)
        check_required_columns(df)
        
        df = clean_fps_and_worker_scores(df)
        
        df = normalize_gender_worker(df)
        
        df = create_detection_flags(df)
        
        logger.info(f"Data cleaning successful. Final shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}")
        raise