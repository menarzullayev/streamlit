import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error

from config import TRUE_AGE, AGE_BINS, AGE_LABELS, CAMERAS

logger = logging.getLogger('analytics')
logger.setLevel(logging.INFO)


def calculate_age_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    
    df['age_group'] = pd.cut(df[TRUE_AGE], bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)
    
    results = []
    
    for prefix in CAMERAS:
        pred_col = f'{prefix}_age'
        
        for group in AGE_LABELS:
            df_segment = df[(df['age_group'] == group)].dropna(subset=[TRUE_AGE, pred_col])
            
            if not df_segment.empty:
                mae = mean_absolute_error(df_segment[TRUE_AGE], df_segment[pred_col])
                
                # Bias (Tavsiya #11)
                bias = (df_segment[pred_col] - df_segment[TRUE_AGE]).mean()
                
                results.append({
                    'Camera': prefix.upper(),
                    'Age Group': group,
                    'MAE': mae,
                    'Bias (Mean Error)': bias,
                    'Count': len(df_segment)
                })
        
    df_results = pd.DataFrame(results)
    logger.info("Calculated Age Segmentation and Bias Analysis.")
    
    return df_results


def calculate_fps_metrics(df: pd.DataFrame) -> pd.DataFrame:
    
    fps_metrics = []
    
    for prefix in CAMERAS:
        fps_col = f'{prefix}_fps'
        if fps_col in df.columns:
            avg_fps = df[fps_col].mean()
            fps_metrics.append({'Camera': prefix.upper(), 'Average FPS': avg_fps})
            
    df_fps = pd.DataFrame(fps_metrics)
    logger.info("Calculated Average FPS for all cameras.")
    
    return df_fps


def calculate_bias_metrics(df: pd.DataFrame) -> pd.DataFrame:
    
    bias_results = []
    
    for prefix in CAMERAS:
        pred_col = f'{prefix}_age'
        
        df_comp = df.dropna(subset=[TRUE_AGE, pred_col])
        
        if not df_comp.empty:
            bias = (df_comp[pred_col] - df_comp[TRUE_AGE]).mean()
            bias_results.append({'Camera': prefix.upper(), 'Global Bias (Years)': bias})
        
    df_bias = pd.DataFrame(bias_results)
    logger.info("Calculated Global Age Bias Metrics.")
    
    return df_bias
