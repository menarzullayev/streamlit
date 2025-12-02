import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix
import logging

from config import TRUE_AGE, TRUE_GENDER, CAMERAS

logger = logging.getLogger('metrics_calculator')
logger.setLevel(logging.INFO)


def calculate_mae_age(df: pd.DataFrame) -> dict:
    results = {}
    
    age_cols = {
        'v201': 'v201_age',
        'hd': 'hd_age',
        'fhd': 'fhd_age'
    }
    
    for prefix, pred_col in age_cols.items():
        true_col = TRUE_AGE
        comparison_df = df.dropna(subset=[true_col, pred_col]).copy()
        
        if not comparison_df.empty:
            mae = mean_absolute_error(comparison_df[true_col], comparison_df[pred_col])
            results[f'{prefix}_mae_age'] = mae
            logger.info(f"Calculated MAE for {prefix} age: {mae:.2f}")
        else:
            results[f'{prefix}_mae_age'] = np.nan
            logger.warning(f"Not enough data to calculate MAE for {prefix} age.")
            
    return results


def calculate_classification_metrics(df: pd.DataFrame, true_col: str, pred_col: str, positive_label=1) -> dict:
    
    comparison_df = df.dropna(subset=[true_col, pred_col]).copy()
    
    if comparison_df.empty:
        return {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan, 'accuracy': np.nan}
        
    y_true = comparison_df[true_col].astype(int)
    y_pred = comparison_df[pred_col].astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'total_samples': len(y_true)
    }


def aggregate_all_metrics(df: pd.DataFrame) -> dict:
    
    all_metrics = {}
    
    all_metrics.update(calculate_mae_age(df))
    
    metric_pairs = [
        (TRUE_GENDER, '_gender', 'gender'),
        ('true_count', '_count', 'count'),
        ('worker_gt', f'_worker_detected', 'worker'), 
    ]
    
    for true_col, pred_suffix, metric_type in metric_pairs:
        
        for prefix in CAMERAS:
            pred_col = f'{prefix}{pred_suffix}'
            
            metrics = calculate_classification_metrics(df, true_col, pred_col, positive_label=1)
                
            if metric_type == 'gender':
                
                male_metrics = metrics
                for key, val in male_metrics.items():
                    all_metrics[f'{prefix}_{metric_type}_male_{key}'] = val
                
                tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']
                tp_f, fp_f, fn_f, tn_f = tn, fn, fp, tp
                
                precision_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
                recall_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
                f1_score_f = 2 * (precision_f * recall_f) / (precision_f + recall_f) if (precision_f + recall_f) > 0 else 0
                
                female_metrics = {
                    'tp': tp_f, 'tn': tn_f, 'fp': fp_f, 'fn': fn_f, 
                    'precision': precision_f, 'recall': recall_f, 'f1_score': f1_score_f, 
                    'accuracy': metrics['accuracy']
                }

                for key, val in female_metrics.items():
                    all_metrics[f'{prefix}_{metric_type}_female_{key}'] = val
                
            else:
                for key, val in metrics.items():
                    all_metrics[f'{prefix}_{metric_type}_{key}'] = val
                         
            logger.info(f"Calculated classification metrics for {prefix} {metric_type}.")

    all_metrics['total_records'] = len(df)
    
    return all_metrics
