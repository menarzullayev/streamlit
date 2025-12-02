import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import COLOR_MAP_MAIN, CAMERAS, TRUE_FILENAME, TRUE_AGE, AGE_LABELS


def plot_metrics_table(df_metrics: pd.DataFrame, title: str):
    st.subheader(title)
    st.dataframe(df_metrics.style.format('{:.3f}'))


def plot_bar_chart(df_data: pd.DataFrame, y_col: str, title: str, color_col='Camera'):
    df_plot = df_data.copy().reset_index().rename(columns={'index': 'Camera', y_col: 'Value'})
    
    fig = px.bar(
        df_plot,
        x='Camera',
        y='Value',
        color=color_col,
        title=title,
        template="plotly_white",
        height=400,
        text_auto='.2f'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix(metrics: dict, camera: str, metric_type: str):
    tp = metrics.get(f'{camera}_{metric_type}_tp', 0)
    tn = metrics.get(f'{camera}_{metric_type}_tn', 0)
    fp = metrics.get(f'{camera}_{metric_type}_fp', 0)
    fn = metrics.get(f'{camera}_{metric_type}_fn', 0)
    
    z = [[tn, fp], [fn, tp]]
    x = ['Predicted Non-Positive (0)', 'Predicted Positive (1)']
    y = ['True Non-Positive (0)', 'True Positive (1)']

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix: {camera.upper()} ({metric_type.title()})',
        xaxis_title="Prediction",
        yaxis_title="Ground Truth",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_age_segment_bias(df_segment: pd.DataFrame, chart_type='MAE'):
    
    df_plot = df_segment.rename(columns={'Age Group': 'Group', chart_type: 'Value'})
    
    if chart_type == 'Bias (Mean Error)':
        # Xato chizig'ini qo'shish (0 qiymat)
        fig = px.bar(
            df_plot, x='Group', y='Value', color='Camera',
            title=f'{chart_type} by Age Group (Positive=Overestimate)',
            template="plotly_white", barmode='group', color_discrete_map=COLOR_MAP_MAIN, height=400
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
    else:
        fig = px.bar(
            df_plot, x='Group', y='Value', color='Camera',
            title=f'{chart_type} by Age Group',
            template="plotly_white", barmode='group', color_discrete_map=COLOR_MAP_MAIN, height=400
        )
        
    st.plotly_chart(fig, use_container_width=True)


def plot_individual_detection_comparison_chart(df: pd.DataFrame):
    
    st.subheader("Age Prediction vs Ground Truth (Per Record)")

    unique_filenames = sorted(df[df[TRUE_AGE].notna()][TRUE_FILENAME].unique())
    selected_filename = st.selectbox(
        'Choose Person to Display Full Detection History (Pagination)',
        options=unique_filenames,
        index=0 if unique_filenames else None
    )

    if not selected_filename:
        st.info("No persons with recorded Ground Truth Age found for individual analysis.")
        return
    
    df_person = df[df[TRUE_FILENAME] == selected_filename].copy().reset_index(drop=True)
    total_detections = len(df_person)
    gt_age_value = df_person[TRUE_AGE].dropna().iloc[0] if not df_person[TRUE_AGE].dropna().empty else np.nan
    
    st.info(f"Note: The selection box above acts as 'Pagination' to view each person's full history.")
    st.markdown(f"**Selected Person:** {selected_filename} (GT Age: {gt_age_value}), **Total Detections:** {total_detections}")

    detection_ids = [f'D{i+1}' for i in range(total_detections)]
    df_melt_list = []
    
    # 1. GT Age - Yagona ustun
    gt_df = pd.DataFrame({
        'X-Label': ['GT Age'],
        'Source': ['True Age (GT)'],
        'Age': [gt_age_value],
    })
    df_melt_list.append(gt_df)

    # 2. Kameralar Prognozlari (barcha deteksiyalarni qo'shish)
    for prefix in CAMERAS:
        df_cam = df_person[[f'{prefix}_age']].copy()
        df_cam.columns = ['Age']
        df_cam['Detection ID'] = detection_ids
        df_cam['Source'] = prefix.upper()
        
        # X-Labelni to'g'ri tartiblash uchun alohida ustun yaratish
        df_cam['X-Label'] = [f"{prefix.upper()} {id_}" for id_ in detection_ids]
        df_melt_list.append(df_cam)

    df_final_plot = pd.concat(df_melt_list, ignore_index=True)
    
    # Tartibni majburlash (GT, V201 D1..DN, HD D1..DN, FHD D1..DN)
    custom_order = ['GT Age']
    for prefix in CAMERAS:
        for i in range(total_detections):
            custom_order.append(f"{prefix.upper()} D{i+1}")

    fig_age = px.bar(
        df_final_plot,
        x='X-Label',
        y='Age',
        color='Source',
        title=f"Age Prognozi: {selected_filename} (Barcha {total_detections} Deteksiyalar)",
        template="plotly_white",
        barmode='group', # Grouped barmode faqat X-o'qidagi ustunlarni guruhlashda muhim, bu yerda Source'ga rang beriladi
        height=500,
        color_discrete_map=COLOR_MAP_MAIN,
        text='Age',
        category_orders={"X-Label": custom_order}
    )
    
    fig_age.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_age.update_xaxes(tickangle=90) 
    st.plotly_chart(fig_age, use_container_width=True)


def plot_worker_score_comparison(df_person: pd.DataFrame, selected_filename: str):
    
    worker_score_cols = [
        'v201_score_mansb', 'v201_score_saidak', 
        'hd_score_mansb', 'hd_score_saidak', 
        'fhd_score_mansb', 'fhd_score_saidak'
    ]
    
    df_scores_person = df_person.copy().dropna(subset=worker_score_cols, how='all').reset_index(drop=True)
    
    df_scores_melt = df_scores_person.melt(
        id_vars=[TRUE_AGE],
        value_vars=worker_score_cols,
        var_name='Source & Identity',
        value_name='Similarity Score (0-100)'
    ).dropna(subset=['Similarity Score (0-100)'])
    
    df_scores_melt['Camera'] = df_scores_melt['Source & Identity'].apply(lambda x: x.split('_')[0].upper())
    df_scores_melt['Identity'] = df_scores_melt['Source & Identity'].apply(lambda x: x.split('_')[2].title())
    
    df_scores_melt['Detection ID'] = [f'D{i+1}' for i in range(len(df_scores_melt))]

    fig_scores = px.bar(
        df_scores_melt,
        x='Detection ID',
        y='Similarity Score (0-100)',
        color='Camera',
        pattern_shape='Identity',
        title=f"Similarity Scores for {selected_filename} Across All Detections",
        template="plotly_white",
        barmode='group',
        height=550,
        color_discrete_map=COLOR_MAP_MAIN
    )
    fig_scores.update_xaxes(title="Deteksiya ID") 
    st.plotly_chart(fig_scores, use_container_width=True)