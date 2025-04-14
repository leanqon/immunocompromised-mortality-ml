import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lifelines import CoxPHFitter


def prepare_cox_data(data):
    """
    Prepare data for Cox proportional hazards regression by encoding categorical variables
    and standardizing continuous variables.
    
    Parameters:
    -----------
    data : DataFrame
        Dataset with survival time and event information
        
    Returns:
    --------
    tuple
        (cox_data, final_features)
    """
    cox_data = data.copy()
    
    # Define different types of features
    binary_features = ['ventilation_status']
    features_1 = [
        'myocardial_infarct', 'congestive_heart_failure', 'cerebrovascular_disease',
        'chronic_pulmonary_disease', 'mild_liver_disease', 'paraplegia',
        'renal_disease', 'severe_liver_disease', 'metastatic_solid_tumor', 'aids'
    ]
    
    continuous_features = [
        'admission_age', 'charlson_comorbidity_index', 'gcs', 'sofa_score', 
        'weight', 'map_max', 'o2sat_max', 'glu_max', 'k_max', 'mcv_max', 
        'phos_max', 'resp_min', 'temp_min', 'sbp_min', 'map_min', 'o2sat_min', 
        'bicar_min', 'bun_min', 'ca_min', 'cl_min', 'hgb_min', 'mchc_min', 
        'na_min', 'plt_min', 'wbc_min', 'hr_mean', 'resp_mean', 'temp_mean', 
        'o2sat_mean', 'inr_pt_mean', 'mg_mean', 'urine_sum'
    ]
    
    # Keep only available features
    continuous_features = [f for f in continuous_features if f in cox_data.columns]
    binary_features = [f for f in binary_features if f in cox_data.columns]
    features_1 = [f for f in features_1 if f in cox_data.columns]
    
    # Convert binary variables to numeric
    for col in binary_features:
        if cox_data[col].dtype == 'object':
            cox_data[col] = (cox_data[col] == 'Yes').astype(int)
    
    # Standardize continuous variables
    scaler = StandardScaler()
    cox_data[continuous_features] = scaler.fit_transform(cox_data[continuous_features])
    
    # One-hot encode ethnicity
    if 'ethnicity' in cox_data.columns:
        encoder = OneHotEncoder(sparse=False, drop='first')
        ethnicity_encoded = encoder.fit_transform(cox_data[['ethnicity']])
        ethnicity_cols = [f"ethnicity_{cat}" for cat in encoder.categories_[0][1:]]
        
        for i, col in enumerate(ethnicity_cols):
            cox_data[col] = ethnicity_encoded[:, i]
        
        # Combine all features
        final_features = continuous_features + binary_features + ethnicity_cols + features_1
    else:
        final_features = continuous_features + binary_features + features_1
    
    return cox_data, final_features


def run_univariate_cox(train_data, features):
    """
    Perform univariate Cox regression analysis for each feature.
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data with survival time and event information
    features : list
        List of features to analyze
        
    Returns:
    --------
    DataFrame
        Summary of univariate Cox regression results
    """
    univariate_results = []
    for var in features:
        df_uni = train_data[[var, 'survival_time', 'event']].copy()
        cph_uni = CoxPHFitter()
        cph_uni.fit(df_uni, duration_col='survival_time', event_col='event', show_progress=False)
        uni_row = cph_uni.summary.copy()
        uni_row.index = [var]
        univariate_results.append(uni_row)
    
    uni_summary_df = pd.concat(univariate_results)
    uni_summary_df['sig'] = uni_summary_df['p'].apply(lambda x: '*' if x < 0.05 else '')
    return uni_summary_df


def run_cox_analysis(train_data, test_data, output_dir):
    """
    Perform both univariate and multivariate Cox regression analysis.
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data with survival time and event information
    test_data : DataFrame
        Test data with survival time and event information
    output_dir : str
        Directory to save results and figures
        
    Returns:
    --------
    tuple
        (model, univariate_results, multivariate_results, c_index)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for Cox regression
    train_cox, features = prepare_cox_data(train_data)
    test_cox, _ = prepare_cox_data(test_data)
    
    # Univariate analysis
    uni_summary_df = run_univariate_cox(train_cox, features)
    formatted_uni = uni_summary_df.to_string(float_format=lambda x: f"{x:.3f}")
    print("Univariate Cox Analysis Results:")
    print(formatted_uni)
    
    # Save univariate results
    uni_csv_path = os.path.join(output_dir, 'cox_results_univariate.csv')
    uni_summary_df.to_csv(uni_csv_path, float_format="%.3f", index=True)
    
    # Select features with p < 0.05 for multivariate analysis
    selected_features = uni_summary_df[uni_summary_df['p'] < 0.05].index.tolist()
    print(f"\nSelected features for multivariate analysis (p < 0.05): {len(selected_features)}")
    
    # Multivariate analysis
    cph = CoxPHFitter(penalizer=1)
    train_cox_final = train_cox[selected_features + ['survival_time', 'event']]
    cph.fit(train_cox_final, duration_col='survival_time', event_col='event', show_progress=True)
    
    multi_summary_df = cph.summary.copy()
    multi_summary_df['sig'] = multi_summary_df['p'].apply(lambda x: '*' if x < 0.05 else '')
    formatted_multi = multi_summary_df.to_string(float_format=lambda x: f"{x:.3f}")
    print("\nMultivariate Cox Analysis Results:")
    print(formatted_multi)
    
    # Save multivariate results
    multi_csv_path = os.path.join(output_dir, 'cox_results_multivariate.csv')
    multi_summary_df.to_csv(multi_csv_path, float_format="%.3f", index=True)
    
    # Evaluate model on test set
    test_cox_final = test_cox[selected_features + ['survival_time', 'event']]
    c_index = cph.score(test_cox_final, scoring_method="concordance_index")
    print(f"\nC-index on test set: {c_index:.3f}")
    
    # Create forest plot with significance markers
    plt.figure(figsize=(12, 16))
    ax = cph.plot(hazard_ratios=True)
    tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    sig_marker = cph.summary['p'].apply(lambda x: '*' if x < 0.05 else '').to_dict()
    new_labels = [f"{label}{sig_marker.get(label, '')}" for label in tick_labels]
    ax.set_yticklabels(new_labels)
    plt.title('Hazard Ratios of Predictors (Multivariate)')
    plt.tight_layout()
    forest_plot_path = os.path.join(output_dir, 'cox_forest_plot_multivariate.png')
    plt.savefig(forest_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create baseline survival curve
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_survival = cph.baseline_survival_
    ax.plot(baseline_survival.index, baseline_survival.values, label='Baseline Survival', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')
    plt.title('Baseline Survival Curve (Multivariate)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    survival_curve_path = os.path.join(output_dir, 'cox_survival_curve_multivariate.png')
    plt.savefig(survival_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined results table (univariate and multivariate)
    create_combined_results_table(uni_summary_df, multi_summary_df, output_dir)
    
    return cph, formatted_uni, formatted_multi, c_index


def create_combined_results_table(uni_df, multi_df, output_dir):
    """
    Create a combined table with both univariate and multivariate Cox regression results.
    
    Parameters:
    -----------
    uni_df : DataFrame
        Univariate Cox regression results
    multi_df : DataFrame
        Multivariate Cox regression results
    output_dir : str
        Directory to save the combined results
    """
    # Generate univariate HR with CI and p-value columns
    uni_df["Uni_HR_95CI"] = uni_df.apply(
        lambda x: f"{x['exp(coef)']:.3f} ({x['exp(coef) lower 95%']:.3f}-{x['exp(coef) upper 95%']:.3f})",
        axis=1
    )
    uni_df["Uni_p"] = uni_df["p"].apply(lambda x: f"{x:.3g}")
    
    # Generate multivariate HR with CI and p-value columns
    multi_df["Multi_HR_95CI"] = multi_df.apply(
        lambda x: f"{x['exp(coef)']:.3f} ({x['exp(coef) lower 95%']:.3f}-{x['exp(coef) upper 95%']:.3f})",
        axis=1
    )
    multi_df["Multi_p"] = multi_df["p"].apply(lambda x: f"{x:.3g}")
    
    # Merge the results
    final_df = pd.merge(
        uni_df[["Uni_HR_95CI", "Uni_p"]],
        multi_df[["Multi_HR_95CI", "Multi_p"]],
        left_index=True,
        right_index=True,
        how="outer"
    )
    
    # Organize the results
    final_df.index.name = "Variables"
    final_df = final_df[["Uni_HR_95CI", "Uni_p", "Multi_HR_95CI", "Multi_p"]]
    
    # Save the combined results
    combined_path = os.path.join(output_dir, 'cox_results_combined.csv')
    final_df.to_csv(combined_path)
    
    return final_df