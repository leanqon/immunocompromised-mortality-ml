import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer


def load_and_filter_data(cohort_path, dynamic_path):
    """
    Load cohort and dynamic data, and perform initial filtering.
    
    Parameters:
    -----------
    cohort_path : str
        Path to the cohort data CSV file
    dynamic_path : str
        Path to the dynamic data Parquet file
        
    Returns:
    --------
    tuple
        (cohort_data, filtered_dynamic_data)
    """
    # Load cohort data
    cohort_data = pd.read_csv(cohort_path)
    
    # Drop unnecessary columns
    cohort_data = cohort_data.drop(columns=[
        'subject_id', 'hadm_id', 'admittime', 'dischtime', 'intime', 'outtime'
    ])
    
    # Load dynamic data
    dynamic_data = pd.read_parquet(dynamic_path)
    
    # Filter dynamic data based on stay_id in cohort data
    filtered_dynamic_data = dynamic_data[dynamic_data['stay_id'].isin(cohort_data['stay_id'])].copy()
    
    # Convert 'time' column to timedelta for filtering
    filtered_dynamic_data['time'] = pd.to_timedelta(filtered_dynamic_data['time'])
    
    # Filter data to include only the first 24 hours
    first_day_data = filtered_dynamic_data[filtered_dynamic_data['time'] < pd.Timedelta(days=1)]
    
    return cohort_data, first_day_data


def process_measurements(first_day_data, cohort_data=None):
    """
    Process measurements from the first 24 hours of ICU stay.
    Calculate min, max, mean values for vital signs and lab tests,
    and sum for cumulative variables.
    
    Parameters:
    -----------
    first_day_data : DataFrame
        Dynamic data filtered for the first 24 hours
    cohort_data : DataFrame, optional
        Cohort data for reference
        
    Returns:
    --------
    DataFrame
        Processed measurements with aggregated values
    """
    # Define variable categories
    vital_signs = ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map', 'o2sat']
    lab_tests = [
        'alb', 'alp', 'alt', 'ast', 'be', 'bicar', 'bili', 'bili_dir', 'bnd', 
        'bun', 'ca', 'cai', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'fgn', 'fio2', 
        'glu', 'hgb', 'inr_pt', 'k', 'lact', 'lymph', 'mch', 'mchc', 'mcv', 
        'methb', 'mg', 'na', 'neut', 'pco2', 'ph', 'phos', 'plt', 'po2', 'ptt', 'wbc'
    ]
    cumulative_vars = ['urine']
    other_vars = [col for col in first_day_data.columns 
                 if col not in vital_signs + lab_tests + cumulative_vars + ['time', 'stay_id']]
    
    # Initialize dictionaries to store processed data
    max_values = {}
    min_values = {}
    mean_values = {}
    sum_values = {}
    
    # Group data by stay_id
    grouped_data = first_day_data.groupby('stay_id')
    
    # Process each variable
    for var in vital_signs + lab_tests + other_vars:
        if var in first_day_data.columns:
            # Calculate max, min, mean values
            max_values[f"{var}_max"] = grouped_data[var].max()
            min_values[f"{var}_min"] = grouped_data[var].min()
            mean_values[f"{var}_mean"] = grouped_data[var].mean()
    
    # Process cumulative variables
    for var in cumulative_vars:
        if var in first_day_data.columns:
            sum_values[f"{var}_sum"] = grouped_data[var].sum()
    
    # Merge all processed values
    processed_data_parts = [pd.DataFrame(d) for d in [max_values, min_values, mean_values, sum_values] if d]
    
    if processed_data_parts:
        processed_measurements = pd.concat(processed_data_parts, axis=1)
        processed_measurements = processed_measurements.reset_index().rename(columns={'index': 'stay_id'})
        return processed_measurements
    else:
        print("Warning: No valid processed data found!")
        return pd.DataFrame()


def apply_ethnicity_mapping(data):
    """
    Apply standardized ethnicity mapping to the dataset.
    
    Parameters:
    -----------
    data : DataFrame
        Dataset containing ethnicity information
        
    Returns:
    --------
    DataFrame
        Dataset with mapped ethnicity values
    """
    ethnicity_map = {
        'ASIAN - CHINESE': 'Asian',
        'WHITE': 'White',
        'BLACK/AFRICAN AMERICAN': 'African-American',
        'ASIAN - SOUTH EAST ASIAN': 'Asian',
        'WHITE - OTHER EUROPEAN': 'White',
        'ASIAN': 'Asian',
        'UNKNOWN': 'Other',
        'UNABLE TO OBTAIN': 'Other',
        'PATIENT DECLINED TO ANSWER': 'Other',
        'PORTUGUESE': 'Other',
        'ASIAN - ASIAN INDIAN': 'Asian',
        'WHITE - RUSSIAN': 'White',
        'OTHER': 'Other',
        'BLACK/AFRICAN': 'African-American',
        'HISPANIC/LATINO - DOMINICAN': 'Hispanic-American',
        'BLACK/CAPE VERDEAN': 'African-American',
        'BLACK/CARIBBEAN ISLAND': 'African-American',
        'HISPANIC OR LATINO': 'Hispanic-American',
        'ASIAN - KOREAN': 'Asian',
        'AMERICAN INDIAN/ALASKA NATIVE': 'Other',
        'HISPANIC/LATINO - SALVADORAN': 'Hispanic-American',
        'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic-American',
        'WHITE - BRAZILIAN': 'White',
        'HISPANIC/LATINO - MEXICAN': 'Hispanic-American',
        'WHITE - EASTERN EUROPEAN': 'White',
        'HISPANIC/LATINO - GUATEMALAN': 'Hispanic-American',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Other',
        'SOUTH AMERICAN': 'Hispanic-American',
        'HISPANIC/LATINO - CUBAN': 'Hispanic-American',
        'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic-American',
        'HISPANIC/LATINO - COLUMBIAN': 'Hispanic-American',
        'HISPANIC/LATINO - HONDURAN': 'Hispanic-American',
        'MULTIPLE RACE/ETHNICITY': 'Other'
    }
    
    data['ethnicity'] = data['ethnicity'].map(ethnicity_map)
    return data


def apply_ventilation_mapping(data):
    """
    Apply standardized ventilation status mapping to the dataset.
    
    Parameters:
    -----------
    data : DataFrame
        Dataset containing ventilation_status information
        
    Returns:
    --------
    DataFrame
        Dataset with mapped ventilation_status values
    """
    ventilation_map = {
        'SupplementalOxygen': 'No',
        np.nan: 'No',
        'HFNC': 'No',
        'InvasiveVent': 'Yes',
        'NonInvasiveVent': 'Yes',
        'Tracheostomy': 'Yes'
    }
    
    data['ventilation_status'] = data['ventilation_status'].map(ventilation_map)
    return data


def prepare_model_dataset(cohort_data, processed_measurements):
    """
    Prepare and clean the dataset for modeling by merging cohort data
    with processed measurements and applying mappings.
    
    Parameters:
    -----------
    cohort_data : DataFrame
        Cohort data with patient demographics and outcomes
    processed_measurements : DataFrame
        Processed measurements from dynamic data
        
    Returns:
    --------
    tuple
        (merged_data, columns_to_exclude)
    """
    # Add survival time columns for Cox analysis
    cohort_data['survival_time'] = cohort_data['survival_days'].fillna(28)
    cohort_data['survival_time'] = cohort_data['survival_time'].apply(lambda x: min(x, 28))
    cohort_data['event'] = cohort_data['death_within_28_days']
    
    # Merge cohort data with processed measurements
    merged_data = pd.merge(cohort_data, processed_measurements, on='stay_id')
    
    # Apply ethnicity and ventilation mappings
    merged_data = apply_ethnicity_mapping(merged_data)
    merged_data = apply_ventilation_mapping(merged_data)
    
    # Analyze missing values
    missing_values = merged_data.isnull().sum()
    missing_pct = (merged_data.isnull().sum() / len(merged_data) * 100)
    
    # Create missing values DataFrame for reporting
    missing_df = pd.DataFrame({
        'Missing Values (Count)': missing_values,
        'Missing Values (%)': missing_pct
    })
    
    # Identify columns with too many missing values (>20%)
    columns_to_exclude = missing_df[missing_df['Missing Values (%)'] > 20].index.tolist()
    
    # Add additional columns to exclude that are not needed for modeling
    exclude_additionally = ['stay_id', 'dod', 'survival_days']
    columns_to_exclude.extend([col for col in exclude_additionally if col in merged_data.columns])
    
    return merged_data, columns_to_exclude


def handle_missing_values(merged_data, columns_to_exclude, random_state=42):
    """
    Handle missing values and prepare data for modeling.
    
    Parameters:
    -----------
    merged_data : DataFrame
        Merged dataset with cohort and measurement data
    columns_to_exclude : list
        Columns to exclude from the modeling dataset
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        X_development, X_test, y_development, y_test, development_data, test_data
        (Note: development data will be further split into training and validation sets)
    """
    # Create modeling dataset by dropping excluded columns
    X = merged_data.drop(columns=['death_within_28_days', 'event', 'survival_time'] + columns_to_exclude)
    y = merged_data['death_within_28_days']
    
    # Split data into development (training+validation) and testing sets
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Identify continuous and categorical variables
    excluded_columns = ['stay_id', 'los']
    categorical_vars = [
        'gender', 'ethnicity', 'myocardial_infarct', 'congestive_heart_failure',
        'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia', 
        'chronic_pulmonary_disease', 'rheumatic_disease', 'peptic_ulcer_disease', 
        'mild_liver_disease', 'diabetes_without_cc', 'diabetes_with_cc', 'paraplegia',
        'renal_disease', 'malignant_cancer', 'severe_liver_disease', 'metastatic_solid_tumor',
        'aids', 'ventilation_status'
    ]
    categorical_vars = [col for col in categorical_vars if col in X_development.columns]
    continuous_vars = [col for col in X_development.columns if col not in categorical_vars + excluded_columns]
    
    # Impute missing values - KNN for continuous variables
    imputer_numeric = KNNImputer(n_neighbors=5)
    X_development_numeric = X_development[continuous_vars].copy()
    X_development_numeric_imputed = pd.DataFrame(
        imputer_numeric.fit_transform(X_development_numeric), 
        columns=continuous_vars,
        index=X_development.index
    )
    
    X_test_numeric = X_test[continuous_vars].copy()
    X_test_numeric_imputed = pd.DataFrame(
        imputer_numeric.transform(X_test_numeric),
        columns=continuous_vars,
        index=X_test.index
    )
    
    # Impute missing values - Most frequent for categorical variables
    if categorical_vars:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        X_development_categorical = X_development[categorical_vars].copy()
        X_development_categorical_imputed = pd.DataFrame(
            imputer_categorical.fit_transform(X_development_categorical),
            columns=categorical_vars,
            index=X_development.index
        )
        
        X_test_categorical = X_test[categorical_vars].copy()
        X_test_categorical_imputed = pd.DataFrame(
            imputer_categorical.transform(X_test_categorical),
            columns=categorical_vars,
            index=X_test.index
        )
    else:
        X_development_categorical_imputed = pd.DataFrame(index=X_development.index)
        X_test_categorical_imputed = pd.DataFrame(index=X_test.index)
    
    # Get non-numeric columns
    X_development_non_numeric = X_development[excluded_columns].copy() if excluded_columns else pd.DataFrame(index=X_development.index)
    X_test_non_numeric = X_test[excluded_columns].copy() if excluded_columns else pd.DataFrame(index=X_test.index)
    
    # Combine imputed datasets
    X_development_imputed = pd.concat([X_development_non_numeric, X_development_numeric_imputed, X_development_categorical_imputed], axis=1)
    X_test_imputed = pd.concat([X_test_non_numeric, X_test_numeric_imputed, X_test_categorical_imputed], axis=1)
    
    # Create complete development and test datasets with target variables
    development_data = pd.concat([X_development_imputed, pd.Series(y_development, name='death_within_28_days')], axis=1)
    test_data = pd.concat([X_test_imputed, pd.Series(y_test, name='death_within_28_days')], axis=1)
    
    # Add survival time and event information for Cox analysis
    development_indices = X_development.index
    test_indices = X_test.index
    
    development_data['survival_time'] = merged_data.loc[development_indices, 'survival_time'].values
    development_data['event'] = merged_data.loc[development_indices, 'event'].values
    
    test_data['survival_time'] = merged_data.loc[test_indices, 'survival_time'].values
    test_data['event'] = merged_data.loc[test_indices, 'event'].values
    
    return X_development_imputed, X_test_imputed, y_development, y_test, development_data, test_data