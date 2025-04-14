import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def get_continuous_columns(X):
    """Get names of continuous variable columns"""
    return X.select_dtypes(include=['float64', 'int64']).columns.tolist()

def get_categorical_columns(X):
    """Get names of categorical variable columns"""
    return X.select_dtypes(include=['object', 'bool']).columns.tolist()

def decision_curve_analysis(y_true, y_pred_proba, threshold_range=None):
    """
    Calculate net benefit for decision curve analysis.
    
    Parameters:
    -----------
    y_true : array-like
        The true binary labels.
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
        
    Returns:
    --------
    tuple
        (thresholds, net_benefit_model, net_benefit_all, net_benefit_none)
    """
    if threshold_range is None:
        threshold_range = np.arange(0.01, 0.99, 0.01)
    
    prevalence = np.mean(y_true)
    
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    for threshold in threshold_range:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        n = len(y_true)
        
        if (tp + fp) > 0:
            nb_model = (tp/n) - (fp/n) * (threshold/(1-threshold))
        else:
            nb_model = 0
        
        nb_all = prevalence - (1-prevalence) * (threshold/(1-threshold))
        nb_none = 0
        
        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)
        net_benefit_none.append(nb_none)
    
    return threshold_range, net_benefit_model, net_benefit_all, net_benefit_none

def bootstrap_ci(y_true, y_pred_proba, metric_func, n_bootstraps=1000, alpha=0.05, threshold=0.5):
    """
    Calculate bootstrap confidence intervals for performance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        The true binary labels.
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
   
    Returns:
    --------
    tuple
        (base_score, lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    stats = []
    
    # Generate binary predictions if needed
    if metric_func.__name__ in ['precision_score', 'recall_score', 'f1_score', 'accuracy_score']:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate the original metric
    if metric_func.__name__ in ['roc_auc_score', 'average_precision_score']:
        base_score = metric_func(y_true, y_pred_proba)
    else:
        base_score = metric_func(y_true, y_pred)
    
    # Bootstrap sampling
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        
        # Calculate metric
        if metric_func.__name__ in ['roc_auc_score', 'average_precision_score']:
            score = metric_func(y_true_boot, y_pred_proba_boot)
        else:
            y_pred_boot = (y_pred_proba_boot >= threshold).astype(int)
            score = metric_func(y_true_boot, y_pred_boot)
        
        stats.append(score)
    
    # Calculate confidence interval
    lower_bound = np.percentile(stats, alpha/2 * 100)
    upper_bound = np.percentile(stats, (1 - alpha/2) * 100)
    
    return base_score, lower_bound, upper_bound

def create_missing_data_summary(data, output_path=None):
    """
    Create a summary of missing data.
    
    Parameters:
    -----------
    data : DataFrame
        Dataset to analyze for missing values
    output_path : str, optional
        Path to save the missing data summary
        
    Returns:
    --------
    DataFrame
        Summary of missing data
    """
    missing_count = data.isnull().sum()
    missing_percent = data.isnull().mean() * 100
    
    missing_data = pd.DataFrame({
        'Variable': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing Percentage (%)': missing_percent.values.round(2)
    })
    
    missing_data['Imputation Method'] = ['Nearest Neighbor' if val < 20 else 'Excluded' 
                                        for val in missing_data['Missing Percentage (%)']]
    
    # Sort by missing percentage in descending order
    missing_data = missing_data.sort_values('Missing Percentage (%)', ascending=False)
    
    if output_path:
        missing_data.to_csv(output_path, index=False)
    
    return missing_data