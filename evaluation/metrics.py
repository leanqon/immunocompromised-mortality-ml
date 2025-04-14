import numpy as np
import pandas as pd
import os
import joblib
import time
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, calibration_curve
)
import sys
sys.path.append('..')
from utils import bootstrap_ci, decision_curve_analysis


def evaluate_model(model_name, model, X_test, y_test, threshold=0.5, n_bootstraps=500):
    """
    Evaluate a single model and calculate performance metrics with confidence intervals
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model : estimator
        Trained model with predict_proba method
    X_test : DataFrame or array
        Test features
    y_test : Series or array
        True labels
        
    Returns:
    --------
    dict
        Evaluation results including metrics and curves
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics with 95% CI
    metrics = {}
    
    # AUC
    auc_mean, auc_lower, auc_upper = bootstrap_ci(
        y_test, y_pred_proba, roc_auc_score, n_bootstraps=n_bootstraps
    )
    metrics['AUC'] = (auc_mean, auc_lower, auc_upper)
    
    # PRC (Average Precision)
    prc_mean, prc_lower, prc_upper = bootstrap_ci(
        y_test, y_pred_proba, average_precision_score, n_bootstraps=n_bootstraps
    )
    metrics['PRC'] = (prc_mean, prc_lower, prc_upper)
    
    # Accuracy
    acc_mean, acc_lower, acc_upper = bootstrap_ci(
        y_test, y_pred_proba, accuracy_score, n_bootstraps=n_bootstraps, threshold=threshold
    )
    metrics['Accuracy'] = (acc_mean, acc_lower, acc_upper)
    
    # Precision (PPV)
    prec_mean, prec_lower, prec_upper = bootstrap_ci(
        y_test, y_pred_proba, precision_score, n_bootstraps=n_bootstraps, threshold=threshold
    )
    metrics['Precision'] = (prec_mean, prec_lower, prec_upper)
    
    # Recall (Sensitivity)
    recall_mean, recall_lower, recall_upper = bootstrap_ci(
        y_test, y_pred_proba, recall_score, n_bootstraps=n_bootstraps, threshold=threshold
    )
    metrics['Recall'] = (recall_mean, recall_lower, recall_upper)
    
    # F1 Score
    f1_mean, f1_lower, f1_upper = bootstrap_ci(
        y_test, y_pred_proba, f1_score, n_bootstraps=n_bootstraps, threshold=threshold
    )
    metrics['F1 Score'] = (f1_mean, f1_lower, f1_upper)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    
    # Calculate curves
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    prob_true, prob_pred = calibration_curve(
        y_test, 
        y_pred_proba, 
        n_bins=20,
        strategy='quantile'
    )
    
    # Calculate decision curve
    thresholds, nb_model, nb_all, nb_none = decision_curve_analysis(y_test, y_pred_proba)
    
    # Compile results
    results = {
        'Metrics': metrics,
        'Specificity': specificity,
        'NPV': npv,
        'ROC': [fpr, tpr],
        'PRC': [precision, recall],
        'Calibration': [prob_true, prob_pred],
        'Decision Curve': [thresholds, nb_model, nb_all, nb_none],
        'Optimal Threshold': threshold
    }
    
    # Print results
    print(f"\nModel: {model_name}")
    print(f"AUC: {metrics['AUC'][0]:.4f} (95% CI: {metrics['AUC'][1]:.4f}--{metrics['AUC'][2]:.4f})")
    print(f"PRC: {metrics['PRC'][0]:.4f} (95% CI: {metrics['PRC'][1]:.4f}--{metrics['PRC'][2]:.4f})")
    print(f"Accuracy: {metrics['Accuracy'][0]:.4f} (95% CI: {metrics['Accuracy'][1]:.4f}--{metrics['Accuracy'][2]:.4f})")
    print(f"Precision: {metrics['Precision'][0]:.4f} (95% CI: {metrics['Precision'][1]:.4f}--{metrics['Precision'][2]:.4f})")
    print(f"Recall: {metrics['Recall'][0]:.4f} (95% CI: {metrics['Recall'][1]:.4f}--{metrics['Recall'][2]:.4f})")
    print(f"F1 Score: {metrics['F1 Score'][0]:.4f} (95% CI: {metrics['F1 Score'][1]:.4f}--{metrics['F1 Score'][2]:.4f})")
    print(f"Specificity: {specificity:.4f}")
    print(f"NPV: {npv:.4f}")
    
    return results


def evaluate_models(models, X_test_linear, X_test_tree, y_test, output_dir, n_bootstraps=500):
    """
    Evaluate multiple models and calculate performance metrics
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test_linear : DataFrame
        Test features for linear models
    X_test_tree : DataFrame
        Test features for tree-based models
    y_test : Series or array
        True labels
    output_dir : str
        Directory to save evaluation results
        
    Returns:
    --------
    dict
        Evaluation results for all models
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    predictions = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        start_time = time.time()
        
        try:
            # Select appropriate dataset based on model type
            if model_name in ["Logistic Regression", "Gaussian Naive Bayes", "Multilayer Perceptron", "Support Vector Machine"]:
                X_test_model = X_test_linear
            elif model_name == "Complement Naive Bayes":
                # Make sure all values are positive for Complement Naive Bayes
                X_test_model = X_test_linear.copy()
                min_val = X_test_model.min().min()
                if min_val < 0:
                    X_test_model = X_test_model - min_val + 0.1
            else:
                X_test_model = X_test_tree
            
            # Find optimal threshold using F1 score
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_f1 = 0
            best_threshold = 0.5
            
            for thresh in thresholds:
                y_pred_thresh = (y_pred_proba >= thresh).astype(int)
                f1 = f1_score(y_test, y_pred_thresh)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
            
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            
            # Store predictions
            predictions[model_name] = {
                'probabilities': y_pred_proba,
                'predictions': y_pred,
                'threshold': best_threshold
            }
            
            # Evaluate the model
            model_results = evaluate_model(
                model_name, model, X_test_model, y_test, 
                threshold=best_threshold, n_bootstraps=n_bootstraps
            )
            
            results[model_name] = model_results
            print(f"Evaluation time: {time.time() - start_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    joblib.dump(results, f'{output_dir}/model_results.pkl')
    joblib.dump(predictions, f'{output_dir}/model_predictions.pkl')
    
    return results, predictions


def analyze_overfitting(models, X_development_linear, X_development_tree, X_test_linear, X_test_tree, 
                        y_development, y_test, output_dir):
    """
    Analyze overfitting by comparing model performance on development and test sets
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_development_linear, X_development_tree : DataFrame
        Development set features (already used for training and calibration)
    X_test_linear, X_test_tree : DataFrame
        Test features
    y_development, y_test : Series or array
        Development and test labels
    output_dir : str
        Directory to save analysis results
        
    Returns:
    --------
    dict
        Overfitting analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    development_test_metrics = {}
    
    for model_name, model in models.items():
        print(f"\nAnalyzing overfitting for model: {model_name}")
        
        # Select appropriate dataset based on model type
        if model_name in ["Logistic Regression", "Gaussian Naive Bayes", "Multilayer Perceptron", "Support Vector Machine"]:
            X_development_model = X_development_linear
            X_test_model = X_test_linear
        elif model_name == "Complement Naive Bayes":
            X_development_model = X_development_linear.copy()
            X_test_model = X_test_linear.copy()
            min_val = X_development_model.min().min()
            if min_val < 0:
                X_development_model = X_development_model - min_val + 0.1
                X_test_model = X_test_model - min_val + 0.1
        else:
            X_development_model = X_development_tree
            X_test_model = X_test_tree
        
        # Get predictions
        development_pred_proba = model.predict_proba(X_development_model)[:, 1]
        test_pred_proba = model.predict_proba(X_test_model)[:, 1]
        
        # Use default threshold of 0.5
        threshold = 0.5
        development_pred = (development_pred_proba >= threshold).astype(int)
        test_pred = (test_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # AUC
        development_auc = roc_auc_score(y_development, development_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        metrics['AUC'] = {'development': development_auc, 'test': test_auc, 'diff': development_auc - test_auc}
        
        # PRC
        development_prc = average_precision_score(y_development, development_pred_proba)
        test_prc = average_precision_score(y_test, test_pred_proba)
        metrics['PRC'] = {'development': development_prc, 'test': test_prc, 'diff': development_prc - test_prc}
        
        # Accuracy
        development_acc = accuracy_score(y_development, development_pred)
        test_acc = accuracy_score(y_test, test_pred)
        metrics['Accuracy'] = {'development': development_acc, 'test': test_acc, 'diff': development_acc - test_acc}
        
        # Precision (PPV)
        development_prec = precision_score(y_development, development_pred)
        test_prec = precision_score(y_test, test_pred)
        metrics['Precision'] = {'development': development_prec, 'test': test_prec, 'diff': development_prec - test_prec}
        
        # Recall (Sensitivity)
        development_recall = recall_score(y_development, development_pred)
        test_recall = recall_score(y_test, test_pred)
        metrics['Recall'] = {'development': development_recall, 'test': test_recall, 'diff': development_recall - test_recall}
        
        # F1 Score
        development_f1 = f1_score(y_development, development_pred)
        test_f1 = f1_score(y_test, test_pred)
        metrics['F1 Score'] = {'development': development_f1, 'test': test_f1, 'diff': development_f1 - test_f1}
        
        # Store results
        development_test_metrics[model_name] = metrics
        
        # Print results
        print(f"Model: {model_name}")
        print(f"AUC - Development: {development_auc:.4f}, Test: {test_auc:.4f}, Difference: {development_auc - test_auc:.4f}")
        print(f"PRC - Development: {development_prc:.4f}, Test: {test_prc:.4f}, Difference: {development_prc - test_prc:.4f}")
        print(f"Accuracy - Development: {development_acc:.4f}, Test: {test_acc:.4f}, Difference: {development_acc - test_acc:.4f}")
        print(f"Precision - Development: {development_prec:.4f}, Test: {test_prec:.4f}, Difference: {development_prec - test_prec:.4f}")
        print(f"Recall - Development: {development_recall:.4f}, Test: {test_recall:.4f}, Difference: {development_recall - test_recall:.4f}")
        print(f"F1 Score - Development: {development_f1:.4f}, Test: {test_f1:.4f}, Difference: {development_f1 - test_f1:.4f}")
    
    # Save overfitting analysis results
    joblib.dump(development_test_metrics, f'{output_dir}/overfitting_analysis.pkl')
    
    return development_test_metrics