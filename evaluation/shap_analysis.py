import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shap
import joblib


def generate_shap_analysis(model, X_test, model_name, output_dir, sample_size=500):
    """
    Generate SHAP analysis for a given model
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : DataFrame
        Test data
    model_name : str
        Name of the model
    output_dir : str
        Directory to save analysis
    sample_size : int, optional
        Number of samples to use for SHAP analysis
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use a sample of the test data for faster computation
        if len(X_test) > sample_size:
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_test_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
        else:
            X_test_sample = X_test
        
        # Extract model from pipeline if needed
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            clf = model.named_steps['classifier']
        else:
            clf = model
        
        # Create SHAP explainer based on model type
        if model_name in ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting", "AdaBoost"]:
            explainer = shap.TreeExplainer(clf)
        elif model_name in ["Logistic Regression", "Support Vector Machine"]:
            # For kernel explainer, we need a prediction function
            def predict_func(X):
                return model.predict_proba(X)[:, 1]
            
            explainer = shap.KernelExplainer(predict_func, shap.sample(X_test_sample, 100))
        else:
            print(f"SHAP analysis for {model_name} is not supported.")
            return
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For binary classification, we take the values for the positive class (index 1)
            shap_values = shap_values[1]
        
        # Create bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
        plt.title('A', fontsize=16, fontweight='bold', loc='left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary plot
        plt.figure(figsize=(9, 8))
        shap.summary_plot(shap_values, X_test_sample, show=False)
        plt.title('B', fontsize=16, fontweight='bold', loc='left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combine images
        img1 = Image.open(f'{output_dir}/{model_name}_shap_bar_plot.png')
        img2 = Image.open(f'{output_dir}/{model_name}_shap_summary_plot.png')
        
        width = img1.width + img2.width
        height = max(img1.height, img2.height)
        combined_img = Image.new('RGB', (width + 100, height + 50), color='white')
        
        combined_img.paste(img1, (50, 25))
        combined_img.paste(img2, (img1.width + 50, 25))
        
        combined_img.save(f'{output_dir}/{model_name}_shap_combined_plot.png')
        
        print(f"SHAP analysis for {model_name} completed successfully")
        
    except Exception as e:
        print(f"Error in SHAP analysis for {model_name}: {e}")
        import traceback
        traceback.print_exc()


def analyze_top_models(results, models, X_test_linear, X_test_tree, output_dir, top_n=3):
    """
    Perform SHAP analysis on top performing models
    
    Parameters:
    -----------
    results : dict
        Results from model evaluation
    models : dict
        Dictionary of trained models
    X_test_linear : DataFrame
        Test features for linear models
    X_test_tree : DataFrame
        Test features for tree-based models
    output_dir : str
        Directory to save analysis
    top_n : int, optional
        Number of top models to analyze
        
    Returns:
    --------
    list
        Names of analyzed models
    """
    # Sort models by AUC
    model_metrics = [(name, res['Metrics']['AUC'][0]) for name, res in results.items()]
    sorted_models = sorted(model_metrics, key=lambda x: x[1], reverse=True)
    
    # Select top N models
    top_models = sorted_models[:top_n]
    print(f"\nGenerating SHAP analysis for top {top_n} models:")
    for name, score in top_models:
        print(f"- {name} (AUC: {score:.4f})")
    
    # Generate SHAP analysis for top models
    analyzed_models = []
    for model_name, _ in top_models:
        if model_name in models:
            # Select appropriate dataset
            if model_name in ["Logistic Regression", "Gaussian Naive Bayes", "Multilayer Perceptron", "Support Vector Machine"]:
                X_test_model = X_test_linear
            elif model_name == "Complement Naive Bayes":
                X_test_model = X_test_linear.copy()
                min_val = X_test_model.min().min()
                if min_val < 0:
                    X_test_model = X_test_model - min_val + 0.1
            else:
                X_test_model = X_test_tree
            
            print(f"\nGenerating SHAP analysis for {model_name}...")
            generate_shap_analysis(models[model_name], X_test_model, model_name, output_dir)
            analyzed_models.append(model_name)
    
    return analyzed_models