import os
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from data_processing.preprocessing import *
from models.model_training import create_model_pipelines, train_models
from evaluation.metrics import evaluate_models, analyze_overfitting
from evaluation.shap_analysis import analyze_top_models
from visualization.plotting import *
from utils import get_continuous_columns, get_categorical_columns


def main(config):
    """
    Main function to execute the complete immunocompromised mortality prediction workflow.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters for the analysis
    """
    print("Starting immunocompromised mortality prediction workflow...")
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    models_dir = os.path.join(config['output_dir'], 'models')
    results_dir = os.path.join(config['output_dir'], 'results')
    figures_dir = os.path.join(config['output_dir'], 'figures')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    cohort_data, dynamic_data = load_and_filter_data(config['cohort_path'], config['dynamic_path'])
    
    # Process measurements from the first 24 hours
    processed_measurements = process_measurements(dynamic_data, cohort_data)
    
    # Prepare and clean the dataset for modeling
    merged_data, columns_to_exclude = prepare_model_dataset(cohort_data, processed_measurements)
    
    # Split data and handle missing values
    X_train, X_test, y_train, y_test, train_data, test_data = handle_missing_values(
        merged_data, columns_to_exclude, random_state=config['random_state']
    )
    
    # Save processed data
    joblib.dump(train_data, os.path.join(results_dir, 'train_data.pkl'))
    joblib.dump(test_data, os.path.join(results_dir, 'test_data.pkl'))
    
    # Step 2: Feature engineering and preparation for different model types
    print("\nStep 2: Feature engineering and preparation for model training...")
    
    # Get continuous and categorical columns
    continuous_features = get_continuous_columns(X_train)
    categorical_features = get_categorical_columns(X_train)
    
    # Print feature information
    print(f"Number of continuous features: {len(continuous_features)}")
    print(f"Number of categorical features: {len(categorical_features)}")
    
    # Step 3: Create and train models
    print("\nStep 3: Creating and training models...")
    models = create_model_pipelines(continuous_features, categorical_features)
    
    # Prepare linear and tree-based feature sets
    X_train_linear = X_train.copy()
    X_test_linear = X_test.copy()
    X_train_tree = X_train.copy()
    X_test_tree = X_test.copy()
    
    # Train models
    trained_models, calibrated_models = train_models(
        models, X_train_linear, X_train_tree, y_train, models_dir
    )
    
    # Step 4: Evaluate models
    print("\nStep 4: Evaluating models...")
    results, predictions = evaluate_models(
        calibrated_models, X_test_linear, X_test_tree, y_test, results_dir
    )
    
    # Analyze potential overfitting
    train_test_metrics = analyze_overfitting(
        calibrated_models, X_train_linear, X_train_tree, 
        X_test_linear, X_test_tree, y_train, y_test, results_dir
    )
    
    # Step 5: Generate visualizations
    print("\nStep 5: Generating visualizations...")
    # Plot model performance comparison
    plot_performance_comparison(results, figures_dir)
    
    # Plot decision curves
    plot_decision_curves(predictions, y_test, figures_dir)
    
    # Plot overfitting analysis
    plot_overfitting_analysis(train_test_metrics, figures_dir)
    
    # Create performance metrics table
    metrics_table = create_performance_table(results, results_dir)
    print("\nModel performance metrics table created and saved.")
    
    # Step 6: Generate SHAP analysis for interpretability
    print("\nStep 6: Generating SHAP analysis for model interpretability...")
    shap_dir = os.path.join(figures_dir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)
    
    analyzed_models = analyze_top_models(
        results, calibrated_models, X_test_linear, X_test_tree, shap_dir
    )
    
    print(f"\nCompleted SHAP analysis for top models: {', '.join(analyzed_models)}")
    
    print("\nWorkflow completed successfully!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run immunocompromised mortality prediction workflow")
    
    parser.add_argument("--cohort_path", type=str, required=False,
                       default="data/cohort_data.csv",
                       help="Path to cohort data CSV file")
    
    parser.add_argument("--dynamic_path", type=str, required=False,
                       default="data/dynamic_data.parquet",
                       help="Path to dynamic data Parquet file")
    
    parser.add_argument("--output_dir", type=str, required=False,
                       default="output",
                       help="Directory to save all outputs")
    
    parser.add_argument("--random_state", type=int, required=False,
                       default=42,
                       help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    # Create configuration dictionary
    config = {
        'cohort_path': args.cohort_path,
        'dynamic_path': args.dynamic_path,
        'output_dir': args.output_dir,
        'random_state': args.random_state,
    }
    
    # Execute the main workflow
    main(config)