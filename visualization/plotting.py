import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import savgol_filter
import sys
sys.path.append('..')
from utils import decision_curve_analysis


def plot_performance_comparison(results, output_dir):
    """
    Create a performance comparison plot for all models (ROC, PRC, Calibration)
    
    Parameters:
    -----------
    results : dict
        Model evaluation results
    output_dir : str
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ABC-labeled figure: ROC, PRC, and calibration curves in one row
    plt.figure(figsize=(18, 6))

    # A: ROC curves
    plt.subplot(1, 3, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    for model_name in results.keys():
        fpr, tpr = results[model_name]['ROC']
        auc_value = results[model_name]['Metrics']['AUC'][0]
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_value:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('A', fontsize=16, fontweight='bold', loc='left')
    plt.legend(loc="lower right", fontsize=8)

    # B: Precision-Recall curves
    plt.subplot(1, 3, 2)
    plt.grid(True, linestyle='--', alpha=0.6)
    for model_name in results.keys():
        precision, recall = results[model_name]['PRC']
        prc_value = results[model_name]['Metrics']['PRC'][0]
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AUPRC = {prc_value:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('B', fontsize=16, fontweight='bold', loc='left')
    plt.legend(loc="upper right", fontsize=8)

    # C: Calibration curves
    plt.subplot(1, 3, 3)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    for model_name in results.keys():
        prob_true, prob_pred = results[model_name]['Calibration']
        plt.plot(prob_pred, prob_true, '-', lw=2, label=f'{model_name}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('C', fontsize=16, fontweight='bold', loc='left')
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    plt.savefig(f'{output_dir}/model_performance_comparison.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/model_performance_comparison.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_decision_curves(predictions, y_test, output_dir):
    """
    Create decision curve analysis plot for all models
    
    Parameters:
    -----------
    predictions : dict
        Model predictions
    y_test : Series or array
        True labels
    output_dir : str
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set threshold range and min benefit
    thresholds = np.arange(0.01, 0.99, 0.005)
    min_benefit = -0.2

    # Plot one reference curve (treat all/none)
    _, _, net_benefit_all, net_benefit_none = decision_curve_analysis(
        y_test, predictions[list(predictions.keys())[0]]['probabilities'], thresholds)
    plt.plot(thresholds, net_benefit_all, 'k--', lw=2, label='Treat All')
    plt.plot(thresholds, net_benefit_none, 'k:', lw=2, label='Treat None')

    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#bcbd22']

    # Plot each model's decision curve
    for (model_name, color) in zip(predictions.keys(), colors):
        y_pred_proba = predictions[model_name]['probabilities']
        _, nb_model, _, _ = decision_curve_analysis(y_test, y_pred_proba, thresholds)
        nb_model_array = np.array(nb_model)
        
        # Smooth the curve
        nb_model_smooth = savgol_filter(np.maximum(nb_model_array, min_benefit), 
                                     window_length=21,
                                     polyorder=3)
        
        plt.plot(thresholds, nb_model_smooth, 
                '-', color=color, lw=2, label=model_name)

    plt.xlim([0, 1])
    plt.ylim([min_benefit, 0.4])
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    plt.savefig(f'{output_dir}/decision_curve_analysis_all_models.svg', 
               format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/decision_curve_analysis_all_models.png', 
               format='png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_overfitting_analysis(train_test_metrics, output_dir):
    """
    Create visualizations to analyze overfitting
    
    Parameters:
    -----------
    train_test_metrics : dict
        Dictionary with training and test metrics
    output_dir : str
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and metrics
    model_names = list(train_test_metrics.keys())
    metrics_list = ['AUC', 'PRC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Create a heatmap of differences
    diff_data = []
    for model_name in model_names:
        model_diffs = {}
        model_diffs['Model'] = model_name
        for metric in metrics_list:
            model_diffs[metric] = train_test_metrics[model_name][metric]['diff']
        diff_data.append(model_diffs)
    
    diff_df = pd.DataFrame(diff_data)
    diff_df.set_index('Model', inplace=True)
    
    # Set custom color range
    vmin = 0
    vmax = 0.2
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(diff_df, annot=True, fmt=".4f", 
                cmap="Reds",
                vmin=vmin, vmax=vmax,
                linewidths=.5, 
                cbar_kws={'label': 'Train-Test Difference'})
    
    plt.title('Model Overfitting: Difference Between Training and Test Metrics', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/overfitting_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/overfitting_heatmap.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar charts comparing train and test metrics
    for metric in metrics_list:
        plt.figure(figsize=(16, 10))
        
        # Prepare data
        train_values = [train_test_metrics[model][metric]['train'] for model in model_names]
        test_values = [train_test_metrics[model][metric]['test'] for model in model_names]
        diff_values = [train_test_metrics[model][metric]['diff'] for model in model_names]
        
        # Set x positions
        x = np.arange(len(model_names))
        width = 0.35
        
        # Plot bar charts
        plt.bar(x - width/2, train_values, width, label='Training Set', color='skyblue')
        plt.bar(x + width/2, test_values, width, label='Test Set', color='lightcoral')
        
        # Determine the maximum value for y-axis limit
        max_value = max(max(train_values), max(test_values))
        
        # Set y-axis limit with extra space for labels and title
        y_max = max_value * 1.15
        plt.ylim(0, y_max)
        
        # Add data labels
        for j, value in enumerate(train_values):
            if value > 0.05:
                plt.text(j - width/2, value + (max_value * 0.01), 
                        f"{value:.3f}", ha='center', va='bottom', 
                        rotation=90, fontsize=9)
        for j, value in enumerate(test_values):
            if value > 0.05:
                plt.text(j + width/2, value + (max_value * 0.01), 
                        f"{value:.3f}", ha='center', va='bottom', 
                        rotation=90, fontsize=9)
        
        # Set figure properties
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{metric} - Training vs. Test Set', fontsize=14, pad=20)
        
        # Adjust x-ticks with more space
        plt.xticks(x, [m.replace(' ', '\n') for m in model_names], 
                  rotation=45, ha='right', fontsize=10)
        
        # Add legend and grid
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add average difference text
        avg_diff = np.mean(diff_values)
        plt.text(0.05, 0.90, f'Avg Diff: {avg_diff:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.95)
        
        plt.savefig(f'{output_dir}/train_test_{metric}_comparison.png', 
                    dpi=300, bbox_inches='tight', pad_inches=1.0)
        plt.savefig(f'{output_dir}/train_test_{metric}_comparison.svg', 
                    format='svg', dpi=300, bbox_inches='tight', pad_inches=1.0)
        plt.close()
    
    print("Overfitting analysis visualizations created successfully")


def create_performance_table(results, output_dir):
    """
    Create a table summarizing model performance metrics
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
    output_dir : str
        Directory to save the table
        
    Returns:
    --------
    DataFrame
        Performance metrics table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Format metrics with confidence intervals
    def format_with_ci(mean, lower, upper):
        return f"{mean:.3f} ({lower:.3f}-{upper:.3f})"
    
    table_data = []
    for model_name, model_results in results.items():
        model_row = {'Models': model_name}
        
        metrics = model_results['Metrics']
        for metric_name in ['AUC', 'PRC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']:
            if metric_name in metrics:
                mean, lower, upper = metrics[metric_name]
                model_row[metric_name] = format_with_ci(mean, lower, upper)
        
        if 'Specificity' in model_results:
            model_row['Specificity'] = f"{model_results['Specificity']:.3f}"
        
        if 'NPV' in model_results:
            model_row['NPV'] = f"{model_results['NPV']:.3f}"
        
        table_data.append(model_row)
    
    # Create DataFrame
    metrics_table = pd.DataFrame(table_data)
    
    # Reorder columns
    column_order = ['Models', 'AUC', 'PRC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'NPV', 'F1 Score']
    metrics_table = metrics_table[[col for col in column_order if col in metrics_table.columns]]
    
    # Save to CSV
    metrics_table.to_csv(f'{output_dir}/model_performance_metrics.csv', index=False)
   
    return metrics_table