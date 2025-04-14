import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb


def create_model_pipelines(continuous_features, categorical_features):
    """
    Create machine learning model pipelines with appropriate preprocessing
    
    Parameters:
    -----------
    continuous_features : list
        List of continuous feature names
    categorical_features : list
        List of categorical feature names
        
    Returns:
    --------
    dict
        Dictionary of model pipelines
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('splines', SplineTransformer(n_knots=5, degree=3), continuous_features),
            ('cat', 'passthrough', categorical_features)
        ])
    
    models = {
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=0.5,                  
                penalty='l2',           
                solver='liblinear',     
                max_iter=2000,          
                fit_intercept=True,     
                class_weight='balanced',
                random_state=42,
                n_jobs=-1               
            ))
        ]),
        
        "Gaussian Naive Bayes": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB(
                var_smoothing=1e-7     
            ))
        ]),
        
        "Complement Naive Bayes": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', ComplementNB(
                alpha=0.7,             
                norm=True,             
                fit_prior=True         
            ))
        ]),
        
        "Support Vector Machine": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                C=1.0,                 
                gamma='scale',         
                kernel='rbf',          
                probability=True,      
                class_weight='balanced',
                cache_size=1000,       
                random_state=42
            ))
        ]),
        
        "Multilayer Perceptron": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(100, 50),  
                activation='relu',      
                solver='adam',          
                alpha=0.0005,           
                batch_size=64,          
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,          
                early_stopping=True,    
                validation_fraction=0.2,
                n_iter_no_change=20,    
                random_state=42
            ))
        ]),
        
        "AdaBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=100,       
                learning_rate=0.1,      
                algorithm='SAMME.R',    
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,      
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',   
                oob_score=True,
                max_samples=0.7,           
                ccp_alpha=0.01,            
                n_jobs=-1,
                random_state=42
            ))
        ]),
        
        "Gradient Boosting": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.02,        
                max_depth=4,
                min_samples_split=25,      
                min_samples_leaf=10,
                subsample=0.8,
                max_features=0.7,
                validation_fraction=0.2,   
                n_iter_no_change=15,       
                tol=0.001,
                random_state=42
            ))
        ]),
        
        "LightGBM": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.01,        
                n_estimators=150,
                min_child_samples=20,
                feature_fraction=0.7,
                bagging_fraction=0.8,
                bagging_freq=5,
                max_depth=6,
                reg_alpha=0.5,
                reg_lambda=1.0,
                class_weight='balanced',
                boosting_type='gbdt',
                verbose=-1,
                random_state=42
            ))
        ]),
        
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                learning_rate=0.01,        
                max_depth=5,
                n_estimators=150,
                subsample=0.8,
                colsample_bytree=0.7,
                colsample_bylevel=0.7,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.5,
                reg_lambda=1.0,
                scale_pos_weight=3.44,
                tree_method='hist',
                grow_policy='lossguide',
                n_jobs=-1,
                random_state=42,
                verbosity=0
            ))
        ])
    }
    
    return models


def train_models(models, X_development_linear, X_development_tree, y_development, output_dir):
    """
    Train and calibrate machine learning models with proper validation
    
    Parameters:
    -----------
    models : dict
        Dictionary of model pipelines
    X_development_linear : DataFrame
        Linear model features for development set (will be split into train/calibration)
    X_development_tree : DataFrame
        Tree-based model features for development set (will be split into train/calibration)
    y_development : Series or array
        Target variable for development set (will be split into train/calibration)
    output_dir : str
        Directory to save trained models
        
    Returns:
    --------
    tuple
        (trained_models, calibrated_models)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Split development data into training and calibration sets
    X_train_linear, X_cal_linear, y_train, y_cal = train_test_split(
        X_development_linear, y_development, test_size=0.2, random_state=42, stratify=y_development
    )
    
    X_train_tree, X_cal_tree, _, _ = train_test_split(
        X_development_tree, y_development, test_size=0.2, random_state=42, stratify=y_development
    )
    
    # Create positive-only copy for Complement Naive Bayes
    X_train_cnb = X_train_linear.copy()
    min_val = X_train_cnb.min().min()
    if min_val < 0:
        X_train_cnb = X_train_cnb - min_val + 0.1
        
    X_cal_cnb = X_cal_linear.copy()
    min_val = X_cal_cnb.min().min()
    if min_val < 0:
        X_cal_cnb = X_cal_cnb - min_val + 0.1
    
    trained_models = {}
    calibrated_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        
        try:
            # Select appropriate dataset based on model type
            if model_name in ["Logistic Regression", "Gaussian Naive Bayes", "Multilayer Perceptron", "Support Vector Machine"]:
                X_train_model = X_train_linear
                X_cal_model = X_cal_linear
            elif model_name == "Complement Naive Bayes":
                X_train_model = X_train_cnb
                X_cal_model = X_cal_cnb
            else:
                X_train_model = X_train_tree
                X_cal_model = X_cal_tree
            
            # Train the model
            model.fit(X_train_model, y_train)
            trained_models[model_name] = model
            
            # Save base model
            joblib.dump(model, f'{output_dir}/{model_name.replace(" ", "_")}_base_model.pkl')
            
            # Select calibration method based on model type
            if model_name in ["Logistic Regression", "Support Vector Machine"]:
                calibration_method = 'sigmoid'
            else:
                calibration_method = 'isotonic'
                
            # Calibrate the model using calibration set
            calibrated = CalibratedClassifierCV(
                base_estimator=model,
                method=calibration_method,
                cv='prefit',
                n_jobs=-1
            )
            calibrated.fit(X_cal_model, y_cal)
            calibrated_models[model_name] = calibrated
            
            # Save calibrated model
            joblib.dump(calibrated, f'{output_dir}/{model_name.replace(" ", "_")}_calibrated_model.pkl')
            
            print(f"Successfully trained and calibrated {model_name}")
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return trained_models, calibrated_models