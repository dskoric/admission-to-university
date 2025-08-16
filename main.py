"""
Main entry point for the university admission prediction project.
"""

import os
import pandas as pd
from pathlib import Path

# Import project modules
from src.data_processing import load_data, preprocess_data, split_data
from src.logistic_regression import LogisticRegressionModel
from src.model_evaluation import calculate_metrics, get_feature_importance
from src.visualization import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_feature_importance, plot_correlation_matrix
)

# Set paths
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / 'data'
models_dir = project_dir / 'models'
results_dir = project_dir / 'results'
plots_dir = results_dir / 'plots'

# Create directories if they don't exist
models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

def main():
    print("University Admission Prediction Project")
    print("=" * 40)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    raw_data_path = data_dir / 'raw' / 'admission_data.csv'
    processed_data_path = data_dir / 'processed' / 'admission_processed.csv'
    
    # Load raw data
    df = load_data(raw_data_path)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocess data
    df_processed = preprocess_data(df)
    # Save processed data
    df_processed.to_csv(processed_data_path, index=False)
    print(f"Data preprocessed and saved to {processed_data_path}")
    
    # 2. Split data into training and testing sets
    print("\n2. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test, scaler = split_data(df_processed)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train the logistic regression model
    print("\n3. Training the logistic regression model...")
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    
    # Save the model
    model_path = models_dir / 'logistic_regression_model.pkl'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # 4. Evaluate the model
    print("\n4. Evaluating the model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [metrics['accuracy'], metrics['precision'], 
                 metrics['recall'], metrics['f1_score'], metrics['roc_auc']]
    })
    metrics_path = results_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # 5. Generate visualizations
    print("\n5. Generating visualizations...")
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(y_test, y_pred)
    cm_path = plots_dir / 'confusion_matrix.png'
    cm_fig.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Plot ROC curve
    roc_fig = plot_roc_curve(y_test, y_proba)
    roc_path = plots_dir / 'roc_curve.png'
    roc_fig.savefig(roc_path)
    print(f"ROC curve saved to {roc_path}")
    
    # Plot feature importance
    feature_names = df_processed.drop(['Chance of Admit', 'Admit'], axis=1).columns
    importance_df = get_feature_importance(model.model, feature_names)
    importance_fig = plot_feature_importance(importance_df)
    importance_path = plots_dir / 'feature_importance.png'
    importance_fig.savefig(importance_path)
    print(f"Feature importance plot saved to {importance_path}")
    
    # Plot correlation matrix
    corr_fig = plot_correlation_matrix(df_processed.drop(['Admit'], axis=1))
    corr_path = plots_dir / 'correlation_matrix.png'
    corr_fig.savefig(corr_path)
    print(f"Correlation matrix saved to {corr_path}")
    
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()