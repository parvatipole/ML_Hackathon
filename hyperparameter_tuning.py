import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import itertools
import time
import os
import tensorflow as tf
from zero_day_detection import ZeroDayDetector, Sampling, VAE

def load_data(file_path):
    """Load and preprocess the data for hyperparameter tuning"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    return df

def evaluate_model(y_true, y_pred, anomaly_scores):
    """Evaluate model performance using various metrics"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, anomaly_scores)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def main():
    # Register custom objects for model loading
    tf.keras.utils.get_custom_objects().update({
        'Sampling': Sampling,
        'VAE': VAE
    })
    
    # Configuration
    data_file = 'UNSW_NB15_training-set.csv'
    results_file = 'hyperparameter_tuning_results.csv'
    
    # Hyperparameter grid
    param_grid = {
        'latent_dim': [8, 16, 32, 64],
        'intermediate_dim': [64, 128, 256],
        'batch_size': [128, 256],
        'epochs': [20, 30],
        'contamination': [0.01, 0.05, 0.1]
    }
    
    # Load the data
    df = load_data(data_file)
    
    # Split data into normal and attack samples
    normal_df = df[df['label'] == 0]
    attack_df = df[df['label'] == 1]
    
    print(f"Normal samples: {len(normal_df)}")
    print(f"Attack samples: {len(attack_df)}")
    
    # Split normal data into train and validation sets
    normal_train, normal_val = train_test_split(normal_df, test_size=0.2, random_state=42)
    
    # Create a balanced validation set with normal and attack samples
    val_attack = attack_df.sample(n=len(normal_val), random_state=42)
    val_df = pd.concat([normal_val, val_attack])
    
    # Generate all combinations of hyperparameters
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    # Initialize results list
    results = []
    
    # Loop through all hyperparameter combinations
    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        print(f"\nTesting combination {i+1}/{len(combinations)}:")
        print(params)
        
        # Start timer
        start_time = time.time()
        
        try:
            # Initialize the detector with current hyperparameters
            detector = ZeroDayDetector(
                latent_dim=params['latent_dim'],
                intermediate_dim=params['intermediate_dim'],
                batch_size=params['batch_size'],
                epochs=params['epochs']
            )
            
            # Preprocess the data
            X_train, _ = detector.preprocess_data(normal_train)
            X_val, y_val = detector.preprocess_data(val_df)
            
            # Build and train the VAE model
            detector.build_vae(input_dim=X_train.shape[1])
            history = detector.train_vae(X_train)
            
            # Set threshold for anomaly detection
            detector.set_threshold(X_train, contamination=params['contamination'])
            
            # Predict anomalies
            y_pred, anomaly_scores, _ = detector.predict_anomalies(X_val)
            
            # Evaluate model performance
            metrics = evaluate_model(y_val, y_pred, anomaly_scores)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Add results
            result = {
                **params,
                **metrics,
                'training_time': training_time
            }
            results.append(result)
            
            print(f"Results: {metrics}")
            print(f"Training time: {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error with combination {i+1}: {e}")
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Find best model based on F1 score
    if results:
        best_idx = results_df['f1_score'].idxmax()
        best_params = results_df.iloc[best_idx]
        
        print("\nBest hyperparameters:")
        for key in keys:
            print(f"{key}: {best_params[key]}")
        
        print("\nBest metrics:")
        print(f"Precision: {best_params['precision']:.4f}")
        print(f"Recall: {best_params['recall']:.4f}")
        print(f"F1 Score: {best_params['f1_score']:.4f}")
        print(f"AUC: {best_params['auc']:.4f}")
        
        # Visualize results
        try:
            # Plot F1 scores for different latent dimensions
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='latent_dim', y='f1_score', data=results_df)
            plt.title('F1 Score vs Latent Dimension')
            plt.savefig('tuning_latent_dim.png')
            
            # Plot F1 scores for different contamination values
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='contamination', y='f1_score', data=results_df)
            plt.title('F1 Score vs Contamination')
            plt.savefig('tuning_contamination.png')
            
            # Plot training time vs model complexity
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x='latent_dim', y='training_time', hue='intermediate_dim', size='batch_size', data=results_df)
            plt.title('Training Time vs Model Complexity')
            plt.savefig('tuning_training_time.png')
            
            print("Visualizations saved as PNG files.")
        except Exception as e:
            print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main() 