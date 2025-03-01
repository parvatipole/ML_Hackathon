import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import tensorflow as tf
from zero_day_detection import ZeroDayDetector, Sampling, VAE

def load_data(file_path):
    """Load and preprocess the data for inference"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Zero-Day Exploit Detection Inference')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing the trained model')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to save the results')
    parser.add_argument('--threshold', type=float, default=None, help='Custom threshold for anomaly detection')
    args = parser.parse_args()
    
    # Load the data
    df = load_data(args.input)
    
    # Initialize the detector
    detector = ZeroDayDetector()
    
    # Register custom objects for model loading
    tf.keras.utils.get_custom_objects().update({
        'Sampling': Sampling,
        'VAE': VAE
    })
    
    # Load the trained model
    detector.load_model(args.model_dir)
    
    # Preprocess the data
    X, _ = detector.preprocess_data(df)
    
    # Set custom threshold if provided
    if args.threshold is not None:
        detector.threshold = args.threshold
        print(f"Using custom threshold: {detector.threshold}")
    
    # Predict anomalies
    predictions, anomaly_scores, reconstruction_errors = detector.predict_anomalies(X)
    
    # Get latent representations
    latent_representations = detector.get_latent_representation(X)
    
    # Add results to the dataframe
    df['anomaly'] = predictions
    df['anomaly_score'] = anomaly_scores
    df['reconstruction_error'] = reconstruction_errors
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\nInference Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Detected anomalies: {predictions.sum()} ({predictions.sum()/len(df)*100:.2f}%)")
    
    # Visualize results
    try:
        # Plot reconstruction error distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(reconstruction_errors, bins=50, kde=True)
        plt.axvline(detector.threshold, color='r', linestyle='--', label=f'Threshold: {detector.threshold:.6f}')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('inference_reconstruction_error.png')
        
        # Plot anomaly scores
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=predictions, cmap='coolwarm', alpha=0.6)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
        plt.title('Anomaly Scores')
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.colorbar(label='Prediction (0=Normal, 1=Anomaly)')
        plt.legend()
        plt.savefig('inference_anomaly_scores.png')
        
        # Plot latent space with TSNE if there are enough samples
        if len(df) >= 50:
            from sklearn.manifold import TSNE
            
            # Apply t-SNE to reduce dimensionality to 2D for visualization
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(latent_representations)
            
            # Plot t-SNE visualization
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=predictions, cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, label='Prediction (0=Normal, 1=Anomaly)')
            plt.title('t-SNE Visualization of Latent Space')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.savefig('inference_tsne_visualization.png')
        
        print("Visualizations saved as PNG files.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main() 