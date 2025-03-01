import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add the parent directory to the Python path so we can import the zero_day_detection module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Now import from zero_day_detection
from zero_day_detection import Sampling, VAE

def preprocess_data(df):
    """Preprocess the data for the VAE model"""
    print("Preprocessing data...")
    
    # Drop the id column as it's just an index
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Separate categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target columns from features
    if 'label' in numerical_cols:
        numerical_cols.remove('label')
    if 'attack_cat' in categorical_cols:
        categorical_cols.remove('attack_cat')
    
    # Extract features and targets
    X = df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
    y = df['label'] if 'label' in df.columns else None
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y

def load_models(model_dir):
    """Load the trained models"""
    print(f"Loading models from {model_dir}...")
    
    # Register custom objects for model loading
    tf.keras.utils.get_custom_objects().update({
        'Sampling': Sampling,
        'VAE': VAE
    })
    
    # Load encoder and decoder models
    encoder = tf.keras.models.load_model(os.path.join(model_dir, 'encoder_model.keras'))
    decoder = tf.keras.models.load_model(os.path.join(model_dir, 'decoder_model.keras'))
    
    # Load threshold
    threshold = np.load(os.path.join(model_dir, 'threshold.npy'))
    
    print(f"Models loaded successfully. Threshold: {threshold}")
    
    return encoder, decoder, threshold

def process_in_batches(data, encoder, decoder, batch_size=64):
    """Process data in batches to avoid memory issues"""
    print(f"Processing data in batches (batch size: {batch_size})...")
    
    num_samples = data.shape[0]
    reconstructions = []
    latent_vectors = []
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        print(f"Processing batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}: samples {i} to {end_idx-1}")
        
        batch = data[i:end_idx]
        
        # Get latent representation
        z_mean, z_log_var, z = encoder(batch)
        latent_vectors.append(z_mean.numpy())
        
        # Reconstruct the data
        reconstructed_batch = decoder(z).numpy()
        reconstructions.append(reconstructed_batch)
    
    # Combine all batch results
    all_reconstructions = np.vstack(reconstructions)
    all_latent_vectors = np.vstack(latent_vectors)
    
    return all_reconstructions, all_latent_vectors

def detect_anomalies(X, X_reconstructed, threshold):
    """Detect anomalies based on reconstruction error"""
    print("Detecting anomalies...")
    
    # Calculate reconstruction error (MSE)
    mse = np.mean(np.square(X - X_reconstructed), axis=1)
    
    # Classify as anomaly if reconstruction error > threshold
    predictions = (mse > threshold).astype(int)
    
    # Calculate anomaly scores (normalized reconstruction error)
    anomaly_scores = mse / threshold
    
    return predictions, anomaly_scores, mse

def create_visualizations(df, predictions, anomaly_scores, reconstruction_errors, latent_vectors, threshold, output_dir):
    """Create visualizations of the results"""
    print("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Plot reconstruction error distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(reconstruction_errors, bins=50, kde=True)
        plt.axvline(threshold, color='r', linestyle='--', 
                   label=f'Threshold: {threshold:.6f}')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'reconstruction_error.png'))
        plt.close()
        
        # Plot anomaly scores
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(anomaly_scores)), anomaly_scores, 
                   c=predictions, cmap='coolwarm', alpha=0.6)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
        plt.title('Anomaly Scores')
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.colorbar(label='Prediction (0=Normal, 1=Anomaly)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'anomaly_scores.png'))
        plt.close()
        
        # Plot latent space with TSNE if there are enough samples
        if len(df) >= 50:
            from sklearn.manifold import TSNE
            
            # Apply t-SNE to reduce dimensionality to 2D for visualization
            print("Applying t-SNE dimensionality reduction...")
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(latent_vectors)
            
            # Plot t-SNE visualization
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=predictions, cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, label='Prediction (0=Normal, 1=Anomaly)')
            plt.title('t-SNE Visualization of Latent Space')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
            plt.close()
        
        print(f"Visualizations saved in {output_dir}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    # Define default paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "UNSW_NB15_testing-set.csv")
    
    # Check if models exist in the script directory first, otherwise use the default models directory
    local_model_dir = script_dir
    default_model_dir = os.path.join(os.path.dirname(script_dir), "models")
    
    # Check if model files exist in the script directory
    if (os.path.exists(os.path.join(local_model_dir, "encoder_model.keras")) and
        os.path.exists(os.path.join(local_model_dir, "decoder_model.keras")) and
        os.path.exists(os.path.join(local_model_dir, "threshold.npy"))):
        model_dir = local_model_dir
    else:
        model_dir = default_model_dir
    
    output_file = os.path.join(script_dir, "results.csv")
    vis_dir = os.path.join(script_dir, "visualizations")
    
    # Parse command line arguments (optional overrides)
    parser = argparse.ArgumentParser(description='Zero-Day Exploit Detection on New Data')
    parser.add_argument('--input', type=str, help='Path to the input CSV file (optional)')
    parser.add_argument('--model_dir', type=str, help='Directory containing the trained model (optional)')
    parser.add_argument('--output', type=str, help='Path to save the results (optional)')
    parser.add_argument('--vis_dir', type=str, help='Directory to save visualizations (optional)')
    parser.add_argument('--threshold', type=float, default=None, help='Custom threshold for anomaly detection')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualizations')
    args = parser.parse_args()
    
    # Override defaults with command line arguments if provided
    if args.input:
        input_file = args.input
    if args.model_dir:
        model_dir = args.model_dir
    if args.output:
        output_file = args.output
    if args.vis_dir:
        vis_dir = args.vis_dir
    
    print(f"Using input file: {input_file}")
    print(f"Using model directory: {model_dir}")
    print(f"Results will be saved to: {output_file}")
    if not args.no_vis:
        print(f"Visualizations will be saved to: {vis_dir}")
    
    # Step 1: Load the data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Data shape: {df.shape}")
    
    # Step 2: Load the models
    encoder, decoder, threshold = load_models(model_dir)
    
    # Override threshold if provided
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using custom threshold: {threshold}")
    
    # Step 3: Preprocess the data
    X, _ = preprocess_data(df)
    print(f"Processed data shape: {X.shape}")
    
    # Step 4: Check if the input shape matches what the model expects
    expected_shape = encoder.input_shape[1]
    actual_shape = X.shape[1]
    
    if expected_shape != actual_shape:
        print(f"Warning: Input shape mismatch. Model expects {expected_shape} features, but data has {actual_shape} features.")
        print("Adjusting input data to match model's expected shape...")
        
        if expected_shape < actual_shape:
            # If the data has more features than the model expects, truncate
            print(f"Truncating input data from {actual_shape} to {expected_shape} features")
            X = X[:, :expected_shape]
        else:
            # If the data has fewer features than the model expects, pad with zeros
            print(f"Padding input data from {actual_shape} to {expected_shape} features")
            padding = np.zeros((X.shape[0], expected_shape - actual_shape))
            X = np.hstack((X, padding))
        
        print(f"Adjusted data shape: {X.shape}")
    
    # Step 5: Process the data in batches
    X_reconstructed, latent_vectors = process_in_batches(X, encoder, decoder, args.batch_size)
    
    # Step 6: Detect anomalies
    predictions, anomaly_scores, reconstruction_errors = detect_anomalies(X, X_reconstructed, threshold)
    
    # Step 7: Add results to the dataframe
    df['anomaly'] = predictions
    df['anomaly_score'] = anomaly_scores
    df['reconstruction_error'] = reconstruction_errors
    
    # Step 8: Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Step 9: Print summary
    print("\nInference Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Detected anomalies: {predictions.sum()} ({predictions.sum()/len(df)*100:.2f}%)")
    
    # Step 10: Create visualizations
    if not args.no_vis:
        create_visualizations(df, predictions, anomaly_scores, reconstruction_errors, 
                             latent_vectors, threshold, vis_dir)

if __name__ == "__main__":
    main()