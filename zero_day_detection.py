import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Custom Sampling Layer
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        config = super().get_config()
        return config

# Custom VAE Layer
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(x - reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder,
            "decoder": self.decoder,
        })
        return config

class ZeroDayDetector:
    def __init__(self, latent_dim=16, intermediate_dim=64, batch_size=128, epochs=50):
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.preprocessor = None
        self.kmeans = None
        self.dbscan = None
        self.threshold = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """Preprocess the data for the VAE model"""
        print("Preprocessing data...")
        
        # Drop the id column as it's just an index
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Save the feature names for later use
        self.feature_names = df.columns.tolist()
        
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
        
        # If preprocessor already exists, use it to transform the data
        # Otherwise, create a new one and fit it
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            # Create preprocessing pipelines
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combine preprocessing steps
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            
            # Fit and transform the data
            X_processed = self.preprocessor.fit_transform(X)
        
        return X_processed, y
    
    def build_vae(self, input_dim):
        """Build the VAE model"""
        print("Building VAE model...")
        
        # Encoder
        encoder_inputs = Input(shape=(input_dim,))
        x = Dense(self.intermediate_dim, activation='relu')(encoder_inputs)
        x = Dropout(0.2)(x)
        x = Dense(self.intermediate_dim // 2, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        decoder_inputs = Input(shape=(self.latent_dim,))
        x = Dense(self.intermediate_dim // 2, activation='relu')(decoder_inputs)
        x = Dropout(0.2)(x)
        x = Dense(self.intermediate_dim, activation='relu')(x)
        decoder_outputs = Dense(input_dim, activation='sigmoid')(x)
        self.decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
        
        # VAE
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer='adam')
        
        return self.vae
    
    def train_vae(self, X_train, validation_data=None):
        """Train the VAE model"""
        print("Training VAE model...")
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.vae.fit(
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def get_reconstruction_error(self, X):
        """Calculate reconstruction error for each sample"""
        try:
            X_pred = self.vae.predict(X, verbose=0)
            mse = np.mean(np.square(X - X_pred), axis=1)
            return mse
        except Exception as e:
            print(f"Error in get_reconstruction_error: {e}")
            print(f"Input shape: {X.shape}")
            # Try to reshape the input to match the expected shape
            if hasattr(self.vae, 'input_shape') and self.vae.input_shape[1] != X.shape[1]:
                print(f"Attempting to reshape input to match VAE input shape...")
                min_features = min(self.vae.input_shape[1], X.shape[1])
                X_reshaped = X[:, :min_features]
                print(f"Reshaped input from {X.shape} to {X_reshaped.shape}")
                # Try again with reshaped input
                X_pred = self.vae.predict(X_reshaped, verbose=0)
                mse = np.mean(np.square(X_reshaped - X_pred), axis=1)
                return mse
            raise
    
    def get_latent_representation(self, X):
        """Get the latent space representation of the data"""
        try:
            # The encoder returns a tuple of [z_mean, z_log_var, z]
            # We need to extract just the z_mean from the tuple
            encoder_output = self.encoder.predict(X, verbose=0)
            # Check if encoder_output is a list/tuple and extract z_mean
            if isinstance(encoder_output, (list, tuple)):
                z_mean = encoder_output[0]  # Get the first element (z_mean)
            else:
                # If for some reason it's not a tuple, return as is
                z_mean = encoder_output
            return z_mean
        except Exception as e:
            print(f"Error in get_latent_representation: {e}")
            print(f"Input shape: {X.shape}")
            print(f"Expected input shape for encoder: {self.encoder.input_shape}")
            # Try to reshape the input to match the expected shape
            if hasattr(self.encoder, 'input_shape') and self.encoder.input_shape[1] != X.shape[1]:
                print(f"Attempting to reshape input to match encoder input shape...")
                min_features = min(self.encoder.input_shape[1], X.shape[1])
                X_reshaped = X[:, :min_features]
                print(f"Reshaped input from {X.shape} to {X_reshaped.shape}")
                # Try again with reshaped input
                encoder_output = self.encoder.predict(X_reshaped, verbose=0)
                if isinstance(encoder_output, (list, tuple)):
                    return encoder_output[0]
                else:
                    return encoder_output
            raise
    
    def apply_kmeans(self, X_latent, n_clusters=10):
        """Apply K-means clustering to the latent space"""
        print("Applying K-means clustering...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(X_latent)
        
        # Calculate silhouette score to evaluate clustering quality
        silhouette_avg = silhouette_score(X_latent, cluster_labels)
        print(f"Silhouette Score for K-means: {silhouette_avg:.4f}")
        
        return cluster_labels
    
    def apply_dbscan(self, X_latent, eps=0.5, min_samples=5):
        """Apply DBSCAN clustering to the latent space"""
        print("Applying DBSCAN clustering...")
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.dbscan.fit_predict(X_latent)
        
        # Count number of clusters and noise points
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")
        
        return cluster_labels
    
    def set_threshold(self, X_train, contamination=0.01):
        """Set threshold for anomaly detection based on reconstruction error"""
        print("Setting threshold for anomaly detection...")
        
        # Calculate reconstruction error
        mse = self.get_reconstruction_error(X_train)
        
        # Set threshold as percentile of reconstruction error
        self.threshold = np.percentile(mse, (1 - contamination) * 100)
        print(f"Threshold set to: {self.threshold:.6f}")
        
        return self.threshold
    
    def predict_anomalies(self, X):
        """Predict anomalies based on reconstruction error and threshold"""
        try:
            # Calculate reconstruction error
            mse = self.get_reconstruction_error(X)
            
            # Classify as anomaly if reconstruction error > threshold
            predictions = (mse > self.threshold).astype(int)
            
            # Calculate anomaly scores (normalized reconstruction error)
            anomaly_scores = mse / self.threshold
            
            return predictions, anomaly_scores, mse
        except Exception as e:
            print(f"Error in predict_anomalies: {e}")
            # If there's an error, try to reshape the input
            if hasattr(self.vae, 'input_shape') and self.vae.input_shape[1] != X.shape[1]:
                print(f"Attempting to reshape input to match VAE input shape...")
                min_features = min(self.vae.input_shape[1], X.shape[1])
                X_reshaped = X[:, :min_features]
                print(f"Reshaped input from {X.shape} to {X_reshaped.shape}")
                
                # Try again with reshaped input
                mse = self.get_reconstruction_error(X_reshaped)
                predictions = (mse > self.threshold).astype(int)
                anomaly_scores = mse / self.threshold
                return predictions, anomaly_scores, mse
            raise
    
    def evaluate(self, X, y_true):
        """Evaluate the model performance"""
        print("Evaluating model performance...")
        
        # Predict anomalies
        y_pred, anomaly_scores, _ = self.predict_anomalies(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Print confusion matrix
        print("Confusion Matrix:")
        print(cm)
        
        return y_pred, anomaly_scores
    
    def save_model(self, model_dir='models'):
        """Save the trained models"""
        print("Saving models...")
        
        # Create directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Register custom objects for model saving
        tf.keras.utils.get_custom_objects().update({
            'Sampling': Sampling,
            'VAE': VAE
        })
        
        # Save encoder and decoder models separately with proper file extensions
        self.encoder.save(os.path.join(model_dir, 'encoder_model.keras'))
        self.decoder.save(os.path.join(model_dir, 'decoder_model.keras'))
        
        # Save threshold
        np.save(os.path.join(model_dir, 'threshold.npy'), self.threshold)
        
        print(f"Models saved to {model_dir}")
    
    def load_model(self, model_dir='models'):
        """Load the trained models"""
        print("Loading models...")
        
        # Register custom objects for model loading
        tf.keras.utils.get_custom_objects().update({
            'Sampling': Sampling,
            'VAE': VAE
        })
        
        # Load encoder and decoder models
        self.encoder = tf.keras.models.load_model(os.path.join(model_dir, 'encoder_model.keras'))
        self.decoder = tf.keras.models.load_model(os.path.join(model_dir, 'decoder_model.keras'))
        
        # Recreate the VAE model
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer='adam')
        
        # Load threshold
        self.threshold = np.load(os.path.join(model_dir, 'threshold.npy'))
        
        print(f"Models loaded from {model_dir}")

def main():
    # Start timer
    start_time = time.time()
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Split data into normal and attack samples
    normal_df = df[df['label'] == 0]
    attack_df = df[df['label'] == 1]
    
    print(f"Normal samples: {len(normal_df)}")
    print(f"Attack samples: {len(attack_df)}")
    
    # Split normal data into train and test sets
    normal_train, normal_test = train_test_split(normal_df, test_size=0.2, random_state=42)
    
    # Create a balanced test set with normal and attack samples
    test_attack = attack_df.sample(n=len(normal_test), random_state=42)
    test_df = pd.concat([normal_test, test_attack])
    
    # Initialize the ZeroDayDetector
    detector = ZeroDayDetector(
        latent_dim=32,
        intermediate_dim=128,
        batch_size=256,
        epochs=30
    )
    
    # Preprocess the training data first to establish the preprocessor
    X_train, _ = detector.preprocess_data(normal_train)
    print(f"Training data shape: {X_train.shape}")
    
    # Then preprocess the test data using the same preprocessor
    X_test, y_test = detector.preprocess_data(test_df)
    print(f"Test data shape: {X_test.shape}")
    
    # Verify that training and test data have the same number of features
    if X_train.shape[1] != X_test.shape[1]:
        print(f"WARNING: Feature dimension mismatch! Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
        # Ensure same dimensions by truncating or padding
        min_features = min(X_train.shape[1], X_test.shape[1])
        X_train = X_train[:, :min_features]
        X_test = X_test[:, :min_features]
        print(f"Adjusted shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Build and train the VAE model
    detector.build_vae(input_dim=X_train.shape[1])
    history = detector.train_vae(X_train)
    
    # Set threshold for anomaly detection
    detector.set_threshold(X_train, contamination=0.01)
    
    # Get latent representations
    X_train_latent = detector.get_latent_representation(X_train)
    X_test_latent = detector.get_latent_representation(X_test)
    
    # Apply clustering to latent space
    kmeans_labels = detector.apply_kmeans(X_train_latent, n_clusters=10)
    dbscan_labels = detector.apply_dbscan(X_train_latent, eps=0.5, min_samples=10)
    
    # Evaluate the model
    y_pred, anomaly_scores = detector.evaluate(X_test, y_test)
    
    # Save the model
    detector.save_model()
    
    # Print execution time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Visualize results
    try:
        # Plot reconstruction error distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(detector.get_reconstruction_error(X_test), bins=50, kde=True)
        plt.axvline(detector.threshold, color='r', linestyle='--', label=f'Threshold: {detector.threshold:.6f}')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('reconstruction_error.png')
        
        # Plot latent space with TSNE
        from sklearn.manifold import TSNE
        
        # Apply t-SNE to reduce dimensionality to 2D for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_test_tsne = tsne.fit_transform(X_test_latent)
        
        # Plot t-SNE visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_test, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='True Label (0=Normal, 1=Attack)')
        plt.title('t-SNE Visualization of Latent Space')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig('tsne_visualization.png')
        
        # Plot anomaly scores
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=y_test, cmap='viridis', alpha=0.6)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
        plt.title('Anomaly Scores')
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.colorbar(label='True Label (0=Normal, 1=Attack)')
        plt.legend()
        plt.savefig('anomaly_scores.png')
        
        print("Visualizations saved as PNG files.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main() 