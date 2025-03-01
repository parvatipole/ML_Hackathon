# Zero-Day Exploit Detection Using Anomaly Detection

This project implements an unsupervised deep learning pipeline for detecting zero-day exploits in network traffic data using the UNSW-NB15 dataset. The solution uses a Variational Autoencoder (VAE) combined with clustering techniques to identify anomalous network behavior that could indicate previously unknown vulnerabilities.

## Overview

Zero-day exploits represent a critical threat in cybersecurity as they involve previously unknown vulnerabilities that can bypass traditional defense mechanisms. This solution learns the normal operating patterns of a system and flags deviations that could be indicative of zero-day exploits.

## Key Components

1. **Unsupervised Learning for Normal Behavior Modeling**:
   - Variational Autoencoders (VAE) that learns a compact, latent representation of normal system behavior
   - Reconstruction Error Analysis to identify anomalies where system behavior deviates from the learned norm

2. **Anomaly Detection and Clustering**:
   - K-means and DBSCAN clustering on the latent space representations to group similar patterns and highlight outliers
   - Dynamic threshold setting based on statistical analysis of reconstruction errors

3. **Handling Noisy and High-Dimensional Data**:
   - Robust preprocessing methods to clean and normalize the system data
   - Dimensionality reduction through the VAE's encoder

## Dataset

The solution uses the UNSW-NB15 dataset, which is a comprehensive dataset designed for network intrusion detection research. It contains a mixture of benign and malicious network traffic that simulates a range of cyber attack scenarios.

## Implementation Details

The implementation consists of the following components:

- `zero_day_detection.py`: Main script that implements the VAE-based anomaly detection model
- `analyze_dataset.py`: Script to analyze and understand the dataset
- `inference.py`: Script to use the trained model for inference on new data
- `hyperparameter_tuning.py`: Script to tune the hyperparameters of the model
- `requirements.txt`: List of required Python packages

## How It Works

1. **Data Preprocessing**:
   - Categorical features are one-hot encoded
   - Numerical features are standardized
   - The model is trained only on normal (non-attack) data

2. **VAE Model**:
   - Learns to compress and reconstruct normal network traffic patterns
   - Creates a latent space representation of the data

3. **Anomaly Detection**:
   - Calculates reconstruction error for each sample
   - Sets a threshold based on the distribution of reconstruction errors in normal data
   - Flags samples with reconstruction errors above the threshold as potential zero-day exploits

4. **Clustering**:
   - Applies K-means and DBSCAN clustering to the latent space
   - Provides additional insights into the structure of the data and potential anomaly groups

## Results

The model generates several visualizations to help understand its performance:

- Reconstruction error distribution
- t-SNE visualization of the latent space
- Anomaly scores for test samples

## Requirements

To run this project, you need:

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
tensorflow==2.12.0
matplotlib==3.7.1
seaborn==0.12.2
```

## Usage

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the main script to train the model:
   ```
   python zero_day_detection.py
   ```

3. Use the trained model for inference on new data:
   ```
   python inference.py --input new_data.csv
   ```

4. Tune the hyperparameters of the model:
   ```
   python hyperparameter_tuning.py
   ```

## Hyperparameter Tuning

The `hyperparameter_tuning.py` script allows you to find the optimal hyperparameters for the model. It tests different combinations of:

- Latent dimension size
- Intermediate dimension size
- Batch size
- Number of epochs
- Contamination rate (for threshold setting)

The script evaluates each combination using precision, recall, F1 score, and AUC metrics, and saves the results to a CSV file.

## Inference

The `inference.py` script allows you to use the trained model to detect anomalies in new data. It:

1. Loads the trained model
2. Preprocesses the new data
3. Calculates reconstruction errors and anomaly scores
4. Flags potential zero-day exploits
5. Generates visualizations of the results

## Future Improvements

- Implement more sophisticated VAE architectures (e.g., convolutional VAEs)
- Explore other clustering algorithms for the latent space
- Add real-time detection capabilities
- Implement ensemble methods combining multiple anomaly detection techniques 