# Team: AI_Mavericks
# Member 1: Komal Sali
# Member 2: Parvati Pole
# Member 3: Shivani Bhat
# Member 4: Ibrahim


# Problem Statement: Advanced
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

Pre-requisites:

--Creating a Virtual Environment:
```
python -m venv .venv
```

--activating the virtual environment
```
.venv\scripts\activate
```


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
   python inference.py --input ./testing/UNSW_NB15_testing-set.csv
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

## Project Structure

### Core Files

- **zero_day_detection.py**: The main implementation of the VAE-based anomaly detection system. This file contains the `ZeroDayDetector` class that handles data preprocessing, model building, training, and evaluation. It implements the complete pipeline from raw data to anomaly detection results.

- **inference.py**: A script for applying the trained model to new data. This file loads a pre-trained model and uses it to detect anomalies in new network traffic data. It handles the entire inference process, including data preprocessing, anomaly detection, and result visualization.

- **hyperparameter_tuning.py**: A script for finding the optimal hyperparameters for the model. It systematically tests different combinations of hyperparameters (latent dimension, intermediate dimension, batch size, epochs, contamination rate) and evaluates their performance using multiple metrics.

- **analyze_dataset.py**: A utility script for exploring and understanding the UNSW-NB15 dataset. It provides statistical analysis and visualizations of the data distribution, feature correlations, and class balance.

### Directories

- **datasets/**: Contains the datasets used for training and testing the model:
  - `UNSW_NB15_training-set.csv`: The training dataset with labeled normal and attack traffic
  - `UNSW_NB15_testing-set.csv`: The testing dataset for evaluating the model's performance

- **models/**: Contains the saved trained models:
  - `encoder_model.keras`: The trained encoder part of the VAE
  - `decoder_model.keras`: The trained decoder part of the VAE
  - `threshold.npy`: The saved anomaly threshold value

- **Visuals/**: Contains all visualization outputs from the model:
  - `reconstruction_error.png`: Distribution of reconstruction errors with the anomaly threshold
  - `tsne_visualization.png`: t-SNE visualization of the latent space, colored by true labels
  - `anomaly_scores.png`: Anomaly scores for each sample in the test set
  - `inference_reconstruction_error.png`: Reconstruction error distribution from inference
  - `inference_anomaly_scores.png`: Anomaly scores from inference
  - `inference_tsne_visualization.png`: t-SNE visualization of the latent space from inference
  - `tuning_latent_dim.png`: F1 scores for different latent dimensions
  - `tuning_contamination.png`: F1 scores for different contamination rates
  - `tuning_training_time.png`: Training time vs model complexity
  - `threshold_tuning.png`: Analysis of different threshold values
  - `roc_curve.png`: ROC curve showing model performance
  - `reconstruction_error_distribution.png`: Detailed view of reconstruction error distribution

- **testing/**: Contains files from the testing phase (not to be considered as the final testing folder). This directory includes intermediate results and experimental code.

## Theoretical Background

### Variational Autoencoders (VAEs)

Variational Autoencoders are a type of generative model that combines elements of autoencoders with variational inference. Unlike traditional autoencoders, VAEs don't just learn to compress and reconstruct data; they learn a probabilistic mapping to a latent space that follows a predefined distribution (typically Gaussian).

**Why VAEs for Zero-Day Detection?**

1. **Unsupervised Learning**: VAEs can learn from unlabeled data, which is crucial for zero-day detection where we don't have examples of the attacks we're trying to detect.

2. **Probabilistic Modeling**: By modeling the distribution of normal behavior, VAEs can quantify the "normality" of new samples, making them ideal for anomaly detection.

3. **Regularized Latent Space**: The VAE's regularization (through the KL divergence term) creates a smooth, continuous latent space that captures the underlying structure of normal data.

4. **Reconstruction-Based Detection**: The reconstruction error provides a natural anomaly score - samples that the model struggles to reconstruct are likely to be anomalous.

In our implementation, the VAE consists of:
- An encoder network that maps input data to a latent distribution
- A sampling layer that implements the "reparameterization trick" for backpropagation
- A decoder network that reconstructs the input from the latent representation

The model is trained to minimize both the reconstruction error and the KL divergence between the learned latent distribution and a standard normal distribution.

### Clustering Techniques

We employ two clustering algorithms to analyze the latent space representations:

1. **K-means Clustering**: A centroid-based algorithm that partitions the data into K clusters by minimizing the within-cluster variance. In our context, K-means helps identify natural groupings in the latent space that might correspond to different types of normal behavior.

2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: A density-based clustering algorithm that groups together points that are closely packed, while marking points in low-density regions as outliers. DBSCAN is particularly useful for anomaly detection as it can identify outliers (potential zero-day exploits) as noise points.

**Why Clustering for Zero-Day Detection?**

1. **Structure Discovery**: Clustering helps uncover the natural structure in the latent space, revealing patterns that might not be apparent in the original high-dimensional data.

2. **Anomaly Identification**: Points that don't belong to any cluster (DBSCAN) or are far from their cluster centroid (K-means) are likely to be anomalous.

3. **Attack Type Grouping**: Clustering can potentially group similar attack types together, even if they weren't seen during training.

4. **Dimensionality Reduction**: By working in the lower-dimensional latent space, clustering algorithms can be more effective than in the original high-dimensional feature space.

The combination of VAEs and clustering provides a powerful framework for zero-day exploit detection:
- The VAE learns a compact representation of normal behavior
- Reconstruction error provides a primary anomaly signal
- Clustering in the latent space provides additional insights into the structure of the data and potential anomaly groups

This approach allows us to detect novel attacks without prior knowledge of their specific characteristics, making it ideal for zero-day exploit detection. 