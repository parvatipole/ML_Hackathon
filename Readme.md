# 🛡️ Zero-Day Exploit Detector

An advanced network security tool that leverages unsupervised machine learning to detect previously unknown (zero-day) network exploits. Built with Streamlit, this interactive application provides real-time anomaly detection, visualization, and analysis capabilities.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9+-orange.svg)

## ✨ Features

- **Anomaly Detection**: Identify unusual network patterns using unsupervised learning techniques
- **Interactive Dashboard**: Real-time monitoring of network traffic and threat levels
- **Advanced Visualization**: Explore network data through dynamic plots and latent space projections
- **Data Analysis**: Upload and analyze your own network data
- **AI Assistant**: Get insights about zero-day exploits and anomaly detection
- **Dark/Light Mode**: User-friendly interface with theme options

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/zero-day-exploit-detector.git
cd zero-day-exploit-detector

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📋 Requirements

```
streamlit>=1.15.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.9.0
plotly>=5.3.0
streamlit-chat>=0.0.2
pillow>=8.0.0
```

## 💻 Usage

1. **Dashboard Tab**: View current threat level, network traffic analysis, and anomaly timeline
2. **Data Analysis Tab**:
   - Choose between sample data or upload your own CSV
   - Adjust anomaly threshold parameters
   - Run analysis to detect anomalies
   - Explore results through interactive visualizations
3. **Model Chat Tab**:
   - Ask questions about zero-day exploits and anomaly detection
   - Load pre-trained or custom models

## 🧠 How It Works

The Zero-Day Exploit Detector uses a combination of:

1. **Variational Autoencoder (VAE)**: Learns to reconstruct normal network traffic patterns
2. **Anomaly Scoring**: Identifies data points with high reconstruction error
3. **Latent Space Analysis**: Projects high-dimensional data to 2D for visualization
4. **Threshold Optimization**: Automatically adjusts sensitivity based on data distribution

## 🤖 Simplified Detector

For demonstration purposes, this repository includes a `SimplifiedDetector` class that simulates the behavior of a full zero-day exploit detection system:

```python
class SimplifiedDetector:
    def __init__(self):
        self.model_loaded = False
        self.preprocessor = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def predict(self, data):
        """Simplified prediction function that returns anomaly scores."""
        # ... (see code for implementation details)
```


## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Happy Coding!
