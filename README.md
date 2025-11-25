# Insider Threat Detection System

**Unsupervised Behavioral Profiling Using Time-Series and Anomaly Detection Techniques**

A comprehensive machine learning system for detecting insider threats through behavioral analysis using three complementary unsupervised models: Isolation Forest, LSTM Autoencoder, and Deep Clustering.

## 🎯 Project Overview

This system implements state-of-the-art unsupervised learning techniques to detect malicious insider activity within enterprise environments by:

- Analyzing user behavior patterns across multiple data sources (logon, file access, email, device usage, HTTP)
- Detecting anomalies without requiring labeled training data
- Combining multiple detection models in an ensemble for improved accuracy
- Maintaining privacy through built-in data anonymization

## 📊 System Architecture

```
Raw Logs → Preprocessing → Feature Engineering → Model Training → Ensemble → Alerts
           (Stage 1)      (Stage 2)             (Stage 3)         (Stage 4-5) (Stage 6)
```

### Models Implemented

1. **Isolation Forest**: Efficient screening for point anomalies in high-dimensional spaces
2. **LSTM Autoencoder**: Sequential anomaly detection for temporal behavior patterns
3. **Deep Clustering**: Behavioral profiling through joint feature learning and clustering

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd insider-threat-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download the CMU-CERT Insider Threat Dataset from: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
2. Extract the dataset files to `data/raw/` directory
3. Ensure the following files are present:
   - `logon.csv`
   - `device.csv`
   - `file.csv`
   - `email.csv`
   - `http.csv`

### Running the Pipeline

**Option 1: Run Complete Pipeline**
```bash
python main.py --full
```

**Option 2: Run Specific Stages**
```bash
# Run only preprocessing and feature engineering
python main.py --stages 1 2

# Skip already completed stages
python main.py --full --skip 1 2
```

**Option 3: Run Individual Stages**
```bash
python main.py --preprocess      # Stage 1
python main.py --feature-eng     # Stage 2
python main.py --train           # Stage 3
python main.py --evaluate        # Stage 4
python main.py --ensemble        # Stage 5
python main.py --visualize       # Stage 6
```

**Option 4: Run Individual Scripts**
```bash
python data_preprocessing.py
python feature_engineering.py
python isolation_forest_model.py
python lstm_autoencoder_model.py
python deep_clustering_model.py
python model_evaluation.py
python ensemble_system.py
python visualization.py
```

## 📁 Project Structure

```
insider-threat-detection/
├── config.py                      # Configuration and hyperparameters
├── utils.py                       # Utility functions
├── main.py                        # Main pipeline orchestrator
├── data_preprocessing.py          # Data loading and cleaning
├── feature_engineering.py         # Feature creation
├── isolation_forest_model.py      # Isolation Forest implementation
├── lstm_autoencoder_model.py      # LSTM Autoencoder implementation
├── deep_clustering_model.py       # Deep Clustering implementation
├── model_evaluation.py            # Model comparison and evaluation
├── ensemble_system.py             # Ensemble integration
├── visualization.py               # Results visualization
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   ├── raw/                       # Raw CMU-CERT dataset files
│   └── processed/                 # Processed data and features
│
├── models/                        # Trained model files
├── results/                       # Metrics, predictions, visualizations
└── logs/                          # Execution logs
```

## ⚙️ Configuration

All configuration is centralized in `config.py`. Key parameters include:

### Data Settings
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: Data split ratios
- `PSEUDONYMIZE_USERS`, `PSEUDONYMIZE_HOSTS`: Privacy settings

### Model Parameters
- **Isolation Forest**: `n_estimators`, `contamination`, etc.
- **LSTM Autoencoder**: `lstm_units`, `encoding_dim`, `epochs`, etc.
- **Deep Clustering**: `n_clusters`, `encoding_dims`, `learning_rate`, etc.

### Ensemble Settings
- `weights`: Model weights for weighted ensemble
- `voting_method`: 'weighted', 'majority', or 'cascade'

## 📈 Expected Results

Based on CMU-CERT dataset evaluation:

| Model              | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------------------|----------|-----------|--------|----------|---------|
| Isolation Forest   | ~92%     | ~85%      | ~84%   | ~84%     | ~0.91   |
| Deep Clustering    | ~97%     | ~94%      | ~97%   | ~96%     | ~0.98   |
| LSTM Autoencoder   | ~98%     | ~97%      | ~97%   | ~97%     | ~0.99   |
| Ensemble (Weighted)| ~98%+    | ~97%+     | ~98%+  | ~98%+    | ~0.99   |

## 🔍 Output Files

### Models
- `models/isolation_forest.pkl`
- `models/lstm_autoencoder.h5`
- `models/deep_clustering.h5`

### Results
- `results/evaluation_report.txt`: Comprehensive evaluation report
- `results/generated_alerts.csv`: Final alerts with severity levels
- `results/*_predictions.csv`: Individual model predictions
- `results/*_metrics.csv`: Performance metrics

### Visualizations
- `results/visualizations/roc_comparison.png`
- `results/visualizations/metrics_comparison_bar.png`
- `results/visualizations/alert_severity_distribution.png`
- `results/visualizations/executive_summary.png`

## 🛡️ Privacy & Ethics

This system implements privacy-by-design principles:

- **Data Anonymization**: All user identifiers are pseudonymized using SHA-256 hashing
- **Minimal Data Collection**: Only metadata is processed, not content
- **Transparency**: Clear documentation of monitoring practices
- **Purpose Limitation**: Data used solely for security threat detection

## 🔬 Research Background

This implementation is based on the thesis:
**"Unsupervised Behavioural Profiling for Insider Threat Detection Using Time-Series and Anomaly Detection Techniques"**

Key research findings:
- Deep learning models significantly outperform traditional methods
- Ensemble approaches reduce false positives while maintaining high detection rates
- Privacy-preserving techniques (anonymization) can improve model performance
- Temporal sequence modeling is crucial for detecting complex threats

## 📚 References

1. Jiang et al. (2023) - Deep Clustering Networks for Insider Threat Detection
2. Kim & Lee (2024) - LSTM Autoencoder for Privacy-Preserving Anomaly Detection
3. CMU-CERT Insider Threat Dataset - Carnegie Mellon University
4. Le & Zincir-Heywood (2021) - Ensemble Methods for Temporal Feature Analysis

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional anomaly detection algorithms
- Real-time streaming data processing
- Extended feature engineering techniques
- Enhanced explainability methods

## 📄 License

This project is for academic research purposes. Please cite appropriately if used in publications.

## 👤 Author

**Bipin Rimal**  
Module: Computing Individual Research Project (STW7048CEM)

## 🆘 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: No data files found`
- **Solution**: Ensure CMU-CERT dataset is in `data/raw/` directory

**Issue**: `MemoryError during training`
- **Solution**: Reduce `batch_size` in `config.py` or process data in chunks

**Issue**: `TensorFlow compatibility errors`
- **Solution**: Ensure TensorFlow >= 2.15.0: `pip install --upgrade tensorflow`

**Issue**: Poor model performance
- **Solution**: Check data quality, adjust hyperparameters in `config.py`, ensure sufficient training data

### Getting Help

1. Check logs in `logs/insider_threat_detection.log`
2. Enable verbose mode in config: `LOGGING['level'] = 'DEBUG'`
3. Run stages individually to isolate issues

## 📞 Contact

For questions or issues, please contact through the project repository or academic institution.

---

**Note**: This system is designed for research and educational purposes. Deployment in production environments should include additional security hardening, compliance reviews, and stakeholder approval.# python_files_thesis
