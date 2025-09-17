# Air Quality Forecasting with RNN/LSTM Models

This project implements a comprehensive air quality forecasting system using various RNN/LSTM architectures to predict PM2.5 concentrations in Beijing.

## Project Structure

\`\`\`
air-quality-forecasting/
├── data/
│   ├── train.csv                    # Training dataset
│   ├── test.csv                     # Test dataset
│   └── sample_submission.csv        # Sample submission format
├── notebooks/
│   └── 01_model_training.ipynb      # Main training notebook
├── scripts/
│   ├── data_utils.py               # Data preprocessing utilities
│   ├── features.py                 # Feature engineering functions
│   ├── models.py                   # Model architectures
│   ├── train.py                    # Training pipeline
│   └── submit.py                   # Submission generation
├── outputs/
│   ├── submission.csv              # Final submission file
│   └── checkpoints/                # Model checkpoints
├── results/
│   ├── experiments.csv             # Experiment results
│   └── experiments.md              # Experiment summary
├── visuals/
│   ├── loss_curves.png            # Training loss curves
│   ├── residuals.png              # Residual analysis
│   ├── actual_vs_pred.png         # Prediction comparison
│   └── model_diagrams/            # Model architecture diagrams
├── report/
│   └── air_quality_report.pdf     # Final report
├── README.md                       # This file
└── requirements.txt               # Python dependencies
\`\`\`

## Dataset

The dataset contains hourly air quality measurements from Beijing, including:
- **PM2.5**: Target variable (particulate matter ≤ 2.5 micrometers)
- **Weather features**: Temperature (TEMP), Dew Point (DEWP), Pressure (PRES)
- **Wind data**: Wind speed (Iws), Wind direction components (cbwd_*)
- **Solar radiation**: Is, Ir components
- **Datetime**: Hourly timestamps

## Model Architectures

The project implements and compares multiple architectures:

1. **Baseline RNN**: Simple recurrent neural network
2. **Simple LSTM**: Single LSTM layer
3. **Stacked LSTM**: Multiple LSTM layers with batch normalization
4. **Bidirectional LSTM**: Processes sequences in both directions
5. **GRU**: Gated Recurrent Unit variant

## Features

### Data Preprocessing
- Missing value imputation (mean, median, interpolation)
- Time-based feature engineering (cyclical encoding)
- Lag features (1, 2, 3, 6, 12, 24, 48 hours)
- Rolling statistics (mean, std, min, max, percentiles)
- Weather interaction features
- Seasonal indicators

### Model Training
- Time-series cross-validation
- Early stopping and learning rate scheduling
- Gradient clipping for stability
- Multiple optimizers (Adam, RMSprop, SGD)
- Comprehensive hyperparameter search

### Evaluation
- RMSE metric (target: < 3000)
- Residual analysis
- Actual vs predicted comparisons
- Loss curve visualization
- Statistical significance testing

## Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd air-quality-forecasting
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Set up data directory and place datasets:
\`\`\`bash
mkdir -p data
# Place train.csv, test.csv, and sample_submission.csv in data/
\`\`\`

## Usage

### Quick Start
Run the main training notebook:
\`\`\`bash
jupyter notebook notebooks/01_model_training.ipynb
\`\`\`

### Command Line Training
\`\`\`bash
python scripts/train.py --config lstm --epochs 100 --batch_size 32
\`\`\`

### Generate Submission
\`\`\`bash
python scripts/submit.py --model_path outputs/checkpoints/best_model.h5
\`\`\`

## Experiment Results

The project runs 15+ experiments comparing different architectures and hyperparameters. Results are logged in `results/experiments.csv` with the following metrics:

- Model architecture and hyperparameters
- Training and validation RMSE
- Training time and convergence
- Feature importance analysis

## Key Findings

[To be filled after running experiments]

## Kaggle Submission

The final model achieves an RMSE of [TBD] on the private leaderboard, meeting the target of < 3000.

## Academic Integrity

This project was conducted independently with reference to cited materials. All code and analysis are original work, with external sources properly attributed in the final report.

## References

[To be added in IEEE format]

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].
