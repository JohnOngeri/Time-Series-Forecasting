# Air Quality Forecasting with RNN/LSTM Models

This project implements a comprehensive air quality forecasting system using various RNN/LSTM architectures to predict PM2.5 concentrations in Beijing.

## Project Structure

## 📁 Project Structure

```
air-quality-forecasting/
├── notebooks/
│   └── air_quality_forecasting_starter.ipynb      # Main training and evaluation notebook
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.h5                           # Saved model checkpoint
│   ├── submission.csv                              # Final prediction submission
│   └── submission_summary.md                       # Summary of the submission
├── results/
│   ├── experiments.csv                             # Logged experiments and metrics
│   ├── experiments.md                              # Notes and observations
│   └── project_summary.md                          # Overview of project progress
├── scripts/
│   ├── __pycache__/                                # Python cache files
│   ├── data_utils.py                               # Data loading and preprocessing
│   ├── features.py                                 # Feature engineering logic
│   ├── models.py                                   # Model architecture definitions
│   └── visualisation.py                            # Plotting and visualization utilities
├── visuals/                                        # Visual outputs (plots, graphs, etc.)
├── README.md                                       # Project description and setup guide
├── package.json                                    # Node.js project config (if applicable)
├── requirements.txt                                # Python dependencies
├── sample_submission.csv                           # Submission format for reference
├── test.csv                                        # Raw test dataset
├── test_data_clean.py                              # Script for cleaning test data
└── train.csv                                       # Raw training dataset
```


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
