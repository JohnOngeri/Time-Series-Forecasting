# Experiment Results Summary
**Total Experiments**: 15
**Best RMSE**: 91.38
**Target RMSE**: 3000
**Target Achieved**: Yes

## Top 5 Models
| Rank | Model | Architecture | Val RMSE | Train Time | Converged |
|------|-------|--------------|----------|------------|-----------|
| 1 | lstm_lr0.01_bs64_dp0.2 | lstm | 91.38 | 191.6s | Yes |
| 2 | lstm_lr0.01_bs64_dp0.1 | lstm | 91.39 | 343.1s | Yes |
| 3 | stacked_lstm | stacked_lstm | 91.41 | 391.0s | Yes |
| 4 | lstm_lr0.01_bs64_dp0.3 | lstm | 91.43 | 163.6s | Yes |
| 5 | lstm_lr0.001_bs16_dp0.1 | lstm | 91.50 | 373.4s | Yes |

## Experiment Log
- Experiment 1: baseline_rnn - Val RMSE: 91.72, Time: 51.1s
- Experiment 2: simple_lstm - Val RMSE: 91.61, Time: 108.9s
- Experiment 3: stacked_lstm - Val RMSE: 91.41, Time: 391.0s
- Experiment 4: bidirectional_lstm - Val RMSE: 91.57, Time: 300.5s
- Experiment 5: gru_model - Val RMSE: 91.54, Time: 248.8s
- Experiment 6: lstm_lr0.01_bs16_dp0.1 - Val RMSE: 102.26, Time: 391.2s
- Experiment 7: lstm_lr0.01_bs16_dp0.2 - Val RMSE: 92.10, Time: 575.3s
- Experiment 8: lstm_lr0.01_bs16_dp0.3 - Val RMSE: 91.56, Time: 554.4s
- Experiment 9: lstm_lr0.01_bs32_dp0.1 - Val RMSE: 92.03, Time: 334.6s
- Experiment 10: lstm_lr0.01_bs32_dp0.2 - Val RMSE: 92.90, Time: 243.6s
- Experiment 11: lstm_lr0.01_bs32_dp0.3 - Val RMSE: 91.80, Time: 230.3s
- Experiment 12: lstm_lr0.01_bs64_dp0.1 - Val RMSE: 91.39, Time: 343.1s
- Experiment 13: lstm_lr0.01_bs64_dp0.2 - Val RMSE: 91.38, Time: 191.6s
- Experiment 14: lstm_lr0.01_bs64_dp0.3 - Val RMSE: 91.43, Time: 163.6s
- Experiment 15: lstm_lr0.001_bs16_dp0.1 - Val RMSE: 91.50, Time: 373.4s
