# Experiment Results Summary
**Total Experiments**: 15
**Best RMSE**: 91.39
**Target RMSE**: 3000
**Target Achieved**: Yes

## Top 5 Models
| Rank | Model | Architecture | Val RMSE | Train Time | Converged |
|------|-------|--------------|----------|------------|-----------|
| 1 | lstm_lr0.001_bs16_dp0.1 | lstm | 91.39 | 383.7s | Yes |
| 2 | lstm_lr0.01_bs16_dp0.1 | lstm | 91.41 | 462.7s | Yes |
| 3 | lstm_lr0.01_bs32_dp0.2 | lstm | 91.43 | 274.7s | Yes |
| 4 | lstm_lr0.01_bs64_dp0.3 | lstm | 91.45 | 211.1s | Yes |
| 5 | stacked_lstm | stacked_lstm | 91.45 | 1157.2s | Yes |

## Experiment Log
- Experiment 1: baseline_rnn - Val RMSE: 91.90, Time: 148.2s
- Experiment 2: simple_lstm - Val RMSE: 91.68, Time: 290.6s
- Experiment 3: stacked_lstm - Val RMSE: 91.45, Time: 1157.2s
- Experiment 4: bidirectional_lstm - Val RMSE: 91.63, Time: 771.6s
- Experiment 5: gru_model - Val RMSE: 91.57, Time: 278.0s
- Experiment 6: lstm_lr0.01_bs16_dp0.1 - Val RMSE: 91.41, Time: 462.7s
- Experiment 7: lstm_lr0.01_bs16_dp0.2 - Val RMSE: 150.44, Time: 370.9s
- Experiment 8: lstm_lr0.01_bs16_dp0.3 - Val RMSE: 94.13, Time: 375.8s
- Experiment 9: lstm_lr0.01_bs32_dp0.1 - Val RMSE: 91.51, Time: 226.0s
- Experiment 10: lstm_lr0.01_bs32_dp0.2 - Val RMSE: 91.43, Time: 274.7s
- Experiment 11: lstm_lr0.01_bs32_dp0.3 - Val RMSE: 91.79, Time: 265.2s
- Experiment 12: lstm_lr0.01_bs64_dp0.1 - Val RMSE: 91.52, Time: 158.2s
- Experiment 13: lstm_lr0.01_bs64_dp0.2 - Val RMSE: 92.43, Time: 189.1s
- Experiment 14: lstm_lr0.01_bs64_dp0.3 - Val RMSE: 91.45, Time: 211.1s
- Experiment 15: lstm_lr0.001_bs16_dp0.1 - Val RMSE: 91.39, Time: 383.7s
