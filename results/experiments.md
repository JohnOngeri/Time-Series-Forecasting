# Experiment Results Summary
**Total Experiments**: 15
**Best RMSE**: 91.37
**Target RMSE**: 3000
**Target Achieved**: Yes

## Top 5 Models
| Rank | Model | Architecture | Val RMSE | Train Time | Converged |
|------|-------|--------------|----------|------------|-----------|
| 1 | stacked_lstm | stacked_lstm | 91.37 | 726.5s | Yes |
| 2 | lstm_lr0.001_bs16_dp0.1 | lstm | 91.38 | 336.4s | Yes |
| 3 | gru_model | gru | 91.42 | 602.1s | Yes |
| 4 | lstm_lr0.01_bs64_dp0.2 | lstm | 91.42 | 135.6s | Yes |
| 5 | bidirectional_lstm | bidirectional_lstm | 91.51 | 896.0s | Yes |

## Experiment Log
- Experiment 1: baseline_rnn - Val RMSE: 91.72, Time: 86.3s
- Experiment 2: simple_lstm - Val RMSE: 91.67, Time: 250.3s
- Experiment 3: stacked_lstm - Val RMSE: 91.37, Time: 726.5s
- Experiment 4: bidirectional_lstm - Val RMSE: 91.51, Time: 896.0s
- Experiment 5: gru_model - Val RMSE: 91.42, Time: 602.1s
- Experiment 6: lstm_lr0.01_bs16_dp0.1 - Val RMSE: 91.68, Time: 1336.2s
- Experiment 7: lstm_lr0.01_bs16_dp0.2 - Val RMSE: 91.96, Time: 1024.9s
- Experiment 8: lstm_lr0.01_bs16_dp0.3 - Val RMSE: 98.38, Time: 896.2s
- Experiment 9: lstm_lr0.01_bs32_dp0.1 - Val RMSE: 91.57, Time: 513.6s
- Experiment 10: lstm_lr0.01_bs32_dp0.2 - Val RMSE: 93.81, Time: 446.9s
- Experiment 11: lstm_lr0.01_bs32_dp0.3 - Val RMSE: 91.77, Time: 245.5s
- Experiment 12: lstm_lr0.01_bs64_dp0.1 - Val RMSE: 91.94, Time: 352.2s
- Experiment 13: lstm_lr0.01_bs64_dp0.2 - Val RMSE: 91.42, Time: 135.6s
- Experiment 14: lstm_lr0.01_bs64_dp0.3 - Val RMSE: 91.63, Time: 150.2s
- Experiment 15: lstm_lr0.001_bs16_dp0.1 - Val RMSE: 91.38, Time: 336.4s
