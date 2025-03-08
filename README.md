Vanilla LSTM Metrics:
2025-03-02 19:08:47,337 - INFO - Plot saved: ./results/plots/20250302_184828_predictions_vs_actuals_vanilla_lstm.png
mse: 4.0005
mae: 1.8857
rmse: 2.0001
r2: -0.0001

Optimized LSTM Metrics:
2025-03-02 19:08:51,548 - INFO - Plot saved: ./results/plots/20250302_184828_predictions_vs_actuals_optimized_lstm.png
mse: 0.0102
mae: 0.0939
rmse: 0.1011
r2: 0.9974


Configs Used:
CONFIG = {
    'input_size': 16,
    'sequence_length': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'num_workers': 4,  # For DataLoader
    'pin_memory': True,  # Faster data transfer to GPU
    'train_size': 0.7,
    'val_size': 0.15,
    'test_size': 0.15,
    'world_size': torch.cuda.device_count(),  # Number of available GPUs
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_dir': os.environ.get('LOG_DIR', './logs'),
    'data_dir': os.environ.get('DATA_DIR', './data'),
    'model_dir': os.environ.get('MODEL_DIR', './models'),
    'cuda_empty_cache': True,
    'gradient_accumulation_steps': 1,
    'amp_enabled': True,  # Automatic Mixed Precision
    'save_dir': os.environ.get('SAVE_DIR', './results'),
    'plot_dir': os.environ.get('PLOT_DIR', './results/plots'),
    'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    'checkpoint_dir': os.environ.get('CHECKPOINT_DIR', './checkpoints'),
    'checkpoint_frequency': 5,  # Save every 5 epochs
    'resume_from_checkpoint': True
}
