## Model Performance Metrics

### 📌 Vanilla LSTM Metrics:
| Metric | Value |
|--------|-------|
| **MSE**  | 4.0005 |
| **MAE**  | 1.8857 |
| **RMSE** | 2.0001 |
| **R²**   | -0.0001 |

---

### 🚀 Optimized LSTM Metrics:
| Metric | Value |
|--------|-------|
| **MSE**  | 0.0102 |
| **MAE**  | 0.0939 |
| **RMSE** | 0.1011 |
| **R²**   | 0.9974 |

🔹 *Lower MSE, MAE, and RMSE indicate better performance, while an R² closer to 1 suggests a stronger fit.*


## Configuration Parameters

```python
CONFIG = {
    # Model & Training Parameters
    'input_size': 16,                 # Size of input features
    'sequence_length': 10,             # Length of input sequence
    'batch_size': 32,                  # Batch size for training
    'learning_rate': 0.001,            # Initial learning rate
    'epochs': 100,                     # Number of training epochs
    'gradient_accumulation_steps': 1,  # Steps for gradient accumulation
    'amp_enabled': True,               # Enable Automatic Mixed Precision (AMP)

    # Data Loading & Processing
    'num_workers': 4,                  # Number of worker threads for DataLoader
    'pin_memory': True,                 # Enable pinned memory for faster GPU transfer
    'train_size': 0.7,                  # Training dataset proportion
    'val_size': 0.15,                   # Validation dataset proportion
    'test_size': 0.15,                  # Test dataset proportion

    # Hardware Configuration
    'world_size': torch.cuda.device_count(),  # Number of available GPUs
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Select device
    'cuda_empty_cache': True,            # Clear CUDA cache after each iteration

    # Logging & Directories
    'log_dir': os.environ.get('LOG_DIR', './logs'),               # Log directory
    'data_dir': os.environ.get('DATA_DIR', './data'),             # Data directory
    'model_dir': os.environ.get('MODEL_DIR', './models'),         # Model directory
    'save_dir': os.environ.get('SAVE_DIR', './results'),          # Save directory
    'plot_dir': os.environ.get('PLOT_DIR', './results/plots'),    # Directory for plots
    'checkpoint_dir': os.environ.get('CHECKPOINT_DIR', './checkpoints'),  # Checkpoints directory

    # Checkpointing & Resumption
    'checkpoint_frequency': 5,          # Save a checkpoint every 5 epochs
    'resume_from_checkpoint': True,      # Resume training from the latest checkpoint

    # Timestamp
    'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # Timestamp for runs
}

