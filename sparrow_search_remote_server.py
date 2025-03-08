import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import random
from torch.cuda.amp import autocast, GradScaler
import logging
import warnings
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import argparse
import socket
import datetime
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set up logging to file and console
def setup_logging(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Configuration with added distributed training parameters
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

print(f"Using device: {CONFIG['device']}")

# Set random seeds for reproducibility
def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Added for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Added for performance

set_seeds(42)

# Add hostname logging
logger.info(f"Running on host: {socket.gethostname()}")

# Create necessary directories
for directory in [CONFIG['save_dir'], CONFIG['plot_dir'], CONFIG['model_dir'], CONFIG['checkpoint_dir']]:
    os.makedirs(directory, exist_ok=True)

def save_plot(fig, filename: str):
    """Save matplotlib figure with timestamp"""
    plot_path = os.path.join(CONFIG['plot_dir'], f"{CONFIG['timestamp']}_{filename}")
    fig.savefig(plot_path)
    logger.info(f"Plot saved: {plot_path}")
    plt.close(fig)

def save_model(model: nn.Module, model_name: str, metrics: Dict = None):
    """Save model and its metrics"""
    model_path = os.path.join(CONFIG['model_dir'], f"{CONFIG['timestamp']}_{model_name}.pt")
    model_info = {
        'state_dict': model.state_dict(),
        'config': CONFIG,
        'metrics': metrics,
        'timestamp': CONFIG['timestamp']
    }
    torch.save(model_info, model_path)
    logger.info(f"Model saved: {model_path}")

class WindSpeedDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length, :-1]
        y = self.data[idx + self.sequence_length - 1, -1]
        return x, y

def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Preprocess the data with both scaling and temporal features
    """
    # Convert Date/Time column to datetime
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    
    # Extract temporal features
    data['hour'] = data['Date/Time'].dt.hour
    data['minute'] = data['Date/Time'].dt.minute
    data['day'] = data['Date/Time'].dt.day
    data['month'] = data['Date/Time'].dt.month
    data['day_of_week'] = data['Date/Time'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Create cyclical features for time
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
    data['minute_sin'] = np.sin(2 * np.pi * data['minute']/60)
    data['minute_cos'] = np.cos(2 * np.pi * data['minute']/60)
    
    # Drop original Date/Time column
    data = data.drop(['Date/Time'], axis=1)
    
    # Convert all numeric columns to float64
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = data[numeric_columns].astype('float64')
    
    # Check for NaN values first
    if data.isna().any().any():
        logger.warning("Found NaN values in the data. Replacing with mean values.")
        data = data.fillna(data.mean())
    
    # Check for infinite values column by column
    has_inf = False
    for col in numeric_columns:
        mask = ~np.isfinite(data[col])
        if mask.any():
            has_inf = True
            mean_value = data[col][np.isfinite(data[col])].mean()
            data.loc[mask, col] = mean_value
    
    if has_inf:
        logger.warning("Found infinite values in the data. Replaced with column means.")
    
    # First apply MinMax scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Apply statistical normalization
    mean = np.mean(scaled_data, axis=0)
    std = np.std(scaled_data, axis=0)
    normalized_data = (scaled_data - mean) / (std + 1e-8)
    
    # Free memory
    del data
    if CONFIG['cuda_empty_cache']:
        torch.cuda.empty_cache()
    
    return normalized_data, scaler

def analyze_temporal_patterns(data: pd.DataFrame):
    """Analyze and visualize temporal patterns in the wind speed data"""
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    
    # Hourly patterns
    fig = plt.figure(figsize=(15, 5))
    hourly_avg = data.groupby(data['Date/Time'].dt.hour)['100m Wind Speed [m/s]'].mean()
    plt.plot(hourly_avg.index, hourly_avg.values)
    plt.title('Average Wind Speed by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Wind Speed (m/s)')
    save_plot(fig, 'hourly_patterns.png')
    
    # Daily patterns
    fig = plt.figure(figsize=(15, 5))
    daily_avg = data.groupby(data['Date/Time'].dt.day)['100m Wind Speed [m/s]'].mean()
    plt.plot(daily_avg.index, daily_avg.values)
    plt.title('Average Wind Speed by Day')
    plt.xlabel('Day of Month')
    plt.ylabel('Wind Speed (m/s)')
    save_plot(fig, 'daily_patterns.png')
    
    # Weekly patterns
    fig = plt.figure(figsize=(15, 5))
    weekly_avg = data.groupby(data['Date/Time'].dt.dayofweek)['100m Wind Speed [m/s]'].mean()
    plt.plot(weekly_avg.index, weekly_avg.values)
    plt.title('Average Wind Speed by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Wind Speed (m/s)')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    save_plot(fig, 'weekly_patterns.png')

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Optimized for GPU execution
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Explicitly set for clarity
        )
        
        self.hidden_bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Apply Xavier only to weight matrices
                nn.init.xavier_normal_(param)
            elif 'bias' in name:  # Bias initialization remains unchanged
                nn.init.constant_(param, 0.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add gradient clipping for stability
        for param in self.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
                
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Apply input batch normalization
        x_reshaped = x.view(-1, x.size(-1))
        x_bn = self.input_bn(x_reshaped)
        x = x_bn.view(batch_size, seq_len, -1)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Apply LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply batch normalization on the output features
        last_output = out[:, -1, :]
        normalized_output = self.hidden_bn(last_output)
        
        # Apply final linear layer
        out = self.fc(normalized_output)
        
        return out.squeeze()

class SparrowSearch:
    def __init__(self, n_particles: int, max_iter: int, param_bounds: Dict):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.param_bounds = param_bounds
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        self.device = CONFIG['device']  # Add device attribute
        
        # Initialize best_solution with middle values from param bounds
        self.best_solution = {}
        for param, (low, high) in self.param_bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                self.best_solution[param] = (low + high) // 2
            else:
                self.best_solution[param] = (low + high) / 2
                
    def initialize_population(self) -> List[Dict]:
        population = []
        for i in range(self.n_particles):
            particle = {}
            for param, (low, high) in self.param_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    particle[param] = random.randint(low, high)
                else:
                    particle[param] = random.uniform(low, high)
            population.append(particle)
        return population
        
    def update_position(self, particle: Dict, r2: float, alarm_value: float) -> Dict:
        new_particle = particle.copy()
        for param, (low, high) in self.param_bounds.items():
            if random.random() < alarm_value:
                # Random exploration with GPU tensor operations
                if isinstance(low, int) and isinstance(high, int):
                    new_value = torch.randint(low, high + 1, (1,), device=self.device)[0].item()
                else:
                    new_value = torch.rand(1, device=self.device)[0].item() * (high - low) + low
                new_particle[param] = new_value
            else:
                # Move towards best solution using GPU tensor operations
                current = torch.tensor([particle[param]], device=self.device)
                best = torch.tensor([self.best_solution[param]], device=self.device)
                step = r2 * (best - current)
                new_value = (current + step).item()
                if isinstance(low, int) and isinstance(high, int):
                    new_value = int(round(new_value))
                new_particle[param] = max(low, min(high, new_value))
        return new_particle
        
    def optimize(self, fitness_func) -> Tuple[Dict, float]:
        population = self.initialize_population()
        self.history = []
        
        # Move relevant computations to GPU
        population_tensor = {
            param: torch.tensor([p[param] for p in population], device=self.device)
            for param in self.param_bounds.keys()
        }
        
        # Add tqdm progress bar
        pbar = tqdm(range(self.max_iter), desc='Sparrow Search Progress')
        
        for iteration in pbar:
            alarm_value = 0.5 - (0.5 * iteration / self.max_iter)
            
            # Batch process fitness evaluations
            batch_fitness = []
            for i in range(self.n_particles):
                particle = {param: population_tensor[param][i].item() for param in self.param_bounds}
                fitness = fitness_func(particle)
                batch_fitness.append(fitness)
            
            # Convert to tensor for GPU operations
            fitness_tensor = torch.tensor(batch_fitness, device=self.device)
            
            # Update best solutions using GPU operations
            valid_mask = ~torch.isnan(fitness_tensor)
            if valid_mask.any():
                min_idx = torch.argmin(fitness_tensor[valid_mask])
                iteration_best_fitness = fitness_tensor[valid_mask][min_idx].item()
                
                if iteration_best_fitness < self.best_fitness:
                    self.best_fitness = iteration_best_fitness
                    for param in self.param_bounds:
                        self.best_solution[param] = population_tensor[param][valid_mask][min_idx].item()
                    pbar.set_postfix({'Best Fitness': f'{self.best_fitness:.6f}'})
            
                # Update population positions using GPU operations
                r2 = torch.rand(1, device=self.device).item()
                for i in range(self.n_particles):
                    particle = {param: population_tensor[param][i].item() for param in self.param_bounds}
                    new_particle = self.update_position(particle, r2, alarm_value)
                    for param in self.param_bounds:
                        population_tensor[param][i] = torch.tensor(new_particle[param], device=self.device)
                
                self.history.append(self.best_fitness)
                logger.info(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {self.best_fitness:.6f}")
            else:
                logger.warning(f"No valid fitness found in iteration {iteration + 1}. Skipping position updates.")
        
        # Plot and save optimization history
        if self.history:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(self.history)
            plt.title('Sparrow Search Optimization History')
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness')
            save_plot(fig, 'sparrow_search_history.png')
        
        # Clear GPU memory after optimization
        if CONFIG['cuda_empty_cache']:
            torch.cuda.empty_cache()
        
        return self.best_solution, self.best_fitness

def create_train_val_test_split(
    scaled_data: np.ndarray,
    sequence_length: int,
    batch_size: int,
    train_size: float = 0.7,
    val_size: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders with optimized settings for GPU"""
    n = len(scaled_data)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    train_data = scaled_data[:train_end]
    val_data = scaled_data[train_end:val_end]
    test_data = scaled_data[val_end:]
    
    train_dataset = WindSpeedDataset(train_data, sequence_length)
    val_dataset = WindSpeedDataset(val_data, sequence_length)
    test_dataset = WindSpeedDataset(test_data, sequence_length)
    
    # Optimized DataLoader settings for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler
) -> float:
    """Train for one epoch and return average loss"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Skip batch if loss is NaN
            if torch.isnan(loss):
                logger.warning("NaN loss detected during training. Skipping batch.")
                continue
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights with scaler
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
    
    # Return average loss
    if CONFIG['cuda_empty_cache']:
        torch.cuda.empty_cache()
    return total_loss / max(num_batches, 1)

def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Validate model and return average loss"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Skip batch if loss is NaN
            if torch.isnan(loss):
                logger.warning("NaN loss detected during validation. Skipping batch.")
                continue
                
            total_loss += loss.item()
            num_batches += 1
    
    # Return average loss
    return total_loss / max(num_batches, 1)

def save_checkpoint(epoch, model, optimizer, loss, filename):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': CONFIG
    }
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer):
    """Load training checkpoint with dynamic model architecture detection"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        
        # Extract model parameters from the checkpoint
        checkpoint_config = checkpoint.get('config', {})
        
        # Check if we need to recreate the model with correct architecture
        recreate_model = False
        current_state = model.state_dict()
        saved_state = checkpoint['model_state_dict']
        
        # Compare model architectures
        if any(param in saved_state and saved_state[param].shape != current_state[param].shape 
               for param in current_state):
            recreate_model = True
            logger.info("Detected model architecture mismatch. Recreating model...")
            
        if recreate_model:
            # Extract architecture parameters
            weight_ih_l0 = saved_state.get('lstm.weight_ih_l0')
            input_size = weight_ih_l0.shape[1] if weight_ih_l0 is not None else CONFIG['input_size']
            
            weight_hh_l0 = saved_state.get('lstm.weight_hh_l0')
            hidden_size = weight_hh_l0.shape[1] if weight_hh_l0 is not None else 60
            
            num_layers = 1
            while f'lstm.weight_ih_l{num_layers}' in saved_state:
                num_layers += 1
            
            dropout = checkpoint_config.get('dropout', 0.2)
            
            logger.info(f"Recreating model with: input_size={input_size}, hidden_size={hidden_size}, "
                       f"num_layers={num_layers}, dropout={dropout}")
            
            # Create new model and optimizer
            model = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(CONFIG['device'])
            
            # Create new optimizer with proper learning rate
            learning_rate = checkpoint_config.get('learning_rate', CONFIG['learning_rate'])
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Try to load optimizer state (with error handling)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {str(e)}. Using fresh optimizer.")
        
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Resumed from checkpoint: {filename}")
        return start_epoch, loss, model, optimizer
    return 0, float('inf'), model, optimizer

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    early_stopping_patience: int = 10
) -> Dict[str, List[float]]:
    """Train model with checkpointing and early stopping"""
    model = model.to(device)
    scaler = GradScaler()
    history = {'train_loss': [], 'val_loss': []}
    
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], f"{CONFIG['timestamp']}_checkpoint.pt")
    
    # Modified call to load_checkpoint that can return a new model and optimizer
    if CONFIG['resume_from_checkpoint']:
        start_epoch, best_val_loss, model, optimizer = load_checkpoint(checkpoint_path, model, optimizer)
        # Make sure the model is on the correct device after loading
        model = model.to(device)
    else:
        start_epoch, best_val_loss = 0, float('inf')
    
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Check for NaN loss
        if np.isnan(train_loss) or np.isnan(val_loss):
            logger.warning(f"NaN loss detected at epoch {epoch+1}. "
                          f"Train loss: {train_loss}, Val loss: {val_loss}")
            
            # Reset optimizer state and adjust learning rate
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=optimizer.param_groups[0]['lr'] * 0.5
            )
            continue
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save checkpoint periodically
        if (epoch + 1) % CONFIG['checkpoint_frequency'] == 0:
            save_checkpoint(epoch + 1, model, optimizer, val_loss, checkpoint_path)
    
    return history

def fitness_function(params, train_loader, val_loader):
    """Fitness function for Sparrow Search optimization"""
    logger.info(f"Evaluating parameters: {params}")
    
    try:
        # Initialize model with parameters
        num_layers = min(20, max(1, int(params['num_layers'])))
        
        model = LSTM(
            input_size=CONFIG['input_size'],
            hidden_size=int(params['hidden_size']),
            num_layers=num_layers,
            dropout=params['dropout']
        ).to(CONFIG['device'])
        
        # Initialize optimizer with the specific learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()
        
        # For hyperparameter search, DISABLE checkpoint loading
        # We want to evaluate each model fresh to accurately compare configurations
        original_resume_setting = CONFIG['resume_from_checkpoint']
        CONFIG['resume_from_checkpoint'] = False
        
        # Train for just a few epochs to evaluate performance
        history = train_model(
            model, train_loader, val_loader,
            criterion, optimizer,
            CONFIG['device'], epochs=5
        )
        
        # Restore original setting
        CONFIG['resume_from_checkpoint'] = original_resume_setting
        
        # Get final validation loss with safety check
        if not history['val_loss']:
            # If history is empty, just evaluate the model once
            final_loss = validate(model, val_loader, criterion, CONFIG['device'])
        else:
            final_loss = history['val_loss'][-1]
        
        logger.info(f"Evaluation complete. Loss: {final_loss:.6f}")
        return final_loss
        
    except Exception as e:
        logger.error(f"Error in fitness evaluation: {str(e)}")
        return 1e6

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    target_scaler: MinMaxScaler,
    target_idx: int,
    device: str,
    model_name: str
) -> Dict[str, float]:
    """Evaluate model on test set and return metrics"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    # Create dummy arrays for inverse transform
    pred_dummy = np.zeros((len(predictions), target_scaler.scale_.shape[0]))
    actual_dummy = np.zeros((len(actuals), target_scaler.scale_.shape[0]))
    
    # Set target column
    pred_dummy[:, target_idx] = np.array(predictions).flatten()
    actual_dummy[:, target_idx] = np.array(actuals).flatten()
    
    # Inverse transform
    pred_transformed = target_scaler.inverse_transform(pred_dummy)[:, target_idx]
    actual_transformed = target_scaler.inverse_transform(actual_dummy)[:, target_idx]
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(actual_transformed, pred_transformed),
        'mae': mean_absolute_error(actual_transformed, pred_transformed),
        'rmse': np.sqrt(mean_squared_error(actual_transformed, pred_transformed)),
        'r2': r2_score(actual_transformed, pred_transformed)
    }
    
    # Plot and save predictions vs actuals
    fig = plt.figure(figsize=(12, 6))
    plt.scatter(actual_transformed, pred_transformed, alpha=0.5)
    plt.plot([min(actual_transformed), max(actual_transformed)], 
             [min(actual_transformed), max(actual_transformed)], 'r--')
    plt.xlabel('Actual Wind Speed')
    plt.ylabel('Predicted Wind Speed')
    plt.title(f'Predictions vs Actuals - {model_name}')
    save_plot(fig, f'predictions_vs_actuals_{model_name}.png')
    
    return metrics

def run_wind_speed_modeling():
    """Main function to run the wind speed modeling pipeline"""
    try:
        # Update data loading path to use CONFIG['data_dir']
        data_path = os.path.join(CONFIG['data_dir'], 'ws_data.csv')
        logger.info(f"Loading data from: {data_path}")
        
        # Load data with error handling
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found at {data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
        print("\nData shape:", data.shape)
        
        # Plot input features distribution
        plt.figure(figsize=(15, 5))
        data.boxplot()
        plt.title('Distribution of Wind Speeds at Different Heights')
        plt.ylabel('Wind Speed (m/s)')
        plt.xticks(rotation=45)
        plt.show()
        
        # Check for NaN or infinite values
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        print("NaN values in data:", data.isna().sum().sum())
        print("Infinite values in data:", data[numeric_columns].apply(lambda x: np.isinf(x).sum()).sum())

        # Preprocess data
        scaled_data, scaler = preprocess_data(data)
        
        # Create train, validation, and test splits
        train_loader, val_loader, test_loader = create_train_val_test_split(
            scaled_data,
            CONFIG['sequence_length'],
            CONFIG['batch_size'],
            CONFIG['train_size'],
            CONFIG['val_size']
        )
        
        # Define parameter bounds for Sparrow Search - Updated with safer num_layers range
        param_bounds = {
            'hidden_size': (32, 128),
            'num_layers': (1, 20),   # Reduced upper bound to avoid state_dict errors
            'dropout': (0.1, 0.5),
            'learning_rate': (0.0005, 0.005)
        }
        
        # Create fitness function with fixed train/val loaders
        def search_fitness(params):
            return fitness_function(params, train_loader, val_loader)
        
        # Initialize and run Sparrow Search
        sparrow = SparrowSearch(n_particles=20, max_iter=10, param_bounds=param_bounds)
        best_params, best_fitness = sparrow.optimize(search_fitness)
        
        print("\nBest parameters found:", best_params)
        print("Best fitness:", best_fitness)
        
        # Train vanilla LSTM with safer num_layers value
        vanilla_lstm = LSTM(
            input_size=CONFIG['input_size'],
            hidden_size=64,
            num_layers=5,  # Reduced from original value for safety
            dropout=0.2
        )
        
        vanilla_optimizer = torch.optim.Adam(vanilla_lstm.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print("\nTraining vanilla LSTM...")
        vanilla_history = train_model(
            vanilla_lstm, train_loader, val_loader,
            criterion, vanilla_optimizer,
            CONFIG['device'], CONFIG['epochs']
        )
        
        # Train optimized LSTM with safety checks
        num_layers = min(20, max(1, int(best_params['num_layers'])))  # Ensure between 1 and 20
        
        optimized_lstm = LSTM(
            input_size=CONFIG['input_size'],
            hidden_size=int(best_params['hidden_size']),
            num_layers=num_layers,
            dropout=best_params['dropout']
        )
        
        optimized_optimizer = torch.optim.Adam(
            optimized_lstm.parameters(),
            lr=best_params['learning_rate']
        )
        
        print("\nTraining optimized LSTM...")
        optimized_history = train_model(
            optimized_lstm, train_loader, val_loader,
            criterion, optimized_optimizer,
            CONFIG['device'], CONFIG['epochs']
        )
        
        # Evaluate models on test set
        target_idx = -1  # Assuming the target is the last column
        
        print("\nVanilla LSTM Metrics:")
        vanilla_metrics = evaluate_model(
            vanilla_lstm, test_loader, scaler, target_idx, CONFIG['device'], 'vanilla_lstm'
        )
        for metric, value in vanilla_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nOptimized LSTM Metrics:")
        optimized_metrics = evaluate_model(
            optimized_lstm, test_loader, scaler, target_idx, CONFIG['device'], 'optimized_lstm'
        )
        for metric, value in optimized_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot and save training history comparison
        fig = plt.figure(figsize=(12, 6))
        plt.plot(vanilla_history['val_loss'], label='Vanilla LSTM', alpha=0.8)
        plt.plot(optimized_history['val_loss'], label='Optimized LSTM', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Training History Comparison')
        plt.legend()
        plt.grid(True)
        save_plot(fig, 'training_history_comparison.png')
        
        # Save vanilla LSTM model and metrics
        save_model(vanilla_lstm, 'vanilla_lstm', vanilla_metrics)
        
        # Save optimized LSTM model and metrics
        save_model(optimized_lstm, 'optimized_lstm', optimized_metrics)
        
        # Save experiment results to JSON
        results = {
            'vanilla_metrics': vanilla_metrics,
            'optimized_metrics': optimized_metrics,
            'best_params': best_params,
            'config': CONFIG,
            'timestamp': CONFIG['timestamp']
        }
        
        results_path = os.path.join(CONFIG['save_dir'], f"{CONFIG['timestamp']}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved: {results_path}")
        
        return results
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("GPU out of memory. Clearing cache and reducing batch size...")
            torch.cuda.empty_cache()
            CONFIG['batch_size'] //= 2
            return run_wind_speed_modeling()
        else:
            logger.error(f"Runtime error: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def log_gpu_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB")
            logger.info(f"Memory Cached: {torch.cuda.memory_reserved(i)/1e9:.2f} GB")

# Run the entire modeling pipeline
if __name__ == "__main__":
    log_gpu_stats()
    results = run_wind_speed_modeling()
    log_gpu_stats()  # Log final GPU state