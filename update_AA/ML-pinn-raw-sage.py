import dask.delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
import numpy as np
from obspy import UTCDateTime
from obspy import read
from obspy.clients.fdsn.client import Client
from obspy.core.inventory.inventory import Inventory
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.signal.filter import bandpass
import seisbench.models as sbm
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import dask

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ PHYSICS-INFORMED FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════════════

class PhysicsInformedFeatures:
    """
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ Physics-Informed Feature Extractor                                                  │
    │ ─────────────────────────────────────────────────────────────────────────────────── │
    │ • Computes traditional seismological features that help with phase identification   │
    │ • STA/LTA ratios at multiple time scales                                            │
    │ • Frequency domain features                                                         │
    │ • Envelope and instantaneous phase                                                  │
    │ • These features provide domain knowledge to guide neural network learning          │
    └─────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
        # STA/LTA parameters for different time scales
        self.sta_lta_configs = [
            {'sta': 0.5, 'lta': 10.0},   # Fast detection
            {'sta': 1.0, 'lta': 20.0},   # Medium scale
            {'sta': 2.0, 'lta': 30.0},   # Slow, stable detection
        ]
        
        # Frequency bands for spectral analysis
        self.freq_bands = [
            {'name': 'low', 'freqmin': 1.0, 'freqmax': 5.0},      # Low frequency
            {'name': 'mid', 'freqmin': 5.0, 'freqmax': 15.0},     # Mid frequency
            {'name': 'high', 'freqmin': 15.0, 'freqmax': 45.0},   # High frequency
        ]
    
    def compute_sta_lta_features(self, waveform):
        """Compute STA/LTA ratios at multiple time scales"""
        features = []
        
        for config in self.sta_lta_configs:
            sta_samples = int(config['sta'] * self.sampling_rate)
            lta_samples = int(config['lta'] * self.sampling_rate)
            
            # Ensure we have enough samples
            if len(waveform) < lta_samples:
                print(f"Warning: Waveform too short ({len(waveform)}) for LTA ({lta_samples})")
                # Create zeros as fallback
                features.extend([np.zeros_like(waveform), np.zeros_like(waveform)])
                continue
            
            try:
                # Compute classic STA/LTA
                sta_lta = classic_sta_lta(waveform, sta_samples, lta_samples)
                features.append(sta_lta)
                
                # Also compute log of STA/LTA for better dynamic range
                log_sta_lta = np.log10(np.maximum(sta_lta, 1e-10))
                features.append(log_sta_lta)
            except Exception as e:
                print(f"Warning: STA/LTA computation failed: {e}")
                # Add zeros as fallback
                features.extend([np.zeros_like(waveform), np.zeros_like(waveform)])
        
        return np.array(features)
    
    def compute_envelope_features(self, waveform):
        """Compute envelope and instantaneous features"""
        try:
            # Analytic signal for envelope and instantaneous phase
            analytic_signal = signal.hilbert(waveform)
            envelope = np.abs(analytic_signal)
            instantaneous_phase = np.angle(analytic_signal)
            
            # Envelope derivative (rate of change)
            envelope_derivative = np.gradient(envelope)
            
            # Instantaneous frequency
            instantaneous_freq = np.gradient(np.unwrap(instantaneous_phase)) / (2.0 * np.pi) * self.sampling_rate
            
            return np.array([
                envelope,
                envelope_derivative,
                instantaneous_freq
            ])
        except Exception as e:
            print(f"Warning: Envelope computation failed: {e}")
            # Return zeros as fallback
            return np.array([
                np.zeros_like(waveform),
                np.zeros_like(waveform), 
                np.zeros_like(waveform)
            ])
    
    def compute_spectral_features(self, waveform):
        """Compute frequency domain features"""
        features = []
        
        for band in self.freq_bands:
            # Bandpass filter
            try:
                filtered = bandpass(waveform, band['freqmin'], band['freqmax'], 
                                  self.sampling_rate, corners=2, zerophase=True)
                
                # Energy in this band
                energy = filtered ** 2
                features.append(energy)
                
                # Envelope of filtered signal
                envelope = np.abs(signal.hilbert(filtered))
                features.append(envelope)
                
            except Exception as e:
                print(f"Warning: Could not compute {band['name']} band features: {e}")
                # Add zeros as fallback
                features.extend([np.zeros_like(waveform), np.zeros_like(waveform)])
        
        return np.array(features)
    
    def compute_all_features(self, waveform):
        """
        Compute all physics-informed features
        
        Args:
            waveform: 1D numpy array of seismic data
            
        Returns:
            features: 2D numpy array of shape (n_features, n_samples)
        """
        # Normalize waveform to prevent numerical issues
        waveform_norm = waveform / (np.std(waveform) + 1e-10)
        
        # Compute different feature types
        sta_lta_features = self.compute_sta_lta_features(waveform_norm)
        envelope_features = self.compute_envelope_features(waveform_norm)
        spectral_features = self.compute_spectral_features(waveform_norm)
        
        # Combine all features
        all_features = np.vstack([
            waveform_norm.reshape(1, -1),  # Original waveform
            sta_lta_features,              # STA/LTA features
            envelope_features,             # Envelope features  
            spectral_features              # Spectral features
        ])
        
        return all_features

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ ENHANCED RAW WAVEFORM PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════════════

class RawWaveformProcessor:
    """
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ Enhanced Raw Waveform Processor                                                     │
    │ ─────────────────────────────────────────────────────────────────────────────────── │
    │ • Creates dual-channel input from single waveform                                   │
    │ • Channel 1: Negative values → Positive values (absolute value of negatives)       │
    │ • Channel 2: Positive values → Negative values (inverted positives)                │
    │ • Channel 3: Original waveform (optional)                                           │
    └─────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, include_original=True):
        """
        Args:
            include_original (bool): Whether to include original waveform as third channel
        """
        self.include_original = include_original
    
    def process_waveform(self, waveform):
        """
        Process raw waveform into multiple channels
        
        Args:
            waveform: 1D numpy array of seismic data
            
        Returns:
            processed_waveform: 2D numpy array of shape (n_channels, n_samples)
        """
        # Normalize waveform to prevent numerical issues
        waveform_norm = waveform / (np.std(waveform) + 1e-10)
        
        # Channel 1: Convert negative values to positive (absolute value of negatives)
        channel_1 = waveform_norm.copy()
        negative_mask = waveform_norm < 0
        channel_1[negative_mask] = np.abs(waveform_norm[negative_mask])  # Make negatives positive
        channel_1[~negative_mask] = 0  # Set positive values to zero
        
        # Channel 2: Convert positive values to negative (invert positives)
        channel_2 = waveform_norm.copy()
        positive_mask = waveform_norm > 0
        channel_2[positive_mask] = -waveform_norm[positive_mask]  # Make positives negative
        channel_2[~positive_mask] = 0  # Set negative values to zero
        
        if self.include_original:
            # Channel 3: Original waveform
            channels = np.array([
                channel_1,           # Negatives → Positives
                channel_2,           # Positives → Negatives  
                waveform_norm = np.pad(waveform_data, (pad_start, pad_end), 
                                    mode='constant', constant_values=0)
                pick_sample_norm = pick_sample + pad_start
            
            # Prepare features based on configuration
            if use_physics_features:
                # Extract physics-informed features
                try:
                    # Create feature extractor for this task
                    feature_extractor = PhysicsInformedFeatures(sampling_rate=sampling_rate)
                    physics_features = feature_extractor.compute_all_features(waveform_norm)
                    
                    print(f"Physics features shape: {physics_features.shape} for station {item['station']}")
                    
                except Exception as e:
                    print(f"Warning: Could not compute physics features for {item['station']}: {e}")
                    # Fallback to just raw waveform
                    physics_features = waveform_norm.reshape(1, -1)
                
                final_features = physics_features
            else:
                # Use enhanced raw waveform processing (dual-channel)
                try:
                    # Create waveform processor for this task
                    waveform_processor = RawWaveformProcessor(include_original=True)
                    processed_channels = waveform_processor.process_waveform(waveform_norm)
                    
                    print(f"Enhanced raw waveform shape: {processed_channels.shape} for station {item['station']}")
                    
                except Exception as e:
                    print(f"Warning: Could not process enhanced raw waveform for {item['station']}: {e}")
                    # Fallback to single channel
                    processed_channels = waveform_norm.reshape(1, -1)
                
                final_features = processed_channels
            
            # Ensure features have correct length
            if final_features.shape[1] != target_length:
                print(f"Warning: Feature length mismatch: {final_features.shape[1]} vs {target_length}")
                # Resize each feature channel to target length
                resized_features = []
                for i in range(final_features.shape[0]):
                    if final_features.shape[1] > target_length:
                        # Truncate
                        excess = final_features.shape[1] - target_length
                        start_trim = excess // 2
                        feature_trimmed = final_features[i][start_trim:start_trim + target_length]
                        resized_features.append(feature_trimmed)
                    elif final_features.shape[1] < target_length:
                        # Pad
                        pad_needed = target_length - final_features.shape[1]
                        pad_start = pad_needed // 2
                        pad_end = pad_needed - pad_start
                        feature_padded = np.pad(final_features[i], (pad_start, pad_end), 
                                            mode='constant', constant_values=0)
                        resized_features.append(feature_padded)
                    else:
                        resized_features.append(final_features[i])
                
                final_features = np.array(resized_features)
            
            # Create labels
            label = np.zeros(target_length)
            
            # Window around pick
            window_samples = int(window_size * sampling_rate / 2)
            pick_sample_norm = np.clip(pick_sample_norm, 0, target_length - 1)
            
            start_idx = max(0, pick_sample_norm - window_samples)
            end_idx = min(target_length, pick_sample_norm + window_samples)
            label[start_idx:end_idx] = 1
            
            return {
                'features': final_features.astype(np.float32),
                'label': label.astype(np.int64)
            }
            
        except Exception as e:
            print(f"Error processing waveform: {e}")
            return None
        
    def __len__(self):
        return len(self.data) if hasattr(self, 'data') else 0
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor(self.labels[idx])
        return features, label

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the adaptive model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss function with class weighting (background vs P-wave)
    class_weights = torch.FloatTensor([1.0, 3.0]).to(device)  # Higher weight for P-wave
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, 
                                                       desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_adaptive_model.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses

def evaluate_picks(model, val_dataset, val_manifest, threshold=0.5):
    """Evaluate pick accuracy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    pick_errors = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            features, true_label = val_dataset[i]
            features = features.unsqueeze(0).to(device)
            
            output = model(features)
            prob = output[0, 1, :].cpu().numpy()  # P-wave probability
            
            # Find predicted pick (maximum probability)
            pred_pick_sample = np.argmax(prob)
            
            # Find true pick (center of labeled window)
            true_pick_samples = np.where(true_label == 1)[0]
            if len(true_pick_samples) > 0:
                true_pick_sample = np.mean(true_pick_samples)
                
                # Calculate error in samples
                error_samples = abs(pred_pick_sample - true_pick_sample)
                pick_errors.append(error_samples)
    
    return np.array(pick_errors)

def train_test_split(data, test_size=0.2, random_state=None):
    """Simple train/test split function"""
    if random_state:
        np.random.seed(random_state)
    
    data_copy = data.copy()
    np.random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * (1 - test_size))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    
    return train_data, test_data

def visualize_features(dataset, sample_idx=0, save_path='features.png', use_physics_features=True):
    """Visualize the features for a sample (physics or enhanced raw)"""
    if sample_idx >= len(dataset):
        print(f"Sample index {sample_idx} out of range")
        return
    
    features, label = dataset[sample_idx]
    features_np = features.numpy()
    label_np = label.numpy()
    
    # Feature names based on mode
    if use_physics_features:
        feature_names = [
            'Raw Waveform',
            'STA/LTA (0.5/10s)', 'Log STA/LTA (0.5/10s)',
            'STA/LTA (1/20s)', 'Log STA/LTA (1/20s)', 
            'STA/LTA (2/30s)', 'Log STA/LTA (2/30s)',
            'Envelope', 'Envelope Derivative', 'Instantaneous Frequency',
            'Low Freq Energy', 'Low Freq Envelope',
            'Mid Freq Energy', 'Mid Freq Envelope', 
            'High Freq Energy', 'High Freq Envelope'
        ]
        title_suffix = "Physics-Informed Features"
    else:
        feature_names = [
            'Channel 1: Negatives → Positives',
            'Channel 2: Positives → Negatives', 
            'Channel 3: Original Waveform'
        ]
        title_suffix = "Enhanced Raw Waveform Channels"
    
    # Truncate if we have fewer features than expected
    n_features = min(len(feature_names), features_np.shape[0])
    
    # Create time vector
    n_samples = features_np.shape[1]
    time_vector = np.linspace(-3, 120, n_samples)  # Assuming 3s pre, 120s post
    
    # Create subplot grid
    if n_features <= 3:
        # Single row for small number of features
        fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 4))
        if n_features == 1:
            axes = [axes]
        n_rows, n_cols = 1, n_features
    else:
        # Multiple plots for physics features
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # Different colors for different channels in raw waveform mode
        if not use_physics_features:
            colors = ['red', 'blue', 'black']
            color = colors[i] if i < len(colors) else 'green'
            alpha = 0.7
        else:
            color = 'b'
            alpha = 0.8
        
        # Plot feature
        ax.plot(time_vector, features_np[i], color=color, linewidth=1.5, alpha=alpha)
        
        # Highlight P-wave region
        p_wave_mask = label_np == 1
        if np.any(p_wave_mask):
            p_wave_times = time_vector[p_wave_mask]
            ax.axvspan(p_wave_times[0], p_wave_times[-1], alpha=0.3, color='orange', 
                      label='P-wave Window')
        
        # Add earthquake time reference
        ax.axvline(0, color='gray', linestyle=':', alpha=0.7, label='Earthquake Time')
        
        ax.set_title(feature_names[i] if i < len(feature_names) else f'Feature {i}')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Add legend only to first subplot
            ax.legend()
    
    # Hide empty subplots
    if n_features > 1 and len(axes) > n_features:
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
    
    plt.suptitle(f'Seismic Features Visualization - {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Features visualization saved to {save_path}")

def plot_model_predictions(model, dataset, manifest, num_examples=2, save_path='model_predictions.png', 
                          use_physics_features=True):
    """Plot model predictions with features"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Adjust subplot layout based on features mode
    if use_physics_features:
        fig, axes = plt.subplots(num_examples, 3, figsize=(18, 4*num_examples))
        plot_cols = 3
    else:
        fig, axes = plt.subplots(num_examples, 3, figsize=(18, 4*num_examples))
        plot_cols = 3
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            features, true_label = dataset[i]
            
            if i < len(manifest):
                station_info = manifest[i]
            else:
                continue
            
            # Get model prediction
            features_batch = features.unsqueeze(0).to(device)
            output = model(features_batch)
            prob = output[0, 1, :].cpu().numpy()  # P-wave probability
            
            # Time vector
            sampling_rate = station_info['sampling_rate']
            pre_time = station_info['pre_time']
            post_time = station_info['post_time']
            total_samples = features.shape[1]
            time_vector = np.linspace(-pre_time, post_time, total_samples)
            
            # First channel (raw or processed)
            ax1 = axes[i, 0]
            if use_physics_features:
                # First channel is raw waveform in physics mode
                channel_data = features[0].numpy()
                ax1.plot(time_vector, channel_data, 'k-', linewidth=0.8, alpha=0.8)
                channel_title = 'Raw Waveform'
            else:
                # First channel is negatives → positives in raw mode
                channel_data = features[0].numpy()
                ax1.plot(time_vector, channel_data, 'r-', linewidth=0.8, alpha=0.8)
                channel_title = 'Negatives → Positives'
            
            # Add picks
            true_pick_samples = np.where(true_label == 1)[0]
            model_pick_sample = np.argmax(prob)
            
            if len(true_pick_samples) > 0:
                true_pick_time = time_vector[int(np.mean(true_pick_samples))]
                ax1.axvline(true_pick_time, color='red', linestyle='--', linewidth=2, 
                           label=f'True Pick ({true_pick_time:.2f}s)')
            
            model_pick_time = time_vector[model_pick_sample]
            ax1.axvline(model_pick_time, color='blue', linestyle='-', linewidth=2, 
                       label=f'Model Pick ({model_pick_time:.2f}s)')
            ax1.axvline(0, color='orange', linestyle=':', alpha=0.7, label='Earthquake Time')
            
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.set_title(f'Station {station_info["station"]} - {channel_title}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Second feature/channel
            ax2 = axes[i, 1]
            if use_physics_features:
                if features.shape[0] > 1:  # Check if we have STA/LTA features
                    sta_lta = features[1].numpy()  # Second channel should be STA/LTA
                    ax2.plot(time_vector, sta_lta, 'g-', linewidth=1.5, label='STA/LTA (0.5/10s)')
                    ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Typical Threshold')
                    feature_title = 'STA/LTA Feature'
                else:
                    ax2.text(0.5, 0.5, 'No STA/LTA features', transform=ax2.transAxes, ha='center')
                    feature_title = 'STA/LTA Feature'
            else:
                # Second channel is positives → negatives in raw mode
                if features.shape[0] > 1:
                    channel_data = features[1].numpy()
                    ax2.plot(time_vector, channel_data, 'b-', linewidth=1.5, label='Positives → Negatives')
                    feature_title = 'Positives → Negatives'
                else:
                    ax2.text(0.5, 0.5, 'No second channel', transform=ax2.transAxes, ha='center')
                    feature_title = 'Second Channel'
            
            if len(true_pick_samples) > 0:
                ax2.axvline(true_pick_time, color='red', linestyle='--', linewidth=2)
            ax2.axvline(model_pick_time, color='blue', linestyle='-', linewidth=2)
            ax2.axvline(0, color='orange', linestyle=':', alpha=0.7)
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.set_title(f'Station {station_info["station"]} - {feature_title}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Model probability
            ax3 = axes[i, 2]
            ax3.plot(time_vector, prob, 'b-', linewidth=2, label='P-wave Probability')
            ax3.fill_between(time_vector, 0, prob, alpha=0.3, color='blue')
            
            # Add picks
            if len(true_pick_samples) > 0:
                ax3.axvline(true_pick_time, color='red', linestyle='--', linewidth=2, 
                           label='True Pick')
                # Highlight true pick window
                window_start = time_vector[true_pick_samples[0]]
                window_end = time_vector[true_pick_samples[-1]]
                ax3.axvspan(window_start, window_end, alpha=0.2, color='green', 
                           label='True Pick Window')
            
            ax3.axvline(model_pick_time, color='blue', linestyle='-', linewidth=2, 
                       label='Model Pick')
            ax3.axvline(0, color='orange', linestyle=':', alpha=0.7, label='Earthquake Time')
            
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('P-wave Probability')
            ax3.set_title(f'Station {station_info["station"]} - Model Output')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    feature_mode = "Physics-Informed" if use_physics_features else "Enhanced Raw Waveform"
    plt.suptitle(f'Model Predictions - {feature_mode} Mode', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Model predictions plot saved to {save_path}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ ENHANCED OUTPUT MANAGEMENT WITH SEPARATE FOLDERS
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_output_directories(base_output_dir):
    """Create separate directories for different types of outputs"""
    
    # Create main output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create subdirectories
    pinn_plots_dir = os.path.join(base_output_dir, 'pinn_features_plots')
    raw_plots_dir = os.path.join(base_output_dir, 'raw_waveform_plots')
    models_dir = os.path.join(base_output_dir, 'trained_models')
    results_dir = os.path.join(base_output_dir, 'results')
    
    os.makedirs(pinn_plots_dir, exist_ok=True)
    os.makedirs(raw_plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"✅ Created output directories:")
    print(f"   Main: {base_output_dir}")
    print(f"   PINN plots: {pinn_plots_dir}")
    print(f"   Raw waveform plots: {raw_plots_dir}")
    print(f"   Models: {models_dir}")
    print(f"   Results: {results_dir}")
    
    return {
        'base': base_output_dir,
        'pinn_plots': pinn_plots_dir,
        'raw_plots': raw_plots_dir,
        'models': models_dir,
        'results': results_dir
    }

def save_results_with_directories(output_dirs, use_physics_features, train_losses, val_losses, 
                                pick_errors_seconds, model, feature_weights=None):
    """Save all results to appropriate directories"""
    
    # Determine mode suffix
    mode_suffix = "physics_informed" if use_physics_features else "enhanced_raw_waveform"
    plots_dir = output_dirs['pinn_plots'] if use_physics_features else output_dirs['raw_plots']
    
    # Save training curves and error histogram
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(pick_errors_seconds, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(pick_errors_seconds), color='red', linestyle='--', 
               label=f'Mean: {np.mean(pick_errors_seconds):.3f}s')
    plt.axvline(np.median(pick_errors_seconds), color='orange', linestyle='--', 
               label=f'Median: {np.median(pick_errors_seconds):.3f}s')
    plt.xlabel('Pick Time Error (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Pick Time Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature weights visualization (if available)
    plt.subplot(1, 3, 3)
    if feature_weights is not None:
        if use_physics_features:
            feature_names = ['Raw', 'STA/LTA 1', 'Log STA/LTA 1', 'STA/LTA 2', 'Log STA/LTA 2', 
                            'STA/LTA 3', 'Log STA/LTA 3', 'Envelope', 'Env. Deriv.', 'Inst. Freq.',
                            'Low Energy', 'Low Env.', 'Mid Energy', 'Mid Env.', 'High Energy', 'High Env.']
        else:
            feature_names = ['Neg→Pos', 'Pos→Neg', 'Original']
        
        weights = feature_weights[:len(feature_names)]
        names = feature_names[:len(weights)]
        
        bars = plt.bar(range(len(weights)), weights, color='lightblue', edgecolor='navy')
        plt.xlabel('Feature')
        plt.ylabel('Learned Weight')
        plt.title('Learned Feature Weights')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'No feature weights available', transform=plt.gca().transAxes, 
                ha='center', va='center')
        plt.title('Feature Weights')
    
    plt.tight_layout()
    training_results_path = os.path.join(plots_dir, f'{mode_suffix}_training_results.png')
    plt.savefig(training_results_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training results saved to: {training_results_path}")
    
    # Save model
    model_path = os.path.join(output_dirs['models'], f'{mode_suffix}_phase_picker.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save numerical results
    results_path = os.path.join(output_dirs['results'], f'{mode_suffix}_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"SEISMIC PHASE PICKER RESULTS - {mode_suffix.upper().replace('_', ' ')} MODE\n")
        f.write("="*80 + "\n\n")
        
        f.write("PICK ACCURACY METRICS:\n")
        f.write(f"Mean absolute error: {np.mean(pick_errors_seconds):.4f} ± {np.std(pick_errors_seconds):.4f} seconds\n")
        f.write(f"Median absolute error: {np.median(pick_errors_seconds):.4f} seconds\n")
        f.write(f"90th percentile error: {np.percentile(pick_errors_seconds, 90):.4f} seconds\n")
        f.write(f"95th percentile error: {np.percentile(pick_errors_seconds, 95):.4f} seconds\n\n")
        
        # Performance categories
        excellent = np.sum(pick_errors_seconds < 0.5)
        good = np.sum((pick_errors_seconds >= 0.5) & (pick_errors_seconds < 1.0))
        fair = np.sum((pick_errors_seconds >= 1.0) & (pick_errors_seconds < 2.0))
        poor = np.sum(pick_errors_seconds >= 2.0)
        total = len(pick_errors_seconds)
        
        f.write("PERFORMANCE CATEGORIES:\n")
        f.write(f"Excellent (< 0.5s): {excellent:3d} ({excellent/total*100:.1f}%)\n")
        f.write(f"Good (0.5-1.0s):   {good:3d} ({good/total*100:.1f}%)\n")
        f.write(f"Fair (1.0-2.0s):   {fair:3d} ({fair/total*100:.1f}%)\n")
        f.write(f"Poor (> 2.0s):     {poor:3d} ({poor/total*100:.1f}%)\n\n")
        
        f.write("TRAINING SUMMARY:\n")
        f.write(f"Final training loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final validation loss: {val_losses[-1]:.4f}\n")
        f.write(f"Best validation loss: {min(val_losses):.4f} (epoch {np.argmin(val_losses)+1})\n")
        
        if feature_weights is not None:
            f.write("\nLEARNED FEATURE WEIGHTS:\n")
            if use_physics_features:
                feature_names = ['Raw', 'STA/LTA 1', 'Log STA/LTA 1', 'STA/LTA 2', 'Log STA/LTA 2', 
                                'STA/LTA 3', 'Log STA/LTA 3', 'Envelope', 'Env. Deriv.', 'Inst. Freq.',
                                'Low Energy', 'Low Env.', 'Mid Energy', 'Mid Env.', 'High Energy', 'High Env.']
            else:
                feature_names = ['Negatives→Positives', 'Positives→Negatives', 'Original Waveform']
            
            for i, (name, weight) in enumerate(zip(feature_names[:len(feature_weights)], feature_weights)):
                f.write(f"  {name:20s}: {weight:.4f}\n")
    
    print(f"Numerical results saved to: {results_path}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN EXECUTION WITH ENHANCED DUAL-CHANNEL PROCESSING AND OUTPUT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════════════

def main(use_physics_features=True, num_epochs=25, output_directory="ml_pipeline_output"):
    """
    Main execution function with enhanced dual-channel processing and output management
    
    Args:
        use_physics_features (bool): Whether to use physics-informed features or enhanced raw waveforms
        num_epochs (int): Number of training epochs
        output_directory (str): Base directory for all outputs
    """
    
    print("="*80)
    if use_physics_features:
        print("PHYSICS-INFORMED SEISMIC PHASE PICKER")
    else:
        print("ENHANCED DUAL-CHANNEL RAW WAVEFORM SEISMIC PHASE PICKER")
    print("="*80)
    
    feature_mode = "Physics-Informed Features" if use_physics_features else "Enhanced Dual-Channel Raw Waveforms"
    print(f"🔧 Configuration: {feature_mode} mode")
    print(f"🕐 Training epochs: {num_epochs}")
    print(f"📁 Output directory: {output_directory}")
    
    # Create output directories
    print("\n📁 SETTING UP OUTPUT DIRECTORIES")
    print("-" * 50)
    output_dirs = create_output_directories(output_directory)
    
    # Example earthquakes
    earthquakes = [
        {"time": '2022-12-20T10:34:24', "lat": 40.369, "lon": -124.588, "radius": 100},  # Ferndale
        {"time": "2010-01-10T00:27:39", "lat": 40.652, "lon": -124.693, "radius": 100},  # Eureka
    ]
    
    # Collect data from multiple earthquakes
    print('\n')
    print("-" * 25)
    print("DATA MINING")
    print("-" * 25)

    all_manifest = []
    for eq in earthquakes:
        print(f"\n  Processing earthquake: {eq['time']}")
        try:
            manifest = analyze_earthquake_manifest(
                eq_time=eq['time'],
                eq_lat=eq['lat'],
                eq_lon=eq['lon'],
                radius_km=eq['radius']
            )
            all_manifest.extend(manifest)
            print(f"    ✅ Collected {len(manifest)} stations for this earthquake")
        except Exception as e:
            print(f"    ❌ Error processing earthquake {eq['time']}: {e}")
    
    print(f"\nTotal stations collected: {len(all_manifest)}")
    
    if len(all_manifest) == 0:
        print("❌ No data collected. Exiting.")
        return
    
    # Split data
    print('\n')
    print("-" * 25)
    print("DATA SPLITTING")
    print("-" * 25)

    train_manifest, val_manifest = train_test_split(all_manifest, test_size=0.2, random_state=42)
    print(f"    Training samples: {len(train_manifest)}")
    print(f"    Validation samples: {len(val_manifest)}")
    
    # Create datasets
    print('\n')
    print("-" * 50)
    print(f"CREATING DATASETS - {feature_mode.upper()} MODE")
    print("-" * 50)

    print("    Creating training dataset...")
    train_dataset = AdaptiveSeismicDataset(train_manifest, window_size=5, 
                                         use_physics_features=use_physics_features)

    print("    Creating validation dataset...")
    val_dataset = AdaptiveSeismicDataset(val_manifest, window_size=5, 
                                       use_physics_features=use_physics_features)

    # Check if datasets were created successfully
    if not hasattr(train_dataset, 'data') or len(train_dataset.data) == 0:
        print("❌ Failed to create training dataset. No data loaded.")
        return

    if not hasattr(val_dataset, 'data') or len(val_dataset.data) == 0:
        print("❌ Failed to create validation dataset. No data loaded.")
        return

    print(f"    ✅ Training dataset size: {len(train_dataset)}")
    print(f"    ✅ Validation dataset size: {len(val_dataset)}")

    # Visualize features for first sample
    print(f"\n📊 VISUALIZING {feature_mode.upper()} FEATURES")
    print("-" * 50)
    
    if len(train_dataset) > 0:
        # Save to appropriate directory
        plots_dir = output_dirs['pinn_plots'] if use_physics_features else output_dirs['raw_plots']
        feature_viz_path = os.path.join(plots_dir, 'features_visualization.png')
        
        visualize_features(train_dataset, sample_idx=0, 
                         save_path=feature_viz_path,
                         use_physics_features=use_physics_features)
    else:
        print("❌ No training data available for visualization")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Initialize model
    print('\n')
    print("-" * 50)
    print(f"INITIALIZING {feature_mode.upper()} MODEL")
    print("-" * 50)
    
    # Get number of input channels from first sample
    sample_features, _ = train_dataset[0]
    n_input_channels = sample_features.shape[0]
    print(f"\n     ✅ Input channels (features): {n_input_channels}")
    print(f"     ✅ Feature shape: {sample_features.shape}")
    
    if use_physics_features:
        print("     📊 Physics-informed features detected:")
        feature_names = ['Raw', 'STA/LTA 1', 'Log STA/LTA 1', 'STA/LTA 2', 'Log STA/LTA 2', 
                        'STA/LTA 3', 'Log STA/LTA 3', 'Envelope', 'Env. Deriv.', 'Inst. Freq.',
                        'Low Energy', 'Low Env.', 'Mid Energy', 'Mid Env.', 'High Energy', 'High Env.']
        for i, name in enumerate(feature_names[:n_input_channels]):
            print(f"          Channel {i+1}: {name}")
    else:
        print("     🌊 Enhanced dual-channel raw waveforms detected:")
        channel_names = ['Negatives → Positives', 'Positives → Negatives', 'Original Waveform']
        for i, name in enumerate(channel_names[:n_input_channels]):
            print(f"          Channel {i+1}: {name}")
    
    model = AdaptiveUNet1D(in_channels=n_input_channels, out_channels=2,
                          use_physics_features=use_physics_features)
    
    # Initialize feature weights with correct dimensions
    model.initialize_feature_weights(n_input_channels)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        Total parameters: {total_params:,}")
    print(f"        Trainable parameters: {trainable_params:,}")
    
    # Print initial feature weights
    if hasattr(model, 'feature_weights') and model.feature_weights is not None:
        print(f"        Initial learnable feature weights: {model.feature_weights.data}")
    
    # Train model
    print('\n')
    print("-" * 50)
    print("🚀 STARTING TRAINING")
    print("-" * 50)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=num_epochs)
    
    # Load best model
    try:
        model.load_state_dict(torch.load('best_adaptive_model.pth'))
        print("✅ Loaded best model from training")
    except:
        print("⚠️ Could not load best model, using current state")
    
    # Evaluate model
    print("\n📊 EVALUATING MODEL")
    print("-" * 50)
    
    pick_errors = evaluate_picks(model, val_dataset, val_manifest)
    
    # Calculate metrics (assuming 100 Hz sampling rate)
    sampling_rate = 100
    pick_errors_seconds = pick_errors / sampling_rate
    
    print(f"\n🎯 PICK ACCURACY RESULTS - {feature_mode.upper()} MODE")
    print("=" * 60)
    print(f"Mean absolute error: {np.mean(pick_errors_seconds):.4f} ± {np.std(pick_errors_seconds):.4f} seconds")
    print(f"Median absolute error: {np.median(pick_errors_seconds):.4f} seconds")
    print(f"90th percentile error: {np.percentile(pick_errors_seconds, 90):.4f} seconds")
    print(f"95th percentile error: {np.percentile(pick_errors_seconds, 95):.4f} seconds")
    
    # Performance categories
    excellent = np.sum(pick_errors_seconds < 0.5)
    good = np.sum((pick_errors_seconds >= 0.5) & (pick_errors_seconds < 1.0))
    fair = np.sum((pick_errors_seconds >= 1.0) & (pick_errors_seconds < 2.0))
    poor = np.sum(pick_errors_seconds >= 2.0)
    
    print(f"\n📈 Performance Categories:")
    print(f"  Excellent (< 0.5s): {excellent:3d} ({excellent/len(pick_errors_seconds)*100:.1f}%)")
    print(f"  Good (0.5-1.0s):   {good:3d} ({good/len(pick_errors_seconds)*100:.1f}%)")
    print(f"  Fair (1.0-2.0s):   {fair:3d} ({fair/len(pick_errors_seconds)*100:.1f}%)")
    print(f"  Poor (> 2.0s):     {poor:3d} ({poor/len(pick_errors_seconds)*100:.1f}%)")
    
    # Get final feature weights
    final_feature_weights = None
    if hasattr(model, 'feature_weights') and model.feature_weights is not None:
        final_feature_weights = model.feature_weights.data.cpu().numpy()
        print(f"\n🎛️ Final learned feature weights:")
        if use_physics_features:
            feature_names = ['Raw', 'STA/LTA 1', 'Log STA/LTA 1', 'STA/LTA 2', 'Log STA/LTA 2', 
                            'STA/LTA 3', 'Log STA/LTA 3', 'Envelope', 'Env. Deriv.', 'Inst. Freq.',
                            'Low Energy', 'Low Env.', 'Mid Energy', 'Mid Env.', 'High Energy', 'High Env.']
        else:
            feature_names = ['Negatives→Positives', 'Positives→Negatives', 'Original Waveform']
        
        for i, (name, weight) in enumerate(zip(feature_names[:len(final_feature_weights)], final_feature_weights)):
            print(f"  {name:20s}: {weight:.4f}")
    
    # Create visualizations and save results
    print("\n📊 CREATING VISUALIZATIONS AND SAVING RESULTS")
    print("-" * 50)
    
    # Save all results to organized directories
    save_results_with_directories(output_dirs, use_physics_features, train_losses, val_losses, 
                                pick_errors_seconds, model, final_feature_weights)
    
    # Plot model predictions and save to appropriate directory
    plots_dir = output_dirs['pinn_plots'] if use_physics_features else output_dirs['raw_plots']
    pred_save_path = os.path.join(plots_dir, 'model_predictions.png')
    plot_model_predictions(model, val_dataset, val_manifest, num_examples=2, 
                          save_path=pred_save_path, use_physics_features=use_physics_features)
    
    # Clean up temporary best model file
    if os.path.exists('best_adaptive_model.pth'):
        os.remove('best_adaptive_model.pth')
    
    print(f"\n🎉 TRAINING COMPLETE - {feature_mode.upper()} MODE!")
    print(f"📁 All outputs saved to: {output_directory}")
    print("=" * 80)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'pick_errors_seconds': pick_errors_seconds,
        'feature_weights': final_feature_weights,
        'output_dirs': output_dirs
    }

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ COMPARISON FUNCTION FOR BOTH MODES
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_comparison_study(num_epochs=25, base_output_dir="comparison_study_output"):
    """
    Run both physics-informed and enhanced raw waveform modes for comparison
    
    Args:
        num_epochs (int): Number of training epochs for each mode
        base_output_dir (str): Base directory for comparison outputs
    """
    
    print("\n" + "="*80)
    print("🔄 COMPREHENSIVE COMPARISON STUDY")
    print("   Physics-Informed Features vs Enhanced Dual-Channel Raw Waveforms")
    print("="*80)
    
    # Create comparison output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    results = {}
    
    # Run Physics-Informed Features mode
    print("\n🧪 RUNNING PHYSICS-INFORMED FEATURES MODE...")
    print("-" * 60)
    pinn_output_dir = os.path.join(base_output_dir, "physics_informed_mode")
    results['physics'] = main(use_physics_features=True, num_epochs=num_epochs, 
                             output_directory=pinn_output_dir)
    
    # Run Enhanced Raw Waveform mode
    print("\n🌊 RUNNING ENHANCED DUAL-CHANNEL RAW WAVEFORM MODE...")
    print("-" * 60)
    raw_output_dir = os.path.join(base_output_dir, "enhanced_raw_mode")
    results['raw'] = main(use_physics_features=False, num_epochs=num_epochs, 
                         output_directory=raw_output_dir)
    
    # Create comparison summary
    print("\n📊 CREATING COMPARISON SUMMARY")
    print("-" * 50)
    
    # Compare results
    physics_errors = results['physics']['pick_errors_seconds']
    raw_errors = results['raw']['pick_errors_seconds']
    
    comparison_data = {
        'Physics-Informed': {
            'mean_error': np.mean(physics_errors),
            'median_error': np.median(physics_errors),
            'std_error': np.std(physics_errors),
            '90th_percentile': np.percentile(physics_errors, 90),
            'excellent_picks': np.sum(physics_errors < 0.5),
            'good_picks': np.sum((physics_errors >= 0.5) & (physics_errors < 1.0)),
        },
        'Enhanced Raw': {
            'mean_error': np.mean(raw_errors),
            'median_error': np.median(raw_errors),
            'std_error': np.std(raw_errors),
            '90th_percentile': np.percentile(raw_errors, 90),
            'excellent_picks': np.sum(raw_errors < 0.5),
            'good_picks': np.sum((raw_errors >= 0.5) & (raw_errors < 1.0)),
        }
    }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Error distribution comparison
    axes[0, 0].hist(physics_errors, bins=30, alpha=0.7, label='Physics-Informed', color='blue')
    axes[0, 0].hist(raw_errors, bins=30, alpha=0.7, label='Enhanced Raw', color='red')
    axes[0, 0].set_xlabel('Pick Time Error (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[0, 1].boxplot([physics_errors, raw_errors], labels=['Physics-Informed', 'Enhanced Raw'])
    axes[0, 1].set_ylabel('Pick Time Error (seconds)')
    axes[0, 1].set_title('Error Distribution Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance categories comparison
    categories = ['Excellent\n(< 0.5s)', 'Good\n(0.5-1.0s)']
    physics_perf = [comparison_data['Physics-Informed']['excellent_picks'],
                   comparison_data['Physics-Informed']['good_picks']]
    raw_perf = [comparison_data['Enhanced Raw']['excellent_picks'],
               comparison_data['Enhanced Raw']['good_picks']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, physics_perf, width, label='Physics-Informed', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, raw_perf, width, label='Enhanced Raw', color='red', alpha=0.7)
    axes[1, 0].set_ylabel('Number of Picks')
    axes[1, 0].set_title('Performance Categories Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training loss comparison
    physics_losses = results['physics']['val_losses']
    raw_losses = results['raw']['val_losses']
    
    axes[1, 1].plot(physics_losses, label='Physics-Informed', color='blue')
    axes[1, 1].plot(raw_losses, label='Enhanced Raw', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_title('Training Progress Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(base_output_dir, 'comprehensive_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save comparison summary
    summary_path = os.path.join(base_output_dir, 'comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE COMPARISON STUDY RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-"*40 + "\n")
        for mode, data in comparison_data.items():
            f.write(f"\n{mode} Mode:\n")
            f.write(f"  Mean error:        {data['mean_error']:.4f} ± {data['std_error']:.4f} seconds\n")
            f.write(f"  Median error:      {data['median_error']:.4f} seconds\n")
            f.write(f"  90th percentile:   {data['90th_percentile']:.4f} seconds\n")
            f.write(f"  Excellent picks:   {data['excellent_picks']:3d}\n")
            f.write(f"  Good picks:        {data['good_picks']:3d}\n")
        
        # Determine winner
        f.write("\nSUMMARY:\n")
        f.write("-"*20 + "\n")
        if comparison_data['Physics-Informed']['mean_error'] < comparison_data['Enhanced Raw']['mean_error']:
            f.write("🏆 Physics-Informed mode achieved better mean accuracy\n")
        else:
            f.write("🏆 Enhanced Raw mode achieved better mean accuracy\n")
            
        if comparison_data['Physics-Informed']['median_error'] < comparison_data['Enhanced Raw']['median_error']:
            f.write("🏆 Physics-Informed mode achieved better median accuracy\n")
        else:
            f.write("🏆 Enhanced Raw mode achieved better median accuracy\n")
            
        if comparison_data['Physics-Informed']['excellent_picks'] > comparison_data['Enhanced Raw']['excellent_picks']:
            f.write("🏆 Physics-Informed mode achieved more excellent picks\n")
        else:
            f.write("🏆 Enhanced Raw mode achieved more excellent picks\n")
    
    print(f"📊 Comparison summary saved to: {summary_path}")
    print(f"📈 Comparison plots saved to: {comparison_plot_path}")
    print(f"\n🎉 COMPREHENSIVE COMPARISON STUDY COMPLETE!")
    print(f"📁 All comparison outputs saved to: {base_output_dir}")
    
    return results, comparison_data

if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════════════════════════
    # 🎯 MAIN CONFIGURATION SECTION
    # ═══════════════════════════════════════════════════════════════════════════════════════
    
    # Choose your mode:
    # True = Physics-Informed Features mode
    # False = Enhanced Dual-Channel Raw Waveform mode
    USE_PHYSICS_FEATURES = False   # 🌊 SET TO FALSE FOR ENHANCED RAW WAVEFORM MODE
    
    # Set training parameters
    TRAINING_EPOCHS = 25
    
    # Set output directory (user configurable)
    OUTPUT_DIRECTORY = "seismic_phase_picker_output"  # 📁 CHANGE THIS TO YOUR PREFERRED PATH
    
    # ═══════════════════════════════════════════════════════════════════════════════════════
    # 🚀 SINGLE MODE EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════════════════
    
    # Run single mode
    results = main(use_physics_features=USE_PHYSICS_FEATURES, 
                  num_epochs=TRAINING_EPOCHS, 
                  output_directory=OUTPUT_DIRECTORY)
    
    # ═══════════════════════════════════════════════════════════════════════════════════════
    # 🔄 COMPARISON STUDY (UNCOMMENT TO RUN BOTH MODES)
    # ═══════════════════════════════════════════════════════════════════════════════════════
    
    # Uncomment the lines below to run a comprehensive comparison study of both modes:
    
    # print("\n" + "="*80)
    # print("🔄 RUNNING COMPREHENSIVE COMPARISON STUDY")
    # print("="*80)
    
    # comparison_results, comparison_data = run_comparison_study(
    #     num_epochs=15,  # Reduced epochs for faster comparison
    #     base_output_dir="comprehensive_comparison_study"
    # )
    
    # ═══════════════════════════════════════════════════════════════════════════════════════
    # 📊 USAGE EXAMPLES
    # ═══════════════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("✅ SCRIPT EXECUTION COMPLETE!")
    print("="*80)
    print("\n📖 USAGE SUMMARY:")
    print("  • Physics-Informed Mode: Traditional seismological features + raw waveform")
    print("  • Enhanced Raw Mode: Dual-channel processing (neg→pos, pos→neg, original)")
    print("  • All outputs organized in separate folders:")
    print("    - PINN features plots → pinn_features_plots/")
    print("    - Raw waveform plots → raw_waveform_plots/")
    print("    - Trained models → trained_models/")
    print("    - Numerical results → results/")
    print("\n🔧 To change modes, modify USE_PHYSICS_FEATURES in the main section")
    print("📁 To change output directory, modify OUTPUT_DIRECTORY in the main section")        # Original waveform
            ])
        else:
            channels = np.array([
                channel_1,           # Negatives → Positives
                channel_2            # Positives → Negatives
            ])
        
        return channels

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ ENHANCED U-NET ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Enhanced convolution block with batch normalization and dropout"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.1):
        super().__init__()
        
        self.doubleConv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout)
        )
        
    def forward(self, x):
        return self.doubleConv(x)

class AttentionBlock(nn.Module):
    """Attention mechanism to focus on important features"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

class AdaptiveUNet1D(nn.Module):
    """
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │ Adaptive 1-D U-Net                                                                   │
    │ ──────────────────────────────────────────────────────────────────────────────────── │
    │ • Can work with or without physics-informed features                                 │
    │ • Can work with enhanced raw waveform channels                                       │
    │ • Attention mechanisms to focus on important features                                │
    │ • Batch normalization and dropout for better generalization                          │
    │ • Multi-scale feature extraction through encoder-decoder architecture                │
    │ • Dynamically adapts to different input channel sizes                                │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256], 
                 dropout=0.1, use_physics_features=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.use_physics_features = use_physics_features
        
        # ==============================
        # 1️⃣ Downsampling Path (ENCODER)
        # ==============================
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attentions_down = nn.ModuleList()
        
        current_channels = in_channels
        for feat in features:
            self.downs.append(ConvBlock(current_channels, feat, dropout=dropout))
            self.pools.append(nn.MaxPool1d(2))
            self.attentions_down.append(AttentionBlock(feat))
            current_channels = feat
        
        # ============================================
        # 2️⃣ Bottleneck (connects ENCODER & DECODER)
        # ============================================
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, dropout=dropout)
        self.bottleneck_attention = AttentionBlock(features[-1]*2)
        
        # ==============================
        # 3️⃣ Upsampling path (DECODER)
        # ==============================
        self.ups = nn.ModuleList()
        self.attentions_up = nn.ModuleList()
        
        for feat in reversed(features):
            # Transposed convolution for upsampling
            self.ups.append(nn.ConvTranspose1d(feat*2, feat, kernel_size=2, stride=2))
            # Convolution block for feature fusion
            self.ups.append(ConvBlock(feat*2, feat, dropout=dropout))
            # Attention for refined features
            self.attentions_up.append(AttentionBlock(feat))
        
        # ===========================
        # 4️⃣ Final Classification Layer
        # ===========================
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
        # Feature weighting (for both physics features and raw channels)
        self.feature_weights = None
        self.feature_weights = nn.Parameter(torch.ones(in_channels))
        
    def initialize_feature_weights(self, actual_channels):
        """Initialize feature weights based on actual number of input channels"""
        if self.feature_weights is None or self.feature_weights.size(0) != actual_channels:
            self.feature_weights = nn.Parameter(torch.ones(actual_channels))
            self.in_channels = actual_channels
            print(f"Initialized feature weights for {actual_channels} channels")
        
    def forward(self, x):
        # Apply learnable weights to input features
        # Initialize feature weights if needed
        if self.feature_weights is None:
            self.initialize_feature_weights(x.size(1))
        
        if x.size(1) == self.feature_weights.size(0):
            weighted_x = x * self.feature_weights.view(1, -1, 1)
        else:
            print(f"Warning: Input channels ({x.size(1)}) != feature weights ({self.feature_weights.size(0)})")
            # Adjust feature weights if mismatch
            self.initialize_feature_weights(x.size(1))
            weighted_x = x * self.feature_weights.view(1, -1, 1)
        
        skip_connections = []
        
        # ---------------- Encoder ----------------
        for i, (down, pool, attention) in enumerate(zip(self.downs, self.pools, self.attentions_down)):
            weighted_x = down(weighted_x)
            weighted_x = attention(weighted_x)  # Apply attention
            skip_connections.append(weighted_x)
            weighted_x = pool(weighted_x)
        
        # --------------- Bottleneck ---------------
        weighted_x = self.bottleneck(weighted_x)
        weighted_x = self.bottleneck_attention(weighted_x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # ---------------- Decoder ----------------
        for idx in range(0, len(self.ups), 2):
            # Upsampling
            weighted_x = self.ups[idx](weighted_x)
            
            # Get corresponding skip connection
            skip_conn = skip_connections[idx//2]
            
            # Handle size mismatches
            if weighted_x.shape[-1] != skip_conn.shape[-1]:
                weighted_x = F.pad(weighted_x, (0, skip_conn.shape[-1] - weighted_x.shape[-1]))
            
            # Concatenate skip connection
            weighted_x = torch.cat((skip_conn, weighted_x), dim=1)
            
            # Refine features
            weighted_x = self.ups[idx+1](weighted_x)
            
            # Apply attention
            attention_idx = idx // 2
            if attention_idx < len(self.attentions_up):
                weighted_x = self.attentions_up[attention_idx](weighted_x)
        
        # Final classification
        output = self.final_conv(weighted_x)
        return F.softmax(output, dim=1)

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ DATA LOADING AND PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════════════

# Configure S3 Client for Public NCEDC Access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-west-2')
BUCKET_NAME = 'ncedc-pds'

def station_data_exists(station, eq_time, pre_time, post_time, client: Client, 
                       network: str, location: str, channel: str,
                       s3_client, bucket_name: str) -> bool:
    """Check if station data exists in S3"""
    day0 = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    jul = day0.julday
    fname = f"{station.code}.{network}.{channel}..D.{day0.year}.{jul:03d}"
    key = f"continuous_waveforms/{network}/{day0.year}/{day0.year}.{jul:03d}/{fname}"
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError:
        return False

def filter_inventory(inventory: Inventory, eq_time, pre_time, post_time, 
                    client: Client, network: str, location: str, channel: str,
                    s3_client, bucket_name: str) -> Inventory:
    """Filter inventory to only include stations with available data"""
    kept_networks = []
    for net in inventory.networks:
        kept_stns = []
        for st in net.stations:
            if station_data_exists(st, eq_time, pre_time, post_time, client, 
                                 network, location, channel, s3_client, bucket_name):
                kept_stns.append(st)
        if kept_stns:
            net.stations = kept_stns
            kept_networks.append(net)
    
    inventory.networks = kept_networks
    return inventory

# Load pretrained PhaseNet for reference picks
picker = sbm.PhaseNet.from_pretrained("original")

def process_stations(station, start_time, pre_time, post_time, eq_time,
                    network, channel, inventory):
    """Process individual stations and return metadata"""
    station_code = station.code
    
    # Build S3 key
    file_name = f'{station_code}.{network}.{channel}..D.{start_time.year}.{start_time.julday:03d}'
    key = f"continuous_waveforms/{network}/{start_time.year}/{start_time.year}.{start_time.julday:03d}/{file_name}"
    
    try:
        # Stream data from S3
        resp = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data_stream = resp['Body']
        buff = BytesIO(data_stream.read())
        buff.seek(0)
        
        # Read waveform
        st_stream = read(buff, format='MSEED')
        st_stream.trim(starttime=eq_time - pre_time, endtime=eq_time + post_time)
        
        print(f"- Streamed {len(st_stream)} traces for station {station_code}")
        
        if len(st_stream) < 1:
            print(f"-- Skipping station {station_code} (no data)")
            return None
        
        # Get single trace
        tr = st_stream[0]
        
        # Remove instrument response
        tr.remove_response(inventory=inventory, output="DISP")
        
        # Get PhaseNet picks for reference
        picks = picker.classify(st_stream, batch_size=256, P_threshold=0.075, S_threshold=0.1).picks
        if not picks:
            print(f"-- No picks found for station {station_code}")
            return None
        
        # Use first P arrival
        p_time = picks[0].peak_time
        
        # Return the station data
        return {
            'bucket': BUCKET_NAME,
            'key': key,
            'pick_time': p_time,
            'pre_time': pre_time,
            'post_time': post_time,
            'eq_time': eq_time,
            'station': station_code,
            'sampling_rate': tr.stats.sampling_rate
        }
        
    except Exception as e:
        print(f"-- Error processing station {station_code}: {e}")
        return None

def analyze_earthquake_manifest(eq_time, eq_lon, eq_lat, radius_km,
                               client_name='NCEDC', network='NC', location='*', 
                               channel='HNE', pre_time=3, post_time=120):
    """Analyze earthquake and build station manifest"""
    
    if not isinstance(eq_time, UTCDateTime):
        eq_time = UTCDateTime(eq_time)
    
    # Define time window
    start_time = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = eq_time.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Get station inventory
    client = Client(client_name)
    print("    Making inventory of stations...")
    inventory = client.get_stations(
        network=network, latitude=eq_lat, longitude=eq_lon,
        starttime=start_time, endtime=end_time, maxradius=radius_km/111.2,
        location=location, channel=channel, level="response"
    )
    
    print("    Filtering inventory...")
    inventory = filter_inventory(inventory, eq_time, pre_time, post_time, 
                               client, network, location, channel, s3, BUCKET_NAME)
    
    stations = inventory[0].stations
    print(f"    Found {len(stations)} stations within {radius_km} km")
    
    manifest = []

    # Process stations in parallel
    tasks = [dask.delayed(process_stations)(
        station, start_time, pre_time, post_time, eq_time,
        network, channel, inventory
    ) for station in stations]
    
    # Compute all tasks and get results
    results = dask.compute(*tasks, scheduler='threads')
    
    # Filter out None results and build manifest
    manifest = [result for result in results if result is not None]
    
    print(f"    Successfully processed {len(manifest)} stations")
    
    return manifest

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ ENHANCED DATASET WITH DUAL PROCESSING MODES
# ═══════════════════════════════════════════════════════════════════════════════════════

class AdaptiveSeismicDataset(Dataset):
    """
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │ Adaptive Seismic Dataset                                                             │
    │ ──────────────────────────────────────────────────────────────────────────────────── │
    │ • Loads raw seismic waveforms                                                        │
    │ • Mode 1: Physics-informed features (STA/LTA, envelope, spectral)                   │
    │ • Mode 2: Enhanced raw waveforms (dual-channel processing)                           │
    │ • Creates multi-channel input based on configuration                                 │
    │ • Generates labels for P-wave detection                                              │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, manifest, window_size=5, target_length=None, use_physics_features=True):
        self.window_size = window_size
        self.use_physics_features = use_physics_features
        
        # Initialize processors based on mode
        if self.use_physics_features:
            self.feature_extractor = PhysicsInformedFeatures(sampling_rate=100)
            self.waveform_processor = None
        else:
            self.feature_extractor = None
            self.waveform_processor = RawWaveformProcessor(include_original=True)
        
        # Determine target length BEFORE parallel processing
        if target_length is None:
            if manifest:
                sample_item = manifest[0]
                duration = sample_item['pre_time'] + sample_item['post_time']
                self.target_length = int(duration * 100)  # Assume 100 Hz
                print(f"Using estimated target length: {self.target_length} samples")
            else:
                self.target_length = 3000
                print(f"Using fallback target length: {self.target_length} samples")
        else:
            self.target_length = target_length
        
        # Process all waveforms in parallel using Dask
        if use_physics_features:
            feature_mode = "with physics-informed features"
        else:
            feature_mode = "with enhanced dual-channel raw waveforms"
        
        print(f"Processing {len(manifest)} waveforms in parallel ({feature_mode})...")
        
        # Create delayed tasks for each waveform
        tasks = [
            dask.delayed(self._process_waveform_standalone)(
                item, window_size, self.target_length, use_physics_features
            ) for item in manifest
        ]
        
        # Compute all tasks
        results = dask.compute(*tasks, scheduler='threads')
        
        # Unpack successful results
        self.data = []
        self.labels = []
        successful_loads = 0
        for result in results:
            if result is not None:
                self.data.append(result['features'])
                self.labels.append(result['label'])
                successful_loads += 1
        
        print(f"Successfully loaded {successful_loads}/{len(manifest)} waveforms")
    
    @staticmethod
    def _process_waveform_standalone(item, window_size, target_length, use_physics_features):
        """Standalone static method for Dask processing"""
        
        try:
            # Load waveform from S3
            resp = s3.get_object(Bucket=item['bucket'], Key=item['key'])
            buff = BytesIO(resp['Body'].read())
            buff.seek(0)
            
            # Read waveform
            st_stream = read(buff, format='MSEED')
            st_stream.trim(starttime=item['eq_time'] - item['pre_time'], 
                        endtime=item['eq_time'] + item['post_time'])
            
            if len(st_stream) < 1:
                return None
            
            waveform = st_stream[0]
            sampling_rate = waveform.stats.sampling_rate
            
            # Calculate pick sample index
            pick_offset = (item['pick_time'] - (item['eq_time'] - item['pre_time']))
            pick_sample = int(pick_offset * sampling_rate)
            
            # Normalize waveform length
            waveform_data = waveform.data
            current_length = len(waveform_data)
            
            if current_length == target_length:
                waveform_norm = waveform_data.copy()
                pick_sample_norm = pick_sample
            elif current_length > target_length:
                # Truncate, keeping pick centered
                excess = current_length - target_length
                if pick_sample < target_length // 2:
                    start_trim = max(0, excess // 4)
                elif pick_sample > current_length - target_length // 2:
                    start_trim = excess - max(0, excess // 4)
                else:
                    start_trim = excess // 2
                
                end_idx = start_trim + target_length
                waveform_norm = waveform_data[start_trim:end_idx]
                pick_sample_norm = pick_sample - start_trim
            else:
                # Pad with zeros
                pad_needed = target_length - current_length
                pad_start = pad_needed // 2
                pad_end = pad_needed - pad_start
                
                waveform_norm