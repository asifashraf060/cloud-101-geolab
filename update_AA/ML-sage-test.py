import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Import the necessary module from PyTorch
import torch.nn as nn

# Define a class to represent a double convolution block
class ConvBlock(nn.Module): # inherit the base class for neural networks for PyTorch
    """
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ ConvBlock: The basic building brick                                                              │
    │ ------------------------------------------------------------------------------------------------ │
    │ • Two 1-D convolutions (doubleConv), each followed by ReLU.                                      │
    │ • Keeps the time length unchanged (padding=1) but learns richer representations (more channels). |                                    
    │ • Re-using the same pattern everywhere keeps the code short and consistent.                      |                                    
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, 
                 in_channels,       # No of input channels (e.g., 1 for grayscale, 3 for RGB)
                 out_channels,      # No of output channels (i.e., number of feature maps).
                 kernel_size = 3,   # Size of the convolutional filter (default is 3x3).
                 padding = 1        # Padding added to both sides of input (default is 1 to preserve size).
                 ):
        super().__init__()  # Initialize the parent nn.Module class
        
        # ① First 1-D convolution.
        #    – Looks at a sliding 3-sample window (for kernel_size=3).
        #    – padding=1 so output length == input length.
        #    – Learns out_channel different “filters” (patterns) in parallel.
        conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

        # ② ReLU: keeps only positive values → adds non-linearity.
        relu1 = nn.ReLU(inplace=True)

        # ③ repeat these two steps once more, so we have a 'two-layer feature extractor'
        conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        relu2 = nn.ReLU(inplace=True)

        # Put the four layers into one tidy “Sequential” container.
        self.doubleConv = nn.Sequential(conv1, relu1, conv2, relu2)
        
    def forward(self, x):
        # Simply run x through the mini-network with four layers (Conv-ReLu+Conv-ReLu) defined above.
        return self.doubleConv(x) # Pass input through the doubleConv block
    
    # now, we are going to create the entire U-net structure

class UNet1D(nn.Module): # inherit the base class for neural networks for PyTorch
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 1-D U-Net (originally designed for images, adapted to time series)     │
    │                                                                        │
    │ • Goal: predict a label for every time sample (e.g., “noise / P / S”). │
    │ • Shape convention: (batch, channels, length) — PyTorch’s default.     │
    │ • Two big parts:                                                       │
    │     Encoder (Down path): “What is present?”                            │
    │     Decoder (Up path)  : “Where exactly is it?”                        │
    │   Skip connections copy high-resolution info from encoder to decoder.  │
    └────────────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, 
                 in_channels = 3,               # e.g., 3-component seismogram
                 out_channels = 3,              # e.g., P-wave, S-wave, noise
                 features = [16, 32, 64, 128]   # network width; doubles every step by default
                 ):
        super().__init__()

        # ==============================
        # 1️⃣ Downsampling Path (ENCODER)
        # ==============================
        # Build the ENCODER (“Downs”) — series of ConvBlock + MaxPool  
        #  • Each ConvBlock learns richer features                     
        #  • MaxPool (done later in `forward`) halves time resolution
        #      doubling the “receptive field” (context window).
        # ---------------------------------------------------------------
        self.downs = nn.ModuleList()
        for feat in features:
            self.downs.append(ConvBlock(in_channels, feat)) # using the ConvBlock function we defined earlier
            in_channels = feat          # update in_channels for the next block where we are incrasing the features

        # ============================================
        # 2️⃣ Bottleneck (connects ENCODER & DECODER)
        # ============================================
        # bottleneck refers to the deepest layer in the U-Net, connects encoder and decoder
        #    • Sees the shortest signal (most compressed) but richest channels.
        #    • Doubles channels one last time.
        # --------------------------------------------------------------------------------------
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        # ==============================
        # 3️⃣ Upsampling path (DECODER)
        # ==============================
        # Build the DECODER (“Ups”) — mirror of encoder
        #    For every level we create two layers:
        #       a) ConvTranspose1d for learnable upsampling (×2 length)
        #       b) ConvBlock to fuse the upsampled data with a skip connection
        # ------------------------------------------------------------------------
        self.ups   = nn.ModuleList()
        for feat in reversed(features):         # traverse 128→64→32→16
            # a) Up-convolution (upsample via transposed convolution): halves channels, doubles length
            self.ups.append(nn.ConvTranspose1d(feat*2, feat, kernel_size = 2, stride = 2))
            # b) ConvBlock: input has 2×feat channels (feat from up + feat skip)            
            self.ups.append(ConvBlock(feat*2, feat))

        # ===========================
        # Final output convolution
        # ===========================
        #    • Acts like a fully-connected layer applied at each time step.
        # ----------------------------------------------------------------------
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    # ════════════════════════════════════════════════════════════════════════
    # Forward pass: Encoder ➜ Bottleneck ➜ Decoder ➜ Classifier
    # ════════════════════════════════════════════════════════════════════════
    def forward(self, x):
        skip_stack = []         # will collect encoder outputs for skip connections

        # ---------------- Encoder ----------------
        for down in self.downs:
            x = down(x)                             # ConvBlock (keeps length)
            skip_stack.append(x)                    # save high-res features
            x = F.max_pool1d(x, kernel_size = 2)     # ↓2: halve length, double context
        
        # --------------- Bottleneck ---------------
        x = self.bottleneck(x)

        # Reverse list so the first pop corresponds to the last encoder block
        skip_stack = skip_stack[::-1]

        # ---------------- Decoder ----------------
        # Iterate pair-wise: (upconv, convblock), (upconv, convblock), ...
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)                # (a) learnable upsampling
            skip_conn = skip_stack[idx//2]      # (b) matching skip feature map
            
            # If the lengths differ by 1 (can happen with odd numbers),
            # right-pad the upsampled tensor so they match.
            if x.shape[-1] != skip_conn.shape[-1]:
                x = F.pad(x, (0, skip_conn.shape[-1] - x.shape[-1]))

            # Concatenate along channel dimension: [skip | upsampled]            
            x = torch.cat((skip_conn, x), dim = 1)
            x = self.ups[idx+1](x)
        
        x = self.final_conv(x)          # raw scores per class
        return F.softmax(x, dim = 1)    # convert to probabilities
    
# ----------------------------------------------
# Import Required Libraries
# ----------------------------------------------
import os
import dask.delayed
import numpy as np
from obspy import UTCDateTime
from obspy import read
from obspy.clients.fdsn.client import Client
from obspy.core.inventory.inventory import Inventory
from scipy import signal
import seisbench.models as sbm  # Import PhaseNet model from SeisBench
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
import dask
from io import BytesIO
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------
# Configure S3 Client for Public NCEDC Access
# ----------------------------------------------
s3 = boto3.client('s3', config = Config(signature_version = UNSIGNED), region_name='us-west-2')
BUCKET_NAME = 'ncedc-pds'

# -------------------------------------------------------------------------
# 📌 Utility Function to check if data is available in repository
# -------------------------------------------------------------------------
def station_data_exists(station, eq_time, pre_time, post_time, client: Client, network: str, location: str, channel: str,
                        s3_client, bucket_name: str) -> bool:

    # S3 check — build the same key you use in your main function
    day0 = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    jul   = day0.julday
    fname = f"{station.code}.{network}.{channel}..D.{day0.year}.{jul:03d}"
    key   = f"continuous_waveforms/{network}/{day0.year}/" \
            f"{day0.year}.{jul:03d}/{fname}"
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
    except ClientError:
        return False

    return True

# --------------------------------------------------------------
# 📌 Utility Function to Filter Inventory Based on Data Availability
# --------------------------------------------------------------
def filter_inventory(inventory: Inventory, eq_time, pre_time, post_time, 
                     client: Client, network: str, location: str, channel: str,
                     s3_client, bucket_name: str) -> Inventory:
    
    # iterate all networks
    kept_networks = []
    for net in inventory.networks:
        kept_stns = []
        for st in net.stations:
            if station_data_exists(st, eq_time, pre_time, post_time, client, network, location, channel,
                                   s3_client, bucket_name):
                kept_stns.append(st)
        if kept_stns:
            net.stations = kept_stns
            kept_networks.append(net)

    inventory.networks = kept_networks
    return inventory

# ----------------------------------------------
# Load Pretrained Phase Picker (PhaseNet)
# ----------------------------------------------
picker = sbm.PhaseNet.from_pretrained("original")

# --------------------------------------------------
# --------------------------------------------------
# ✂️ SEPERATE FUNCTION TO PROCESS STATIONS IN PARALLEL
# --------------------------------------------------
# --------------------------------------------------
def process_stations(station, start_time, pre_time, post_time, eq_time,
                     network, channel, inventory):
    station_code = station.code
    # -----------------------------------------------------------
    # Step 2️⃣: # Download waveforms & apply instrument correction
    # -----------------------------------------------------------        
    file_name = f'{station_code}.{network}.{channel}..D.{start_time.year}.{start_time.julday:03d}'
    KEY = f"continuous_waveforms/{network}/{start_time.year}/{start_time.year}.{start_time.julday:03d}/{file_name}"

    # stream the object from S3 and wrap in a BytesIO
    resp = s3.get_object(Bucket=BUCKET_NAME, Key=KEY)
    data_stream = resp['Body']              # this is a file-like StreamingBody
    buff = BytesIO(data_stream.read())      # read all bytes into an in-memory buffer
    buff.seek(0)                            # rewind to the front
    
    # now read directly from that buffer
    st_stream = read(buff, format='MSEED')
    st_stream.trim(starttime=eq_time - pre_time, endtime=eq_time + post_time)

    print(f"- Streamed {len(st_stream)} traces for station {station_code}.")

    if len(st_stream)<1:
        print(f"-- skipping station {station_code}")

    # Assuming single trace per station
    tr = st_stream[0]

    # Remove the instrument response to convert counts to ground displacement (in meters)
    tr.remove_response(inventory=inventory, output="DISP")

    # ----------------------------------------------
    # Step 3️⃣: Pick P-wave Arrivals & Slice Waveform
    # ----------------------------------------------
    picks = picker.classify(st_stream, batch_size=256, P_threshold=0.075, S_threshold=0.1).picks
    if not picks:
        raise Exception(f"- No picks found for station {station_code}.")
    
    # Use first P arrival time for plotting
    p_time = picks[0].peak_time

    manifest.append({
        'bucket': BUCKET_NAME,
        'key':    KEY,
        'pick_time': p_time, 'pre_time':  pre_time, 'post_time': post_time, 'eq_time':   eq_time,
        'station': station_code, 'sampling_rate':tr.stats.sampling_rate
    })

# ----------------------------------------------
# 🧰 Main Function: Seismogram Analysis Workflow
# ----------------------------------------------

def analyze_earthquake_manifest(eq_time, eq_lon, eq_lat, radius_km,
                       client_name='NCEDC', network='NC', location='*', channel='HNE',
                       pre_time=3, post_time=120, output_dir='plots/dask', use_dask=True):
    
    # Ensure eq_time is UTCDateTime
    if not isinstance(eq_time, UTCDateTime):
        eq_time = UTCDateTime(eq_time)

    # Define waveform time window
    start_time = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = eq_time.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Create output directory if it doesn't exist
    os.makedirs(f"{output_dir}", exist_ok=True)

    # ----------------------------------------------
    # Step 1️⃣: Retrieve Station Metadata
    # ----------------------------------------------
    client = Client(client_name)
    print("Making inventory of stations ...")
    inventory = client.get_stations(network=network, latitude=eq_lat, longitude=eq_lon,
                                    starttime=start_time, endtime=end_time, maxradius=radius_km/111.2, # Convert km to degrees
                                    location=location, channel=channel, level="response")
    
    print("Filtering the inventory ...")
    inventory = filter_inventory(inventory, eq_time, pre_time, post_time, client, network, location, channel, s3, BUCKET_NAME)

    stations = inventory[0].stations
    print(f"Found {len(stations)} stations within {radius_km} km of ({eq_lat}, {eq_lon}).")

    # ----------------------------------------------
    # Step 2️⃣: Loop Over Each Station
    # ----------------------------------------------
    for st in stations:
            try:
                process_stations(st, start_time, pre_time, post_time, eq_time,
                         network, channel, inventory)
            except:
                continue
    
    return manifest

# Custom Dataset class for seismic data with length normalization
class SeismicDataset(Dataset): # Inherits from torch.utils.data.Dataset
                               # This makes it compatible with PyTorch’s DataLoader for batching, shuffling, and parallel loading.
    '''
    Responsibility:
    - Load seismic waveforms (in MiniSEED format) from S3
    - Trim each waveform around an earthquake event
    - Normalize all to a common length
    - Produce a binary label sequence marking the P-wave arrival
    '''
    def __init__(self, manifest, window_size=5, target_length=None):
        """
        Args:
            manifest: List of processed station data
            window_size: Size of window around P-pick for positive labels (in seconds)
            target_length: Fixed length for all waveforms (if None, uses the median length)
        """
        self.data = []
        self.labels = []
        self.window_size = window_size
        
        if target_length is None:
            # Calculate a reasonable default based on the time window and typical sampling rates
            # Assuming typical sampling rate of 100 Hz and using the time window from manifest
            if manifest:
                # Use the first item to estimate typical duration
                sample_item = manifest[0]
                duration = sample_item['pre_time'] + sample_item['post_time']
                estimated_samples = int(duration * 100)  # Assume 100 Hz sampling rate
                self.target_length = estimated_samples
                print(f"Using estimated target length: {self.target_length} samples (based on {duration}s duration)")
            else:
                self.target_length = 3000  # Fallback default
                print(f"Using fallback target length: {self.target_length} samples")
        else:
            self.target_length = target_length
            print(f"Using provided target length: {self.target_length} samples")
        
        # Process each waveform
        actual_lengths = []
        for item in manifest:
            bucket_name   = item['bucket']
            key_name      = item['key']
            pick_time     = item['pick_time']
            eq_time       = item['eq_time']
            pre_time      = item['pre_time']
            post_time     = item['post_time']
            station_code  = item['station']
            
            # stream the object from S3 and wrap in a BytesIO
            resp = s3.get_object(Bucket=bucket_name, Key=key_name)
            data_stream = resp['Body']              # this is a file-like StreamingBody
            buff = BytesIO(data_stream.read())      # read all bytes into an in-memory buffer
            buff.seek(0)                            # rewind to the front
            
            # now read directly from that buffer
            st_stream = read(buff, format='MSEED')
            st_stream.trim(starttime=eq_time - pre_time, endtime=eq_time + post_time)

            print(f"- Streaming {len(st_stream)} trace for station {station_code}.")

            # Assuming single trace per station
            waveform = st_stream[0]
            sampling_rate = waveform.stats.sampling_rate
            actual_lengths.append(len(waveform.data))

            # Calculate pick sample index
            pick_offset = (pick_time - (eq_time - pre_time))  # seconds from start
            pick_sample = int(pick_offset * sampling_rate)
            
            # Normalize waveform length
            waveform_norm, label_norm = self._normalize_length(waveform.data, pick_sample, sampling_rate)
            
            self.data.append(waveform_norm)
            self.labels.append(label_norm)

        # Report actual length statistics after processing
        if actual_lengths:
            print(f"Actual length stats - Min: {min(actual_lengths)}, Max: {max(actual_lengths)}, Median: {int(np.median(actual_lengths))}")
            print(f"Target length used: {self.target_length}")
    
    def _normalize_length(self, waveform, pick_sample, sampling_rate):
        """
        Normalize waveform to target length by padding or truncating
        """
        current_length = len(waveform)
        
        if current_length == self.target_length:
            # Perfect match, no changes needed
            normalized_waveform = waveform.copy()
            normalized_pick_sample = pick_sample
            
        elif current_length > self.target_length:
            # Truncate: try to keep the pick near the center
            excess = current_length - self.target_length
            
            # Calculate how much to trim from start and end
            # Try to keep pick in the center portion
            if pick_sample < self.target_length // 2:
                # Pick is in first half, trim more from the end
                start_trim = max(0, excess // 4)
                end_trim = excess - start_trim
            elif pick_sample > current_length - self.target_length // 2:
                # Pick is in last half, trim more from the start
                end_trim = max(0, excess // 4)
                start_trim = excess - end_trim
            else:
                # Pick is in middle, trim equally from both ends
                start_trim = excess // 2
                end_trim = excess - start_trim
            
            normalized_waveform = waveform[start_trim:current_length-end_trim]
            normalized_pick_sample = pick_sample - start_trim
            
        else:
            # Pad: add zeros equally to both ends
            pad_needed = self.target_length - current_length
            pad_start = pad_needed // 2
            pad_end = pad_needed - pad_start
            
            normalized_waveform = np.pad(waveform, (pad_start, pad_end), mode='constant', constant_values=0)
            normalized_pick_sample = pick_sample + pad_start
        
        # Ensure the normalized waveform is exactly the target length
        assert len(normalized_waveform) == self.target_length, f"Length mismatch: {len(normalized_waveform)} != {self.target_length}"
        
        # Create labels (0 = background, 1 = P-wave)
        label = np.zeros(self.target_length)
        window_samples = int(self.window_size * sampling_rate / 2)  # ±window_size/2 seconds
        
        # Ensure pick sample is within bounds
        normalized_pick_sample = np.clip(normalized_pick_sample, 0, self.target_length - 1)
        
        start_idx = max(0, normalized_pick_sample - window_samples)
        end_idx = min(self.target_length, normalized_pick_sample + window_samples)
        label[start_idx:end_idx] = 1
        
        return normalized_waveform, label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        waveform = torch.FloatTensor(self.data[idx]).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor(self.labels[idx])
        return waveform, label
    
# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation
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
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# Evaluation function
def evaluate_picks(model, val_dataset, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    pick_errors = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            waveform, true_label = val_dataset[i]
            waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension
            
            output = model(waveform)
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

# Main training pipeline

# Example earthquakes (you can add more)
earthquakes = [
    {"time": '2022-12-20T10:34:24', "lat": 40.369, "lon": -124.588, "radius": 200}, # Frendale
    {"time": "2010-01-10T00:27:39", "lat": 40.652, "lon": -124.693, "radius": 200}, # Eureka
    #{"time": "2014-08-24T10:20:44", "lat": 38.215, "lon": -122.312, "radius": 500}, # South Napa
]

# Collect data from multiple earthquakes
all_manifest = []
for eq in earthquakes:
    print(f"\nProcessing earthquake: {eq['time']}")
    manifest = []
    manifest = analyze_earthquake_manifest(
        eq_time=eq['time'],
        eq_lat=eq['lat'],
        eq_lon=eq['lon'],
        radius_km=eq['radius']
    )
    all_manifest.extend(manifest)
    print(f"Collected {len(manifest)} stations for this earthquake")
    

print(f"\nTotal stations collected: {len(all_manifest)}")

if len(all_manifest) == 0:
    print("No data collected. Exiting.")

def train_test_split(data, test_size=0.2, random_state=None):
    """Fallback train_test_split implementation"""
    if random_state:
        np.random.seed(random_state)
    
    data_copy = data.copy()
    np.random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * (1 - test_size))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    
    return train_data, test_data

import torch
from tqdm import tqdm

# Split data into training and validation (80/20)
train_manifest, val_manifest = train_test_split(all_manifest, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_manifest)}")
print(f"Validation samples: {len(val_manifest)}")

# Create datasets
train_dataset = SeismicDataset(train_manifest, window_size=5)
val_dataset = SeismicDataset(val_manifest, window_size=5)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize model
model = UNet1D(in_channels=1, out_channels=2)

# Train model
print("\nStarting training...")
train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=20)

# Evaluate model
print("\nEvaluating model...")
pick_errors = evaluate_picks(model, val_dataset)

# Assuming 100 Hz sampling rate for error calculation
sampling_rate = 100  # Hz
pick_errors_seconds = pick_errors / sampling_rate

print(f"\nPick Time Accuracy Results:")
print(f"Mean absolute error: {np.mean(pick_errors_seconds):.3f} seconds")
print(f"Standard deviation: {np.std(pick_errors_seconds):.3f} seconds")
print(f"Median absolute error: {np.median(pick_errors_seconds):.3f} seconds")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(pick_errors_seconds, bins=30, alpha=0.7)
plt.xlabel('Pick Time Error (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Pick Time Errors')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Save model
torch.save(model.state_dict(), 'unet_phase_picker.pth')
print("Model saved as 'unet_phase_picker.pth'")

# Add this code at the end of your script, after the model training and evaluation

def plot_seismogram_with_picks(model, dataset, manifest, num_examples=3, save_path='seismogram_picks.png'):
    """
    Updated visualization function that works with normalized length data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 2, figsize=(16, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            waveform, true_label = dataset[i]
            
            # Find corresponding manifest entry (this is tricky with normalization)
            # For simplicity, we'll use the index, but note that some entries might be missing
            # if they were filtered out during processing
            if i < len(manifest):
                station_info = manifest[i]
            else:
                print(f"Warning: No manifest entry for dataset index {i}")
                continue
            
            # Get model prediction
            waveform_batch = waveform.unsqueeze(0).to(device)  # Add batch dimension
            output = model(waveform_batch)
            prob = output[0, 1, :].cpu().numpy()  # P-wave probability
            
            # Calculate time axes for normalized data
            sampling_rate = station_info['sampling_rate']
            pre_time = station_info['pre_time']
            post_time = station_info['post_time']
            
            # Time vector for normalized data
            total_samples = len(waveform[0])
            time_vector = np.linspace(-pre_time, post_time, total_samples)
            
            # Find pick locations in normalized data
            true_pick_samples = np.where(true_label == 1)[0]
            model_pick_sample = np.argmax(prob)
            
            # Plot waveform
            ax1 = axes[i, 0]
            waveform_data = waveform[0].numpy()
            ax1.plot(time_vector, waveform_data, 'k-', linewidth=0.8, alpha=0.8)
            
            # Add picks
            if len(true_pick_samples) > 0:
                true_pick_time = time_vector[int(np.mean(true_pick_samples))]
                ax1.axvline(true_pick_time, color='red', linestyle='--', linewidth=2, 
                           label=f'True Pick Window Center ({true_pick_time:.2f}s)')
            
            model_pick_time = time_vector[model_pick_sample]
            ax1.axvline(model_pick_time, color='blue', linestyle='-', linewidth=2, 
                       label=f'Model Pick ({model_pick_time:.2f}s)')
            
            # Add earthquake time reference
            ax1.axvline(0, color='orange', linestyle=':', linewidth=1, alpha=0.7, 
                       label='Earthquake Time')
            
            ax1.set_xlabel('Time relative to earthquake (s)')
            ax1.set_ylabel('Amplitude (normalized)')
            ax1.set_title(f'Station {station_info["station"]} - Normalized Waveform')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot probability
            ax2 = axes[i, 1]
            ax2.plot(time_vector, prob, 'b-', linewidth=2, label='P-wave Probability')
            ax2.fill_between(time_vector, 0, prob, alpha=0.3, color='blue')
            
            # Add picks to probability plot
            if len(true_pick_samples) > 0:
                ax2.axvline(true_pick_time, color='red', linestyle='--', linewidth=2, 
                           label='True Pick Window Center')
                # Highlight true pick window
                window_start = time_vector[true_pick_samples[0]]
                window_end = time_vector[true_pick_samples[-1]]
                ax2.axvspan(window_start, window_end, alpha=0.2, color='green', 
                           label='True Pick Window')
            
            ax2.axvline(model_pick_time, color='blue', linestyle='-', linewidth=2, 
                       label='Model Pick')
            ax2.axvline(0, color='orange', linestyle=':', linewidth=1, alpha=0.7, 
                       label='Earthquake Time')
            
            ax2.set_xlabel('Time relative to earthquake (s)')
            ax2.set_ylabel('P-wave Probability')
            ax2.set_title(f'Station {station_info["station"]} - Model Output')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Seismogram plots saved to {save_path}")

def create_summary_plot(model, val_dataset, val_manifest, save_path='pick_summary.png'):
    """
    Create a summary plot showing pick accuracy statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    pick_errors = []
    original_picks = []
    model_picks = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            waveform, true_label = val_dataset[i]
            station_info = val_manifest[i]
            
            # Get model prediction
            waveform_batch = waveform.unsqueeze(0).to(device)
            output = model(waveform_batch)
            prob = output[0, 1, :].cpu().numpy()
            
            # Calculate time vector
            sampling_rate = station_info['sampling_rate']
            eq_time = station_info['eq_time']
            pre_time = station_info['pre_time']
            post_time = station_info['post_time']
            total_samples = len(waveform[0])
            time_vector = np.linspace(-pre_time, post_time, total_samples)
            
            # Original and model picks
            original_pick_time = station_info['pick_time']
            original_pick_offset = original_pick_time - eq_time
            model_pick_sample = np.argmax(prob)
            model_pick_time = time_vector[model_pick_sample]
            
            pick_error = abs(model_pick_time - original_pick_offset)
            pick_errors.append(pick_error)
            original_picks.append(original_pick_offset)
            model_picks.append(model_pick_time)
    
    pick_errors = np.array(pick_errors)
    original_picks = np.array(original_picks)
    model_picks = np.array(model_picks)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error histogram
    axes[0, 0].hist(pick_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(pick_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(pick_errors):.3f}s')
    axes[0, 0].axvline(np.median(pick_errors), color='orange', linestyle='--', 
                      label=f'Median: {np.median(pick_errors):.3f}s')
    axes[0, 0].set_xlabel('Pick Error (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Pick Errors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Original vs Model picks scatter plot
    axes[0, 1].scatter(original_picks, model_picks, alpha=0.6, s=50)
    min_pick = min(min(original_picks), min(model_picks))
    max_pick = max(max(original_picks), max(model_picks))
    axes[0, 1].plot([min_pick, max_pick], [min_pick, max_pick], 'r--', 
                   label='Perfect Agreement')
    axes[0, 1].set_xlabel('Original Pick Time (s)')
    axes[0, 1].set_ylabel('Model Pick Time (s)')
    axes[0, 1].set_title('Original vs Model Picks')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error vs Original pick time
    axes[1, 0].scatter(original_picks, pick_errors, alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Original Pick Time (s)')
    axes[1, 0].set_ylabel('Pick Error (s)')
    axes[1, 0].set_title('Error vs Original Pick Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative error distribution
    sorted_errors = np.sort(pick_errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 1].plot(sorted_errors, cumulative_prob, 'b-', linewidth=2)
    axes[1, 1].axvline(np.percentile(pick_errors, 90), color='red', linestyle='--', 
                      label=f'90th percentile: {np.percentile(pick_errors, 90):.3f}s')
    axes[1, 1].axvline(np.percentile(pick_errors, 95), color='orange', linestyle='--', 
                      label=f'95th percentile: {np.percentile(pick_errors, 95):.3f}s')
    axes[1, 1].set_xlabel('Pick Error (seconds)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Summary plot saved to {save_path}")
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("DETAILED PICK ACCURACY STATISTICS")
    print("="*50)
    print(f"Total validation samples: {len(pick_errors)}")
    print(f"Mean absolute error: {np.mean(pick_errors):.4f} ± {np.std(pick_errors):.4f} seconds")
    print(f"Median absolute error: {np.median(pick_errors):.4f} seconds")
    print(f"90th percentile error: {np.percentile(pick_errors, 90):.4f} seconds")
    print(f"95th percentile error: {np.percentile(pick_errors, 95):.4f} seconds")
    print(f"Maximum error: {np.max(pick_errors):.4f} seconds")
    print(f"Minimum error: {np.min(pick_errors):.4f} seconds")
    
    # Performance categories
    excellent = np.sum(pick_errors < 0.1)
    good = np.sum((pick_errors >= 0.1) & (pick_errors < 0.5))
    fair = np.sum((pick_errors >= 0.5) & (pick_errors < 1.0))
    poor = np.sum(pick_errors >= 1.0)
    
    print(f"\nPerformance Categories:")
    print(f"  Excellent (< 0.1s): {excellent:3d} ({excellent/len(pick_errors)*100:.1f}%)")
    print(f"  Good (0.1-0.5s):   {good:3d} ({good/len(pick_errors)*100:.1f}%)")
    print(f"  Fair (0.5-1.0s):   {fair:3d} ({fair/len(pick_errors)*100:.1f}%)")
    print(f"  Poor (> 1.0s):     {poor:3d} ({poor/len(pick_errors)*100:.1f}%)")
    print("="*50)

# Add this to the end of your main script after model evaluation:

print("\nCreating visualization plots...")

# Plot individual seismograms with picks
plot_seismogram_with_picks(model, val_dataset, val_manifest, num_examples=5, 
                          save_path='seismogram_picks.png')

# Create summary statistics plot
create_summary_plot(model, val_dataset, val_manifest, save_path='pick_summary.png')

print("\nVisualization complete!")