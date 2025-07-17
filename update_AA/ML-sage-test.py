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
    │ • Keeps the time length unchanged (padding=1) but learns richer representations (more channels). |                                    │
    │ • Re-using the same pattern everywhere keeps the code short and consistent.                      |                                    │
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
        conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
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
                 features = [16, 32, 64, 182]   # network width; doubles every step by default
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
            x = F.maxpool1d(x, kernel_size = 2)     # ↓2: halve length, double context
        
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

    print(f"- Downloaded {len(st_stream)} traces for station {station_code}.")

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
        'pick_time': p_time, 'pre_time':  pre_time, 'post_time': post_time, 'eq_time':   eq_time
    })

# ----------------------------------------------
# 🧰 Main Function: Seismogram Analysis Workflow
# ----------------------------------------------
manifest = []
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
    # Step 2️⃣: Loop Over Each Station using DASK
    # ----------------------------------------------

    for st in stations:
            process_stations(st, start_time, pre_time, post_time, eq_time,
                     network, channel, inventory)
    
    return manifest