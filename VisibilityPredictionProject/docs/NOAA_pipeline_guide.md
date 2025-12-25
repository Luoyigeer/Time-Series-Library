# NOAA Data Pipeline Guide

## Overview

This guide provides detailed instructions for setting up and running the NOAA-based visibility prediction pipeline.

## Data Sources

### NOAA Weather Stations

The project is designed to work with NOAA Integrated Surface Database (ISD) data, which contains:
- Hourly weather observations
- Visibility measurements
- Temperature, humidity, wind, pressure
- Precipitation and cloud cover

### Obtaining NOAA Data

#### Option 1: Official NOAA FTP Server

```bash
# Download from NOAA FTP
wget ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/2023/*.gz

# Extract files
gunzip *.gz
```

#### Option 2: NOAA API

```python
import requests

# Example API call
station_id = "72530014819"  # Chicago O'Hare
year = 2023
url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"
response = requests.get(url)
```

#### Option 3: Use Synthetic Data

The project automatically generates synthetic data if real NOAA data is not available:

```python
from VisibilityPredictionProject.data_loader import NOAAVisibilityLoader

# This will create synthetic data
dataset = NOAAVisibilityLoader(
    data_path='./data/nonexistent',
    num_stations=10
)
```

## Data Format

### Required CSV Format

The NOAA loader expects CSV files with the following structure:

```csv
datetime,station_id,visibility,temperature,humidity,wind_speed,pressure,precipitation,cloud_cover
2023-01-01 00:00:00,KORD,10.0,15.2,65.0,5.5,1013.2,0.0,50.0
2023-01-01 01:00:00,KORD,9.5,14.8,67.0,6.0,1013.0,0.0,55.0
...
```

### Column Descriptions

| Column | Description | Unit | Range |
|--------|-------------|------|-------|
| datetime | Timestamp | ISO 8601 | - |
| station_id | Station identifier | String | - |
| visibility | Horizontal visibility | km | 0-20+ |
| temperature | Air temperature | °C | -50 to 50 |
| humidity | Relative humidity | % | 0-100 |
| wind_speed | Wind speed | m/s | 0-50 |
| pressure | Atmospheric pressure | hPa | 950-1050 |
| precipitation | Precipitation | mm | 0+ |
| cloud_cover | Cloud coverage | % | 0-100 |

## Data Preprocessing

### 1. Data Cleaning

```python
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('noaa_raw.csv')

# Handle missing values
df['visibility'] = df['visibility'].fillna(df['visibility'].mean())

# Remove outliers
df = df[(df['visibility'] >= 0) & (df['visibility'] <= 50)]
df = df[(df['temperature'] >= -50) & (df['temperature'] <= 50)]

# Ensure proper datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Save cleaned data
df.to_csv('noaa_clean.csv', index=False)
```

### 2. Feature Engineering

```python
from VisibilityPredictionProject.data_loader.data_utils import create_time_features

# Add temporal features
time_features = create_time_features(df['datetime'].values)

# Add derived features
df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
df['wind_pressure_ratio'] = df['wind_speed'] / df['pressure']

# Normalize features
from VisibilityPredictionProject.data_loader.data_utils import normalize_features

normalized_data, mean, std = normalize_features(
    df[['visibility', 'temperature', 'humidity']].values
)
```

### 3. Creating Train/Val/Test Splits

```python
from VisibilityPredictionProject.data_loader import NOAAVisibilityLoader

# Automatically handles splitting
train_dataset = NOAAVisibilityLoader(
    data_path='./data/noaa_clean.csv',
    split='train',  # 70% of data
    seq_len=24,
    pred_len=1
)

val_dataset = NOAAVisibilityLoader(
    data_path='./data/noaa_clean.csv',
    split='val',  # 15% of data
    seq_len=24,
    pred_len=1
)

test_dataset = NOAAVisibilityLoader(
    data_path='./data/noaa_clean.csv',
    split='test',  # 15% of data
    seq_len=24,
    pred_len=1
)
```

## Complete Pipeline Example

### Step 1: Data Preparation

```bash
# Create data directory
mkdir -p data/noaa_visibility

# Download and prepare data (or use synthetic)
python scripts/prepare_noaa_data.py \
    --output_dir ./data/noaa_visibility
```

### Step 2: Train Multi-Station Model

```bash
cd VisibilityPredictionProject/scripts

python train_multi_station.py \
    --data_path ../../data/noaa_visibility \
    --model STGNNVisibility \
    --num_stations 10 \
    --seq_len 24 \
    --pred_len 1 \
    --hidden_dim 128 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --save_dir ../../checkpoints/noaa_multistation
```

### Step 3: Evaluate Model

```python
import torch
from VisibilityPredictionProject.models import STGNNVisibility
from VisibilityPredictionProject.data_loader import create_noaa_dataloader
from VisibilityPredictionProject.utils import evaluate_model, plot_predictions

# Load model
model = STGNNVisibility(num_stations=10)
checkpoint = torch.load('./checkpoints/noaa_multistation/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create test loader
test_loader = create_noaa_dataloader(
    './data/noaa_visibility',
    batch_size=32,
    split='test'
)

# Evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

metrics = evaluate_model(
    model, test_loader, device,
    return_predictions=True
)

print(f"Test MAE: {metrics['mae']:.4f} km")
print(f"Test RMSE: {metrics['rmse']:.4f} km")
print(f"Test R²: {metrics['r2']:.4f}")

# Visualize predictions
plot_predictions(
    metrics['predictions'],
    metrics['targets'],
    save_path='./results/test_predictions.png'
)
```

### Step 4: Inference on New Data

```python
import torch
import pandas as pd
import numpy as np

# Load model
model = STGNNVisibility(num_stations=10)
checkpoint = torch.load('./checkpoints/noaa_multistation/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare new data
# Format: [batch_size=1, seq_len=24, num_stations=10, num_features=7]
new_data = prepare_input_data(...)  # Your data preparation function

# Make prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
new_data = torch.FloatTensor(new_data).to(device)

with torch.no_grad():
    predictions = model(new_data)

print(f"Predicted visibility for next hour: {predictions[0].cpu().numpy()}")
```

## Station Selection

### Choosing Weather Stations

Consider these factors when selecting stations:
1. **Geographic diversity**: Spread across different climate zones
2. **Data quality**: Stations with consistent, high-quality measurements
3. **Temporal coverage**: Long historical records
4. **Spatial proximity**: Balance between local and regional patterns

### Example Station Lists

#### Major US Airports (High Data Quality)
```python
major_airports = [
    'KORD',  # Chicago O'Hare
    'KJFK',  # New York JFK
    'KLAX',  # Los Angeles
    'KATL',  # Atlanta
    'KDFW',  # Dallas/Fort Worth
    'KDEN',  # Denver
    'KSFO',  # San Francisco
    'KSEA',  # Seattle
    'KBOS',  # Boston
    'KMIA'   # Miami
]
```

#### Regional Network (California Example)
```python
california_stations = [
    'KSFO',  # San Francisco
    'KLAX',  # Los Angeles
    'KSAN',  # San Diego
    'KSAC',  # Sacramento
    'KSJC',  # San Jose
    'KOAK',  # Oakland
    'KBUR',  # Burbank
    'KSNA',  # Orange County
]
```

## Troubleshooting

### Common Issues

#### Issue 1: Missing Data
```python
# Check for missing values
df.isnull().sum()

# Interpolate missing values
df['visibility'] = df['visibility'].interpolate(method='linear')
```

#### Issue 2: Inconsistent Timestamps
```python
# Resample to hourly frequency
df = df.set_index('datetime')
df = df.resample('H').mean()
df = df.reset_index()
```

#### Issue 3: Station Data Mismatch
```python
# Ensure all stations have same time range
from VisibilityPredictionProject.data_loader.data_utils import prepare_graph_data

# This function handles alignment
data, adj_matrix = prepare_graph_data(
    raw_data,
    num_stations=10,
    adjacency_type='correlation'
)
```

## Performance Tips

1. **Use GPU acceleration**: Set `device='cuda'` for faster training
2. **Batch size tuning**: Larger batches (64-128) can improve training speed
3. **Data caching**: Cache preprocessed data to disk
4. **Parallel data loading**: Use `num_workers > 0` in DataLoader

## Next Steps

- [Train single-image visibility model](./single_image_guide.md)
- [Apply transfer learning](./transfer_learning_guide.md)
- [Model optimization guide](./optimization_guide.md)
