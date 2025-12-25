"""
NOAA Visibility Data Loader
===========================

This module implements a data loader for NOAA (National Oceanic and Atmospheric
Administration) visibility and weather station data.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings


class NOAAVisibilityLoader(Dataset):
    """
    Dataset loader for NOAA multi-station visibility data.
    
    This loader processes NOAA weather station data including visibility,
    temperature, humidity, wind speed, and other meteorological variables.
    
    Args:
        data_path: Path to NOAA data file (CSV or directory)
        stations: List of station IDs to include
        seq_len: Length of input sequence
        pred_len: Length of prediction horizon
        features: List of feature names to include
        split: Data split - 'train', 'val', or 'test'
        normalize: Whether to normalize features
        include_time_features: Whether to include temporal features
    """
    
    def __init__(
        self,
        data_path: str,
        stations: Optional[List[str]] = None,
        seq_len: int = 24,
        pred_len: int = 1,
        features: Optional[List[str]] = None,
        split: str = 'train',
        normalize: bool = True,
        include_time_features: bool = True
    ):
        super(NOAAVisibilityLoader, self).__init__()
        
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.normalize = normalize
        self.include_time_features = include_time_features
        
        # Default features if not specified
        if features is None:
            self.features = [
                'visibility',
                'temperature',
                'humidity',
                'wind_speed',
                'pressure',
                'precipitation',
                'cloud_cover'
            ]
        else:
            self.features = features
        
        # Load and process data
        self.data, self.stations = self._load_data(stations)
        self.num_stations = len(self.stations)
        
        # Compute statistics for normalization
        if self.normalize:
            self.mean, self.std = self._compute_statistics()
        
        # Create samples
        self.samples = self._create_samples()
        
        print(f"Loaded {len(self.samples)} samples for {self.num_stations} stations in {split} split")
    
    def _load_data(self, stations: Optional[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load NOAA data from file or directory.
        
        Returns:
            DataFrame with multi-station data and list of station IDs
        """
        if self.data_path.is_file():
            # Load single file
            if self.data_path.suffix == '.csv':
                data = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        elif self.data_path.is_dir():
            # Load multiple files from directory
            csv_files = list(self.data_path.glob('*.csv'))
            if not csv_files:
                raise ValueError(f"No CSV files found in {self.data_path}")
            
            dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                dfs.append(df)
            data = pd.concat(dfs, ignore_index=True)
        else:
            # Generate synthetic data for demonstration
            warnings.warn(f"Data path {self.data_path} not found. Generating synthetic data.")
            data = self._generate_synthetic_data(stations)
        
        # Parse datetime if present
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.sort_values('datetime')
        elif 'date' in data.columns or 'time' in data.columns:
            # Try to create datetime from separate columns
            if 'date' in data.columns and 'time' in data.columns:
                data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
            elif 'date' in data.columns:
                data['datetime'] = pd.to_datetime(data['date'])
            data = data.sort_values('datetime')
        
        # Filter stations if specified
        if stations is not None:
            if 'station_id' in data.columns:
                data = data[data['station_id'].isin(stations)]
                available_stations = stations
            else:
                warnings.warn("station_id column not found, treating as single station")
                available_stations = ['station_0']
        else:
            if 'station_id' in data.columns:
                available_stations = sorted(data['station_id'].unique().tolist())
            else:
                available_stations = ['station_0']
        
        return data, available_stations
    
    def _generate_synthetic_data(self, stations: Optional[List[str]]) -> pd.DataFrame:
        """Generate synthetic NOAA-like data for demonstration."""
        if stations is None:
            stations = [f'station_{i}' for i in range(10)]
        
        n_timesteps = 1000
        n_stations = len(stations)
        
        # Generate synthetic time series
        dates = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='H')
        
        data_list = []
        for station in stations:
            # Generate synthetic features with some correlation
            visibility = 10 + 5 * np.sin(np.linspace(0, 4*np.pi, n_timesteps)) + np.random.randn(n_timesteps) * 2
            visibility = np.clip(visibility, 0, 20)
            
            temperature = 15 + 10 * np.sin(np.linspace(0, 4*np.pi, n_timesteps)) + np.random.randn(n_timesteps) * 3
            humidity = 60 + 20 * np.cos(np.linspace(0, 4*np.pi, n_timesteps)) + np.random.randn(n_timesteps) * 5
            humidity = np.clip(humidity, 0, 100)
            
            wind_speed = 5 + 3 * np.abs(np.sin(np.linspace(0, 8*np.pi, n_timesteps))) + np.random.randn(n_timesteps) * 1
            wind_speed = np.clip(wind_speed, 0, None)
            
            pressure = 1013 + 10 * np.sin(np.linspace(0, 2*np.pi, n_timesteps)) + np.random.randn(n_timesteps) * 5
            precipitation = np.maximum(0, np.random.randn(n_timesteps) * 2)
            cloud_cover = np.clip(50 + 30 * np.cos(np.linspace(0, 6*np.pi, n_timesteps)) + np.random.randn(n_timesteps) * 10, 0, 100)
            
            station_df = pd.DataFrame({
                'datetime': dates,
                'station_id': station,
                'visibility': visibility,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'pressure': pressure,
                'precipitation': precipitation,
                'cloud_cover': cloud_cover
            })
            data_list.append(station_df)
        
        return pd.concat(data_list, ignore_index=True)
    
    def _compute_statistics(self) -> Tuple[Dict, Dict]:
        """Compute mean and std for normalization."""
        mean = {}
        std = {}
        
        for feature in self.features:
            if feature in self.data.columns:
                mean[feature] = self.data[feature].mean()
                std[feature] = self.data[feature].std()
                if std[feature] == 0:
                    std[feature] = 1.0
            else:
                mean[feature] = 0.0
                std[feature] = 1.0
        
        return mean, std
    
    def _create_samples(self) -> List[Dict]:
        """Create training samples from data."""
        samples = []
        
        # Determine split ratios
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        
        # Group data by datetime
        if 'datetime' in self.data.columns:
            time_groups = self.data.groupby('datetime')
            unique_times = sorted(time_groups.groups.keys())
        else:
            # If no datetime, create sequential indices
            n_total = len(self.data) // self.num_stations
            unique_times = list(range(n_total))
        
        n_times = len(unique_times)
        
        # Split data
        train_end = int(n_times * split_ratios['train'])
        val_end = int(n_times * (split_ratios['train'] + split_ratios['val']))
        
        if self.split == 'train':
            time_range = unique_times[:train_end]
        elif self.split == 'val':
            time_range = unique_times[train_end:val_end]
        else:  # test
            time_range = unique_times[val_end:]
        
        # Create samples
        for i in range(len(time_range) - self.seq_len - self.pred_len + 1):
            sample = {
                'start_idx': i,
                'time_range': time_range[i:i + self.seq_len + self.pred_len]
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            x: Input features [seq_len, num_stations, num_features]
            y: Target visibility [pred_len, num_stations]
            adj_matrix: Adjacency matrix [num_stations, num_stations]
        """
        sample = self.samples[idx]
        time_range = sample['time_range']
        
        # Extract features for all stations
        station_data = []
        for station in self.stations:
            if 'station_id' in self.data.columns:
                station_df = self.data[
                    (self.data['station_id'] == station) &
                    (self.data['datetime'].isin(time_range) if 'datetime' in self.data.columns else 
                     self.data.index.isin(time_range))
                ]
            else:
                station_df = self.data.iloc[time_range]
            
            # Extract features
            features = []
            for feature in self.features:
                if feature in station_df.columns:
                    feat_values = station_df[feature].values
                else:
                    feat_values = np.zeros(len(time_range))
                
                # Normalize if requested
                if self.normalize:
                    feat_values = (feat_values - self.mean[feature]) / self.std[feature]
                
                features.append(feat_values)
            
            station_data.append(np.stack(features, axis=-1))
        
        # Stack all stations
        all_data = np.stack(station_data, axis=1)  # [time, num_stations, num_features]
        
        # Split into input and target
        x = all_data[:self.seq_len]
        y = all_data[self.seq_len:self.seq_len + self.pred_len, :, 0]  # Visibility only
        
        # Create adjacency matrix (distance-based or fully connected)
        adj_matrix = self._create_adjacency_matrix()
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(adj_matrix)
        )
    
    def _create_adjacency_matrix(self) -> np.ndarray:
        """
        Create adjacency matrix for stations.
        
        For now, creates a fully connected graph. In practice, this could be
        based on geographic distances or correlations.
        """
        adj_matrix = np.ones((self.num_stations, self.num_stations))
        np.fill_diagonal(adj_matrix, 0)
        
        # Normalize
        row_sum = adj_matrix.sum(axis=1, keepdims=True)
        adj_matrix = adj_matrix / (row_sum + 1e-6)
        
        return adj_matrix


def create_noaa_dataloader(
    data_path: str,
    batch_size: int = 32,
    split: str = 'train',
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for NOAA visibility data.
    
    Args:
        data_path: Path to data
        batch_size: Batch size
        split: Data split
        **kwargs: Additional arguments for NOAAVisibilityLoader
    
    Returns:
        DataLoader instance
    """
    dataset = NOAAVisibilityLoader(data_path, split=split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader
