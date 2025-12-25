"""
Default Configuration for Visibility Prediction Models
======================================================

This file contains default hyperparameters and settings.
"""

# Model configurations
MODEL_CONFIGS = {
    'GraphVisibilityNet': {
        'num_stations': 10,
        'input_dim': 7,
        'hidden_dim': 64,
        'output_dim': 1,
        'num_layers': 3,
        'dropout': 0.1
    },
    
    'STGNNVisibility': {
        'num_stations': 10,
        'input_dim': 7,
        'hidden_dim': 128,
        'output_dim': 1,
        'num_spatial_layers': 2,
        'num_temporal_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    },
    
    'VisibilityVisionNet': {
        'num_classes': 1,
        'pretrained': False,
        'dropout': 0.3
    },
    
    'TransferLearningAdapter': {
        'gnn_feature_dim': 128,
        'vision_feature_dim': 128,
        'use_adversarial': True,
        'freeze_gnn': True,
        'dropout': 0.2
    }
}

# Data configurations
DATA_CONFIGS = {
    'noaa': {
        'seq_len': 24,
        'pred_len': 1,
        'features': [
            'visibility',
            'temperature',
            'humidity',
            'wind_speed',
            'pressure',
            'precipitation',
            'cloud_cover'
        ],
        'normalize': True,
        'include_time_features': True
    },
    
    'image': {
        'image_size': (224, 224),
        'normalize': True,
        'augment': True
    }
}

# Training configurations
TRAINING_CONFIGS = {
    'multi_station': {
        'batch_size': 32,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'patience': 10,
        'max_grad_norm': 1.0
    },
    
    'single_image': {
        'batch_size': 32,
        'epochs': 50,
        'lr': 0.0001,
        'weight_decay': 1e-4,
        'patience': 10,
        'max_grad_norm': 1.0
    },
    
    'transfer_learning': {
        'batch_size': 32,
        'epochs': 50,
        'lr': 0.0001,
        'weight_decay': 1e-4,
        'patience': 10,
        'max_grad_norm': 1.0,
        'adversarial_weight': 0.1
    }
}

# Evaluation configurations
EVAL_CONFIGS = {
    'metrics': ['mae', 'rmse', 'mape', 'r2', 'correlation'],
    'save_predictions': True,
    'visualize': True
}

# Path configurations
PATH_CONFIGS = {
    'data_dir': './data',
    'checkpoint_dir': './checkpoints',
    'results_dir': './results',
    'log_dir': './logs'
}
