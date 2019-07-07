CONFIG = {
    'model_architecture': {
        'num_poolings': 3,
        'num_filters': 64,
        'input_shape': (256, 256, 1)
    },
    'training_config': {
        'checkpoint-dir': 'model/checkpoints',
        'num_epochs': 1000,
        'steps_per_epoch': 10,
        'learning_rate': 0.001,
        'learning_rate_decay': 0.0005
    },
    'input_config': {
        'training_data_dir': '../training-data',
        'batch_size': 4,
        'num_parallel_map_calls': 4
    }
}