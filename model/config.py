CONFIG = {
    'model_architecture': {
        'num_poolings': 2,
        'num_filters': 16
    },
    'training_config': {
        'training_data_dir': 'training-data',
        'checkpoint-dir': 'model/checkpoints',
        'num_epochs': 4,
        'num_iterations': 100,
        'learning_rate': 0.001,
        'learning_rate_decay': 0.01
    },
    'input_config': {
        'batch_size': 16,
        'num_parallel_map_calls': 4
    }
}