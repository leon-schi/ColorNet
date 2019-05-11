CONFIG = {
    'model_architecture': {
        'num_poolings': 3,
        'num_filters': 64
    },
    'training_config': {
        'checkpoint-dir': 'model/checkpoints',
        'num_epochs': 500,
        'num_iterations': 4000,
        'learning_rate': 0.001,
        'learning_rate_decay': 0.0005
    },
    'input_config': {
        'training_data_dir': '../training-data',
        'batch_size': 4,
        'num_parallel_map_calls': 4
    }
}