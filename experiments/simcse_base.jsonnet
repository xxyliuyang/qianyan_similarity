local model_name = "./resources/robert_pytorch";
local train_data ='./data/simcse/train.json';
local dev_data = './data/simcse/dev.json';

{
    "train_data_path": train_data,
    "validation_data_path": dev_data,

    "dataset_reader": {
        "type": "simcse",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "add_special_tokens":true,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "tokens"
            }
        },
        "max_length": 500,
    },

    "pytorch_seed": 42,
    "numpy_seed": 42,
    "random_seed": 42,

    "model": {
        "type": "simcse",
        "model_path": model_name,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64,
        },
    },
    "validation_data_loader": {
        "batch_size": 64,
        "shuffle": false
    },

    "trainer": {
        "num_epochs": 5,
        "use_amp": true,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay":0.01,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac":0.1
        },
        "grad_norm":1.0,
        "cuda_device":1,
        "validation_metric":"+accuracy",
        "checkpointer":{
            "num_serialized_models_to_keep": -1,
        },
    }
}
