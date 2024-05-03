import os
import ray
import logging

# Setting environment variables for Ray configuration
os.environ['RAY_memory_usage_threshold'] = '0.99'
# os.environ['RAY_memory_monitor_refresh_ms'] = '0'

# Initialize Ray with a higher logging level
ray.init(logging_level=logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# config files
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import flwr as fl
import pickle
from pathlib import Path

# self-defined modules
from dataset import prepare_dataset, prepare_dataloaders
from client import generate_client_fn
from server import get_on_fit_config, SaveModelStrategy, fit_metrics_aggregation, evaluate_metrics_aggregation
from model import model_dict

import torch
import numpy as np
import random

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    # Ensures deterministic behavior when using convolutional layers (at the cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load the config in config/base.yaml
@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Ensure reproducibility
    set_seed(cfg.seed)
    
    # 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    # 2. Prepare your dataset
    raw_datasets = prepare_dataset(feature_type='raw_normalized', class_type='binary', overlap_percent=75)
    raw_trainloaders, raw_valloaders, raw_testloaders = prepare_dataloaders(
        raw_datasets, cfg.batch_size, cfg.test_ratio, cfg.seed
    )
    
    grid_datasets = prepare_dataset(feature_type='de_grid', class_type='binary', overlap_percent=75)
    grid_trainloaders, grid_valloaders, grid_testloaders = prepare_dataloaders(
        grid_datasets, cfg.batch_size, cfg.test_ratio, cfg.seed
    )
    
    trainloaders = list(zip(raw_trainloaders, grid_trainloaders))
    valloaders = list(zip(raw_valloaders, grid_valloaders))
    testloaders = list(zip(raw_testloaders, grid_testloaders))
    
    # 3. Define your clients
    # base_model = instantiate(cfg.config_model)
    server_model = model_dict[cfg.model_type]
    client_fn = generate_client_fn(trainloaders, valloaders, testloaders, server_model, client_models_folder=cfg.client_models_folder)

    # 4. Define your strategy
    num_clients = 3
    strategy = SaveModelStrategy(
        # save model
        base_model=server_model,
        save_path=save_path,
        max_rounds=cfg.num_rounds,
        # FL strategy
        fraction_fit=1,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=num_clients,  # number of clients to sample for fit()
        fraction_evaluate=1,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=num_clients,  # number of clients to sample for evaluate()
        min_available_clients=num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        # evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # a function to run on the server side to evaluate the global model.
    
    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        #ray_init_args={'num_cpus': 8, 'num_gpus': 1},
        client_resources={"num_cpus": 6, "num_gpus": 0.5},
    )

    # 6. Save your results
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
