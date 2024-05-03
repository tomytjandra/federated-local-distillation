from typing import List, Tuple, Union, Optional, Dict
from collections import OrderedDict

from omegaconf import DictConfig

import torch
import flwr as fl
import numpy as np
from flwr.common import Metrics

# from model import Net, test

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "current_round": server_round,
            "mode": config.mode,
            "lr": config.lr,
            #"weight_decay": config.weight_decay,
            "local_epochs": config.local_epochs,
            "temperature": config.temperature,
            "alpha": config.alpha,
        }

    return fit_config_fn

# Define metric aggregation function
def weighted_average2(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["mean_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    dataset_metrics = {}

    for num_examples, m in metrics:
        dataset_name = m['dataset_name']
        if dataset_name not in dataset_metrics:
            dataset_metrics[dataset_name] = {'total_examples': 0}

        dataset_metrics[dataset_name]['total_examples'] += num_examples
        dataset_metrics[dataset_name]['train_loss_server'] = m['train_loss_server']
        dataset_metrics[dataset_name]['train_loss_client'] = m['train_loss_client']
        dataset_metrics[dataset_name]['train_accuracy_server'] = m['train_accuracy_server']
        dataset_metrics[dataset_name]['train_accuracy_client'] = m['train_accuracy_client']
        dataset_metrics[dataset_name]['val_loss_server'] = m['val_loss_server']
        dataset_metrics[dataset_name]['val_loss_client'] = m['val_loss_client']
        dataset_metrics[dataset_name]['val_accuracy_server'] = m['val_accuracy_server']
        dataset_metrics[dataset_name]['val_accuracy_client'] = m['val_accuracy_client']

    return dataset_metrics

def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    dataset_metrics = {}

    for num_examples, m in metrics:
        dataset_name = m['dataset_name']
        if dataset_name not in dataset_metrics:
            dataset_metrics[dataset_name] = {'total_examples': 0}

        dataset_metrics[dataset_name]['total_examples'] += num_examples
        dataset_metrics[dataset_name]['test_loss_server'] = m['test_loss_server']
        dataset_metrics[dataset_name]['test_loss_client'] = m['test_loss_client']
        dataset_metrics[dataset_name]['test_accuracy_server'] = m['test_accuracy_server']
        dataset_metrics[dataset_name]['test_accuracy_client'] = m['test_accuracy_client']

    return dataset_metrics

'''
# global evaluation using testloader in server
def get_evaluate_fn(num_classes: int, testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = Net(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        loss, accuracy = test(model, testloader, device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy}

    return evaluate_fn
'''


# Define custom strategy
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, base_model, save_path, max_rounds, *args, **kwargs):
        super(SaveModelStrategy, self).__init__(*args, **kwargs)

        self.model = base_model
        self.save_path = save_path
        self.max_rounds = max_rounds
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Save the model at the end of the last round
        if aggregated_parameters is not None and server_round == self.max_rounds:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(self.model.state_dict(), f"{self.save_path}/model.pth")

        return aggregated_parameters, aggregated_metrics