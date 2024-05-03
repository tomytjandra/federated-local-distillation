from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

from torch.utils.data import DataLoader, Subset

import torch
import flwr as fl
import numpy as np

from torcheeg.trainers import ClassifierTrainer

from distillation import *

def load_checkpoint(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    hparams = checkpoint['hyper_parameters']
    trainer = ClassifierTrainer(**hparams)

    return trainer

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 testloader: DataLoader,
                 server_model: torch.nn.Module,
                 client_models_folder: str,
                 ) -> None:
        super().__init__()
        
        # dataloaders
        self.raw_trainloader, self.grid_trainloader = trainloader
        self.raw_valloader, self.grid_valloader = valloader
        self.raw_testloader, self.grid_testloader = testloader

        # dataset info
        dataset = self.raw_trainloader.dataset.dataset.__dict__['dataset']
        self.dataset_name = dataset.__class__.__name__.lower().replace('dataset', '').replace('binary', '')
        
        # load client-side model
        checkpoint_path = f'{client_models_folder}/{self.dataset_name}.ckpt'
        self.client_model = load_checkpoint(checkpoint_path)

        # a model that is randomly initialised at first
        self.server_model = server_model
        self.num_classes = server_model.num_classes

        # figure out if this client has access to GPU support or not
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = server_model.__class__.__name__.lower()
        self.deterministic = False if model_name == 'cbamfeaturemapping' else True

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.server_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.server_model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        current_round = config.get("current_round", 0)
        mode = config["mode"]
        lr = config["lr"]
        #weight_decay = config["weight_decay"]
        epochs = config["local_epochs"]
        temperature = config["temperature"]
        alpha = config["alpha"]

        # initialize the model
        self.set_parameters(parameters)

        # do local distillation
        """
        for epoch in range(epochs):
            train_loss, train_accuracy = train_offline_kd(
                epoch,
                teacher_model=self.client_model,
                student_model=self.server_model,
                data_raw=self.raw_trainloader,
                data_grid=self.grid_trainloader,
                lr=lr,
                temperature=temperature,
                alpha=alpha,
            )
            
            val_loss, val_accuracy = validate_offline_kd(
                epoch,
                teacher_model=self.client_model,
                student_model=self.server_model,
                data_raw=self.raw_valloader,
                data_grid=self.grid_valloader,
                temperature=temperature,
                alpha=alpha,
            )
        
        
        train_loss, train_accuracy = np.random.random(), np.random.random()
        val_loss, val_accuracy = np.random.random(), np.random.random()
    
        return self.get_parameters({}), len(self.raw_trainloader.dataset), {'dataset_name': self.dataset_name, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy}
        """
        
        for epoch in range(epochs):
            train_loss_server, train_loss_client, train_accuracy_server, train_accuracy_client = train_kd(
                rnd=current_round,
                epoch=epoch,
                teacher_model=self.client_model,
                student_model=self.server_model,
                data_raw=self.raw_trainloader,
                data_grid=self.grid_trainloader,
                lr=lr,
                temperature=temperature,
                alpha=alpha,
                mode=mode
            )
            
            val_loss_server, val_loss_client, val_accuracy_server, val_accuracy_client = validate_kd(
                rnd=current_round,
                epoch=epoch,
                teacher_model=self.client_model,
                student_model=self.server_model,
                data_raw=self.raw_trainloader,
                data_grid=self.grid_trainloader,
                temperature=temperature,
                alpha=alpha,
                mode=mode
            )
          
        """  
        train_loss_server, train_loss_client, train_accuracy_server, train_accuracy_client = np.random.random(), np.random.random(), np.random.random(), np.random.random()
        val_loss_server, val_loss_client, val_accuracy_server, val_accuracy_client = np.random.random(), np.random.random(), np.random.random(), np.random.random()
        """

        return self.get_parameters({}), len(self.raw_trainloader.dataset), {
            'dataset_name': self.dataset_name,
            'train_loss_server': train_loss_server,
            'train_loss_client': train_loss_client,
            'train_accuracy_server': train_accuracy_server,
            'train_accuracy_client': train_accuracy_client,
            'val_loss_server': val_loss_server,
            'val_loss_client': val_loss_client,
            'val_accuracy_server': val_accuracy_server,
            'val_accuracy_client': val_accuracy_client
        }


    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # initialize the model
        self.set_parameters(parameters)
    
        
        test_loss_server, test_accuracy_server = test_model(self.server_model, self.grid_testloader)
        test_loss_client, test_accuracy_client = test_model(self.client_model, self.raw_testloader)
        """
        test_loss_server, test_accuracy_server = np.random.random(), np.random.random()
        test_loss_client, test_accuracy_client = np.random.random(), np.random.random()
        """
        
        return float(test_loss_server), len(self.raw_testloader.dataset), {
            'dataset_name': self.dataset_name,
            'test_loss_server': test_loss_server,
            'test_loss_client': test_loss_client,
            'test_accuracy_server': test_accuracy_server,
            'test_accuracy_client': test_accuracy_client
        }


def generate_client_fn(trainloaders, valloaders, testloaders, server_model, client_models_folder):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloaders[int(cid)],
            server_model=server_model,
            client_models_folder=client_models_folder,
        ).to_client()

    # return the function to spawn client
    return client_fn
