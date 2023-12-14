from flwr.common import Metrics
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timeit
from copy import deepcopy
from torchvision.models import resnet18
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar
)
import argparse

parser = argparse.ArgumentParser(description='FedSoup Flower')
parser.add_argument('--num_clients', type=int, default=5, help='number of clients')
parser.add_argument('--num_rounds', type=int, default=1, help='number of rounds')
parser.add_argument('--alpha', type=float, default=0.75, help='alpha')

args = parser.parse_args()
NUM_CLIENTS = args.num_clients
total_rounds = args.num_rounds
alpha = args.alpha

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils import get_weights, set_weights, read_data, load_datasets, create_global_testset
from utils import train, test
from client import ResnetClient, ResnetClientAvg

trainloaders, valloaders, testloaders = load_datasets(NUM_CLIENTS, val=True)

def get_on_fit_config_fn(total_rounds, alpha):
    def fit_config(server_round: int):
        config = {
            "alpha": alpha,
            "total_rounds":total_rounds,
            "server_round": server_round,
        }
        return config
    return fit_config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def numpyclient_fn(cid) -> ResnetClient:
    trainloader = trainloaders[int(cid)]
    testloader = valloaders[int(cid)]
    return ResnetClient(cid, trainloader, testloader, 1)


def numpyclient_fn_avg(cid) -> ResnetClientAvg:
    trainloader = trainloaders[int(cid)]
    testloader = testloaders[int(cid)]
    return ResnetClientAvg(cid, trainloader, testloader, 1)


testloaderglobal = create_global_testset()
tsnes = []
def get_evaluate_fn(model):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[Metrics]:
        global testloaderglobal
        global tsnes
        global global_last_model
        global total_rounds
        set_weights(model, parameters)
        loss, accuracy = test(model, testloaderglobal)
            
        if server_round == total_rounds - 1:
            torch.save(model.state_dict(), "global_model.pt")
            global_last_model = model
            
        
        return loss, {"global_accuracy": accuracy}
    return evaluate



client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}
global_last_model = None
model = resnet18().to(DEVICE)
model_parameters = get_weights(model)
model_parameters = fl.common.ndarrays_to_parameters(model_parameters)
hist = fl.simulation.start_simulation(
    client_fn=numpyclient_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=total_rounds),
    client_resources=client_resources,
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config_fn(total_rounds, alpha),
        evaluate_fn=get_evaluate_fn(model),
        initial_parameters=model_parameters,
        ),
    
)

print(hist)