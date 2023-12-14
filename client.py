import timeit
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from typing import Any, Callable, Dict, List, Optional, Tuple
from copy import deepcopy
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from utils import train, test, get_weights, set_weights


class ResnetClient(fl.client.Client):
    """Flower client implementing ImageNet image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        nb_clients: int,
        DEVICE: str = 'cuda',
        
    ) -> None:
        self.cid = cid
        self.model = resnet18().to(DEVICE)
        self.trainloader = trainloader
        self.testloader = testloader
        self.nb_clients = nb_clients
        
        self.prev_model = None
        self.wa = None
        self.wa_updated = None
        self.soup_avg= None
        self.soup_avg_num = 0
        

    def get_parameters(self, config): #-> fl.common.ParametersRes:
        print(f"Client {self.cid}: get_parameters")
        weights: fl.common.NDArrays = get_weights(self.model)
        parameters = fl.common.ndarrays_to_parameters(weights)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: fl.common.FitIns): #-> fl.common.FitRes:
        # Set the seed so we are sure to generate the same global batches
        # indices across all clients
        np.random.seed(123)

        print(f"Client {self.cid}: fit")

        weights: fl.common.NDArrays = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        print(config)
        total_rounds = config["total_rounds"]
        alpha = config["alpha"]
        train_round = config["server_round"]
        # # Get training config
        # epochs = int(config["epochs"])
        # batch_size = int(config["batch_size"])

        # Set model parameters
        set_weights(self.model, weights)


        #copying prev iteration
        self.prev_model = deepcopy(self.model)
        self.prev_model.load_state_dict(self.model.state_dict())

        
        train(self.model, self.trainloader, epochs=1)
        if train_round > alpha * total_rounds:
            print("Weight Averaging......")
            if self.soup_avg_num == 0:
                self.soup_avg = deepcopy(self.model)
                self.soup_avg.load_state_dict(
                    self.prev_model.state_dict()
                )

            self.wa = deepcopy(self.model)
            self.wa_updated = deepcopy(self.model)

            self.wa.load_state_dict(self.model.state_dict())
            self.wa_updated.load_state_dict(self.model.state_dict())

            
            # lets explain step by step what is happening here
            
            for wa_param, u_wa_param, soup_param, prev_model in zip(  # we are iterating over the parameters of the models (wa, u_wa, soup, prev_model)
                self.wa.parameters(),
                self.wa_updated.parameters(),
                self.soup_avg.parameters(),
                self.prev_model.parameters(),
            ):
                # updating the parameters of the updated weight averaging model
                wa_param.data = wa_param.data.clone() * (
                    1.0 / (self.soup_avg_num + 1.0)
                ) + soup_param.data.clone() * (
                    self.soup_avg_num / (self.soup_avg_num + 1.0)
                )
                # updating the parameters of the updated weight averaging model
                u_wa_param.data = (
                    u_wa_param.data.clone() * (1.0 / (self.soup_avg_num + 2.0))
                    + soup_param.data.clone()
                    * (self.soup_avg_num / (self.soup_avg_num + 2.0))
                    + prev_model.data.clone()
                    * (1.0 / (self.soup_avg_num + 2.0))
                )
                # preparing for updated per_global_model
                prev_model.data = (1.0 / (self.soup_avg_num + 1.0)) * (
                    self.soup_avg_num * soup_param.data.clone()
                    + prev_model.data.clone()
                )

            # local_acc = self.quick_test(self.model)
            # we obtain the accuracy of the original weight averaging model and the updated weight averaging model
            _, wa_acc = test(self.wa, self.testloader)
            _, update_wa_acc = test(self.wa_updated, self.testloader)
            # print("Local Accuracy: ", local_acc)
            print("Original Weight Averaging Accuracy: ", wa_acc)
            print("Updated Weight Averaging Accuracy: ", update_wa_acc)
            # we update the global model with the updated weight averaging model if the updated weight averaging model has a better accuracy
            if update_wa_acc > wa_acc:
                print("Update Personalized Global Model......")
                self.model.load_state_dict(self.wa_updated.state_dict())
                self.soup_avg.load_state_dict(
                    self.prev_model.state_dict()
                )
                self.soup_avg_num += 1
                print("Personalized Global Model Num: ", self.soup_avg_num)
            else:
                print("Remain the same Personalized Global Model.")
                self.model.load_state_dict(self.wa.state_dict())
            del self.prev_model, self.wa, self.wa_updated

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.common.NDArrays = get_weights(self.model)
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)     
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=params_prime,
            num_examples=len(self.trainloader),
            metrics={},
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        # Set the set so we are sure to generate the same batches
        # across all clients.
        np.random.seed(123)

        print(f"Client {self.cid}: evaluate")

        # config = ins.config
        # batch_size = int(config["batch_size"])

        weights = fl.common.parameters_to_ndarrays(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)
        loss, accuracy = test(self.model, self.testloader)
        status = Status(code=Code.OK, message="Success")
        # Return the number of evaluation examples and the evaluation result (loss)
        #print("im")
        return fl.common.EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.testloader),
            metrics = {"accuracy": float(accuracy),"loss": float(loss)},
        )
        
        
class ResnetClientAvg(fl.client.Client):
    """Flower client implementing ImageNet image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        nb_clients: int,
        DEVICE: str = 'cuda',
        
    ) -> None:
        self.cid = cid
        self.model = resnet18().to(DEVICE)
        self.trainloader = trainloader
        self.testloader = testloader
        self.nb_clients = nb_clients
        
        

    def get_parameters(self, config): #-> fl.common.ParametersRes:
        print(f"Client {self.cid}: get_parameters")
        weights: fl.common.NDArrays = get_weights(self.model)
        parameters = fl.common.ndarrays_to_parameters(weights)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: fl.common.FitIns): #-> fl.common.FitRes:
        # Set the seed so we are sure to generate the same global batches
        # indices across all clients
        np.random.seed(123)

        print(f"Client {self.cid}: fit")

        weights: fl.common.NDArrays = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        print(config)
        total_rounds = config["total_rounds"]
        alpha = config["alpha"]
        train_round = config["server_round"]
        # # Get training config
        # epochs = int(config["epochs"])
        # batch_size = int(config["batch_size"])

        # Set model parameters
        set_weights(self.model, weights)


        
        train(self.model, self.trainloader, epochs=1)
        

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.common.NDArrays = get_weights(self.model)
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)     
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=params_prime,
            num_examples=len(self.trainloader),
            metrics={},
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        # Set the set so we are sure to generate the same batches
        # across all clients.
        np.random.seed(123)

        print(f"Client {self.cid}: evaluate")

        # config = ins.config
        # batch_size = int(config["batch_size"])

        weights = fl.common.parameters_to_ndarrays(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)
        loss, accuracy = test(self.model, self.testloader)
        status = Status(code=Code.OK, message="Success")
        # Return the number of evaluation examples and the evaluation result (loss)
        #print("im")
        return fl.common.EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.testloader),
            metrics = {"accuracy": float(accuracy),"loss": float(loss)},
        )