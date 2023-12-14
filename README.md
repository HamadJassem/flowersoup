# Federated Learning Simulation

This project is a simulation of federated learning using Flower. In this project, a FedSoup adaptation for flower is trialed and compared with FedAvg. The model leveraged in this simulation is ResNet18. The adaptations are contained in the client.py file.

## Requirements

- Python 3.7 or higher
- PyTorch
- Flower
- Numpy

## Usage

To run the simulation, use the following command:

```bash
python main.py --num_clients 5 --total_rounds 40 --alpha 0.75
```

## Client Changes
### The following lines describe the code we provide to adapt FedAvg to FedSoup
fit method:

1) Lines 65-90: obtaining parameters from server
2) Line 92: Local training
3) Lines 93-155 FedSoup Extension

get_parameters method:
1) returns client parameters

evaluate method:
1) Sets parameters with global parameters
2) evaluates model on local test set
3) returns results
