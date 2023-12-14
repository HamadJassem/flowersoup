# Federated Learning Simulation

This project is a simulation of federated learning using Flower. In this project, a FedSoup adaptation for flower is trialed and compared with FedAvg. The model leveraged in this simulation is ResNet18.

## Requirements

- Python 3.7 or higher
- PyTorch
- Flower
- Numpy

## Usage

To run the simulation, use the following command:

```bash
python main.py --num_clients 5 --total_rounds 40 --alpha 0.75