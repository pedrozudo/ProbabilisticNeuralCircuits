# Probabilistic Neural Circuits

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

The experiment for density estimation on MNIST can be run using the following command:

```console
cd ProbabilisticNeuralCircuits
python run.py experiments/GenRC/mnist.json 
```

The other experiments can be run by replacing the `mnist.json` file with either `fmnist.json` or `emnist.json`. Note that the first time you perform an experiment with a dataset this will download the dataset itself.

To run the experiments for the discriminative setting use the following command:
```console
python run.py experiments/GenDisRC/mnist.json 
```