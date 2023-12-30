# Probabilistic Neural Circuits

This is the code repository for the paper [Probabilistic Neural Circuits](https://pedrozudo.github.io/assets/documents/publications/2024/zuidberg2024probabilistic/zuidberg2024probabilistic.paper.pdf).

Please cite as follows:
```
@inproceedings{zuidberg2024probabilistic,
  title = {Probabilistic Neural Circuits},
  author = {Zuidberg Dos Martires, Pedro},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year = {2024},
}
```


## Installation

```sh
git clone git@github.com:pedrozudo/ProbabilisticNeuralCircuits.git
cd ProbabilisticNeuralCircuits
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

The experiment for density estimation on MNIST can be run using the following command:

```sh
python run.py experiments/GenRC/mnist.json 
```

The other experiments can be run by replacing the `mnist.json` file with either `fmnist.json` or `emnist.json`. Note that the first time you perform an experiment with a dataset this will download the dataset itself.

To run the experiments for the discriminative setting use the following command:
```sh
python run.py experiments/GenDisRC/mnist.json 
```