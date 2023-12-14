import subprocess
import argparse
import json
import itertools

import datasets.mnist as mnist


def run(experiment):
    with open(args.experiment) as handle:
        file_contents = handle.read()
    config = json.loads(file_contents)
    # experiment = experiment.split(".")[0].split("/")[-1]

    datasets = config.pop("datasets")
    keys, values = zip(*config.items())
    config = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for dataset in datasets:
        if dataset in mnist.MNIST_DATASETS:
            height = 28
            width = 28
        if dataset in mnist.N_CLASSES.keys():
            n_classes = mnist.N_CLASSES[dataset]

        if (structure := config[0]["structure"]) == "GenRC":
            class_path = "__main__.GenRC"

            for c in config:
                subprocess_arguments = [
                    "python3",
                    f"train.py",
                    "fit",
                    "--config",
                    f"experiments/{structure}/{c['train_config']}",
                    "--seed_everything",
                    f"{c['seed_everything']}",
                    "--model.class_path",
                    f"{class_path}",
                    "--model.dataset",
                    f"{dataset}",
                    "--model.height",
                    f"{height}",
                    "--model.width",
                    f"{width}",
                    "--model.components",
                    f"{c['components']}",
                    "--model.mixing",
                    f"{c['mixing']}",
                    "--model.feature_dim",
                    f"{c['feature_dim']}",
                    "--model.batch_size",
                    f"{c['batch_size']}",
                    "--model.num_workers",
                    f"{c['num_workers']}",
                    "--trainer.max_epochs",
                    f"{c['max_epochs']}",
                    # "--trainer.limit_train_batches",
                    # "3",
                    # "--trainer.limit_val_batches",
                    # "3",
                    # "--trainer.detect_anomaly",
                    # "true",
                    # "--print_config",
                ]

                subprocess.run(subprocess_arguments)

        if (structure := config[0]["structure"]) == "GenDisRC":
            class_path = "__main__.GenDisRC"

            for c in config:
                subprocess_arguments = [
                    "python3",
                    f"train.py",
                    "fit",
                    "--config",
                    f"experiments/{structure}/{c['train_config']}",
                    "--seed_everything",
                    f"{c['seed_everything']}",
                    "--model.class_path",
                    f"{class_path}",
                    "--model.dataset",
                    f"{dataset}",
                    "--model.height",
                    f"{height}",
                    "--model.width",
                    f"{width}",
                    "--model.components",
                    f"{c['components']}",
                    "--model.n_classes",
                    f"{n_classes}",
                    "--model.mixing",
                    f"{c['mixing']}",
                    "--model.loss",
                    f"{c['loss']}",
                    "--model.batch_size",
                    f"{c['batch_size']}",
                    "--model.num_workers",
                    f"{c['num_workers']}",
                    "--trainer.max_epochs",
                    f"{c['max_epochs']}",
                    # "--trainer.limit_train_batches",
                    # "3",
                    # "--trainer.limit_val_batches",
                    # "3",
                    # "--trainer.detect_anomaly",
                    # "true",
                    # "--print_config",
                    "--optimizer.weight_decay",
                    f"{c['weight_decay']}",
                ]

                subprocess.run(subprocess_arguments)

        if (structure := config[0]["structure"]) == "DisRC":
            class_path = "__main__.DisRC"

            for c in config:
                subprocess_arguments = [
                    "python3",
                    f"train.py",
                    "fit",
                    "--config",
                    f"experiments/{structure}/{c['train_config']}",
                    "--seed_everything",
                    f"{c['seed_everything']}",
                    "--model.class_path",
                    f"{class_path}",
                    "--model.dataset",
                    f"{dataset}",
                    "--model.height",
                    f"{height}",
                    "--model.width",
                    f"{width}",
                    "--model.components",
                    f"{c['components']}",
                    "--model.feature_dim",
                    f"{c['feature_dim']}",
                    "--model.n_classes",
                    f"{n_classes}",
                    "--model.mixing",
                    f"{c['mixing']}",
                    "--model.loss",
                    f"{c['loss']}",
                    "--model.batch_size",
                    f"{c['batch_size']}",
                    "--model.num_workers",
                    f"{c['num_workers']}",
                    "--trainer.max_epochs",
                    f"{c['max_epochs']}",
                    # "--trainer.limit_train_batches",
                    # "3",
                    # "--trainer.limit_val_batches",
                    # "3",
                    # "--trainer.detect_anomaly",
                    # "true",
                    # "--print_config",
                ]

                subprocess.run(subprocess_arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)

    args = parser.parse_args()

    run(args.experiment)
