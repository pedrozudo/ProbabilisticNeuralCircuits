from pathlib import Path
import os
import math
import yaml
import hashlib

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_f1_score

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

from circuits import GenPNCRC, GenDisPNCRC


def cross_entropy(lps, y):
    y = y.unsqueeze(-1)
    loss = torch.gather(lps, -1, y).squeeze(-1)
    loss = loss - torch.logsumexp(lps, dim=-1)
    return -torch.mean(loss)


def accuracy(lps, y):
    predicted = torch.argmax(lps, dim=-1)
    return torch.mean((predicted == y).float())


def f1score(lps, y, n_classes):
    predictions = torch.argmax(lps, dim=-1)

    f1 = multiclass_f1_score(
        predictions,
        y,
        n_classes,
        top_k=1,
        average="macro",
        multidim_average="global",
        ignore_index=None,
        validate_args=True,
    )
    return f1


# RC = Row Column
class RC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.init_circuit()

        self.metrics = {
            "train": {},
            "val": {},
            "test": {},
            "parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint(
            dirpath=Path("checkpoints") / Path(self.make_name()),
            filename="checkpoint",
            **self.callbacks_kargs,
        )
        return [model_checkpoint]

    def make_name(self):
        assert len(self.hparams)
        h = [f"{k}:{self.hparams[k]}" for k in sorted(self.hparams.keys())]
        h = "-".join(h)
        h = hashlib.sha256(h.encode()).hexdigest()
        return f"{h}"

    def make_path(self):
        for pg in self.optimizers().optimizer.param_groups:
            weight_decay = pg["weight_decay"]

        optimizer_name = self.optimizers().optimizer.__class__.__name__

        path = (
            Path("results")
            / Path(f"{self.learning_type}")
            / Path(f"{self.hparams.loss}")
            / Path(f"{self.hparams.dataset}")
            / Path(f"{self.hparams.mixing}")
            / Path(f"components:{self.hparams.components}")
            / Path(f"{optimizer_name}:{weight_decay}")
            / Path(self.make_name())
        )
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def forward(self, x):
        return self.circuit(x)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
        )

    def setup_mnist(self, stage):
        if self.hparams.dataset == "mnist":
            from datasets.mnist import MNIST as MnistDataset
        elif self.hparams.dataset == "fmnist":
            from datasets.mnist import FashionMNIST as MnistDataset
        else:
            raise NotImplementedError

        if stage == "fit" or stage is None:
            d = MnistDataset("train")
            train_len = int(0.9 * len(d))
            val_len = len(d) - train_len
            self.train_data, self.val_data = torch.utils.data.random_split(
                d, [train_len, val_len]
            )
        else:
            self.test_data = MnistDataset("test")

    def setup_emnist(self, stage):
        from datasets.mnist import EMNIST

        split = self.hparams.dataset.split(":")[-1]
        if stage == "fit" or stage is None:
            d = EMNIST(split, "train")
            train_len = int(0.9 * len(d))
            val_len = len(d) - train_len

            self.train_data, self.val_data = torch.utils.data.random_split(
                d, [train_len, val_len]
            )
        else:
            self.test_data = EMNIST(split, "test")

    def setup(self, stage):
        if self.hparams.dataset in ["mnist", "fmnist"]:
            self.setup_mnist(stage)
        elif "emnist" in self.hparams.dataset:
            self.setup_emnist(stage)
        else:
            raise NotImplementedError

    def on_test_end(self):
        path = self.make_path()
        self.data2yaml(path, "hparams.yaml", self.hparams)
        self.data2yaml(path, "results.yaml", self.metrics)

    @staticmethod
    def data2yaml(path, filename, data):
        # https://stackoverflow.com/questions/36745577/how-do-you-create-in-python-a-file-with-permissions-other-users-can-write
        # The default umask is 0o22 which turns off write permission of group and others
        os.umask(0)

        flags = (
            os.O_WRONLY  # access mode: write only
            | os.O_CREAT  # create if not exists
            | os.O_TRUNC  # truncate the file to zero
        )

        def opener(path, flags):
            return os.open(path, flags, 0o777)

        with open(path / filename, "w", opener=opener) as handle:
            yaml.dump(data, handle, default_flow_style=False)

    @staticmethod
    def bit_per_dimension(lp, D):
        return -lp / math.log(2) / D


class GenRC(RC):
    def __init__(
        self,
        dataset: str,
        height: int,
        width: int,
        components: int,
        feature_dim: int,
        mixing: str,
        loss="nll",
        batch_size=32,
        num_workers=8,
        compile=False,
    ):
        self.learning_type = "Gen"
        self.callbacks_kargs = {"monitor": "val/bpd", "mode": "min"}

        super().__init__()

        self.metrics["train"]["bpd"] = []
        self.metrics["val"]["bpd"] = []
        self.metrics["test"]["bpd"] = None

    def init_circuit(self):
        params = self.hparams
        self.circuit = GenPNCRC(
            params["height"],
            params["width"],
            params["components"],
            params["feature_dim"],
            params["mixing"],
        )

    def training_step(self, batch, batch_idx):
        lp = self(batch["x"]).mean()
        bpd = self.bit_per_dimension(lp, self.hparams.height * self.hparams.width)

        self.log("train/bpd", bpd, on_epoch=True, prog_bar=True, logger=True)

        return -lp

    def on_train_epoch_end(self) -> None:
        bpd = self.trainer.logged_metrics["train/bpd_epoch"]
        self.metrics["train"]["bpd"].append(float(bpd))
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        lp = self(batch["x"]).mean()
        bpd = self.bit_per_dimension(lp, self.hparams.height * self.hparams.width)

        self.log("val/bpd", bpd, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        bpd = self.trainer.logged_metrics["val/bpd"]
        self.metrics["val"]["bpd"].append(float(bpd))

        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        lp = self(batch["x"]).mean()
        bpd = self.bit_per_dimension(lp, self.hparams.height * self.hparams.width)

        self.log("test/logp", lp, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/bpd", bpd, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        bpd = self.trainer.logged_metrics["test/bpd"]
        self.metrics["test"]["bpd"] = float(bpd)

        return super().on_test_epoch_end()


class GenDisRC(RC):
    def __init__(
        self,
        dataset: str,
        height: int,
        width: int,
        components: int,
        n_classes: int,
        mixing: str,
        loss="ce",
        batch_size=32,
        num_workers=8,
        compile=False,
    ):
        self.learning_type = "GenDis"
        self.callbacks_kargs = {"monitor": "val/acc", "mode": "max"}

        super().__init__()

        self.metrics["train"]["loss"] = []
        self.metrics["train"]["accuracy"] = []

        self.metrics["val"]["loss"] = []
        self.metrics["val"]["accuracy"] = []
        self.metrics["val"]["f1"] = []

        self.metrics["test"]["loss"] = None
        self.metrics["test"]["accuracy"] = None

    def init_circuit(self):
        params = self.hparams
        self.circuit = GenDisPNCRC(
            params["height"],
            params["width"],
            params["components"],
            params["n_classes"],
            params["mixing"],
        )

    def loss(self, x, y):
        return cross_entropy(x, y)

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        lps = self(x)

        loss = self.loss(lps, y)
        acc = accuracy(lps, y)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.trainer.logged_metrics["train/loss_epoch"]
        self.metrics["train"]["loss"].append(float(loss))

        acc = self.trainer.logged_metrics["train/acc_epoch"]
        self.metrics["train"]["accuracy"].append(float(acc))

        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        lps = self(x)

        loss = self.loss(lps, y)
        acc = accuracy(lps, y)
        f1 = f1score(lps, y, self.hparams.n_classes)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/f1", f1, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        loss = self.trainer.logged_metrics["val/loss"]
        self.metrics["val"]["loss"].append(float(loss))

        acc = self.trainer.logged_metrics["val/acc"]
        self.metrics["val"]["accuracy"].append(float(acc))

        f1 = self.trainer.logged_metrics["val/f1"]
        self.metrics["val"]["f1"].append(float(f1))

        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        lps = self(x)

        loss = self.loss(lps, y)
        acc = accuracy(lps, y)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        loss = self.trainer.logged_metrics["test/loss"]
        self.metrics["test"]["loss"] = float(loss)

        acc = self.trainer.logged_metrics["test/acc"]
        self.metrics["test"]["accuracy"] = float(acc)

        return super().on_test_epoch_end()


if __name__ == "__main__":
    cli = LightningCLI(run=True, save_config_callback=None)
    cli.trainer.test(cli.model)
