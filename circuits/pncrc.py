import torch
import torch.nn.functional as F
import einops as E

from .weights import Neural, Sum, Quotient


# probabilsitic neurel circuit row columns
class PNCRC(torch.nn.Module):
    def __init__(
        self,
        height,
        width,
        components,
        feature_dim,
        mixing,
    ):
        super().__init__()

        self.height, self.width = height, width
        self.components = components
        self.mixing = mixing

        assert isinstance(self.mixing, str)

        self.feature_dim = feature_dim

        self.init_layers()
        self.init_parameters()

        self.computed_weights = {}

    def forward(self, x):
        x = self.prepare_input(x)
        return self.compute_layers(x)

    def get_columns(self, i):
        if i == -1:
            return self.width
        return self.layer_columns[i]

    def get_rows(self, i):
        if i == -1:
            return self.height
        return self.layer_rows[i]

    def init_layers(self):
        layer_rows = []
        layer_columns = []
        layer_partitions = []

        layer_rows.append(self.height)
        layer_columns.append(self.width)
        layer_partitions.append(layer_columns[-1] * layer_rows[-1])

        i = 0

        while layer_partitions[-1] != 1:
            if i % 2:
                lc = layer_columns[-1]
                lr = max(1, layer_rows[-1] // 2)
            else:
                lc = max(1, layer_columns[-1] // 2)
                lr = layer_rows[-1]

            layer_columns.append(lc)
            layer_rows.append(lr)

            layer_partitions.append(layer_rows[-1] * layer_columns[-1])
            i += 1

        self.layer_rows = layer_rows
        self.layer_columns = layer_columns
        self.layer_partitions = layer_partitions
        self.n_layers = len(layer_partitions)

    def init_parameters(self):
        self.init_leaf()
        self.init_mixing()
        self.init_root()

    def init_leaf(self, *args, **kwargs):
        raise NotImplementedError

    def init_root(self, *args, **kwargs):
        raise NotImplementedError

    def init_mixing(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_input(self, *args, **kwargs):
        raise NotImplementedError

    def compute_leaf(self, *args, **kwargs):
        raise NotImplementedError

    def compute_root(self, *args, **kwargs):
        raise NotImplementedError

    def compute_layers(self, logp):
        logp = self.compute_leaf(logp)
        for i in range(self.n_layers):
            logp = self.compute_mixing(logp, i)
            logp = self.compute_product(logp, i)

        return self.compute_root(logp)

    def compute_mixing(self, logp, i):
        layer_name = f"MixingLayer{i}"
        weights = self.weights[layer_name](logp)

        logp = E.repeat(logp, "b ... -> b parents ...", parents=self.components)
        weights = F.log_softmax(weights, dim=2)
        return torch.logsumexp(logp + weights, dim=2)

    def compute_product(self, logp, i):
        if i == self.n_layers - 1:
            return logp[..., 0, 0]

        if i % 2:
            if logp.shape[-2] % 2:
                logp = torch.cat(
                    [logp[..., :-2, :], logp[..., -2:-1, :] + logp[..., -1:, :]], dim=-2
                )  # add last two rows to make even
            left = logp[..., ::2, :]
            right = logp[..., 1::2, :]
        else:
            if logp.shape[-1] % 2:
                logp = torch.cat(
                    [logp[..., :-2], logp[..., -2:-1] + logp[..., -1:]], dim=-1
                )  # add last two columns to make even
            left = logp[..., ::2]
            right = logp[..., 1::2]

        logp = left + right
        return logp

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################


class GenPNCRC(PNCRC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_leaf(self):
        self.leaf_weights = torch.nn.Parameter(
            torch.rand(self.components, self.feature_dim, self.height, self.width)
        )

    def init_mixing(self):
        self.weights = torch.nn.ModuleDict()
        for i in range(self.n_layers):
            if self.mixing == "sum":
                Mixing = Sum
            elif self.mixing.startswith("neural"):
                Mixing = Neural
            elif self.mixing.startswith("quotient"):
                Mixing = Quotient
            else:
                raise NotImplementedError

            self.weights[f"MixingLayer{i}"] = Mixing(
                self.components, self.layer_partitions[i], self.mixing
            )

    def init_root(self):
        self.root_weights = torch.nn.Parameter(torch.rand(self.components))

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def prepare_input(self, x):
        if self.feature_dim == 2:
            x = x / 255
            x[x < 0.5] = 0.0
            x[x >= 0.5] = 1.0
            x.int()
        else:
            assert self.feature_dim == 256

        x = F.one_hot(x.long())
        x = E.rearrange(x, "batch ... children -> batch children ...")

        return torch.log(x + 10e-20)

    def compute_leaf(self, logp):
        logp = E.repeat(logp, "b ... -> b parents ...", parents=self.components)
        weights = F.log_softmax(self.leaf_weights, dim=1)
        weights = E.rearrange(weights, "... -> 1 ...")
        return torch.logsumexp(logp + weights, dim=2)

    def compute_root(self, logp):
        weights = F.log_softmax(self.root_weights, dim=-1)
        weights = E.rearrange(weights, "... -> 1 ...")
        return torch.logsumexp(logp + weights, dim=-1)


################################################################################################################################
################################################################################################################################
################################################################################################################################


class GenDisPNCRC(PNCRC):
    def __init__(
        self,
        height,
        width,
        components,
        n_classes,
        mixing,
    ):
        self.n_classes = n_classes
        super().__init__(
            height,
            width,
            components,
            feature_dim=2,
            mixing=mixing,
        )

    def init_leaf(self):
        self.leaf_weights = torch.nn.Parameter(
            torch.rand(
                self.n_classes,
                self.components,
                self.feature_dim,
                self.height,
                self.width,
            )
        )

    def init_mixing(self):
        self.weights = torch.nn.ModuleDict()

        for i in range(self.n_layers):
            if self.mixing.startswith("neural"):
                Mixing = Neural
            elif self.mixing.startswith("quotient"):
                Mixing = Quotient
            elif self.mixing.startswith("sum"):
                Mixing = Sum
            else:
                raise NotImplementedError

            self.weights[f"MixingLayer{i}"] = Mixing(
                self.components, self.layer_partitions[i], self.mixing
            )

    def init_root(self):
        self.root_weights = torch.nn.Parameter(
            torch.rand(self.n_classes, self.components)
        )

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def prepare_input(self, x):
        x = x / 255.0
        x = E.rearrange([x, 1 - x], "f b ... -> b f ...")
        x = E.repeat(x, "b ... -> (b NC) ...", NC=self.n_classes)
        return torch.log(x + 10e-20)

    def compute_leaf(self, logp):
        logp = E.repeat(logp, "b ... -> b parents ...", parents=self.components)
        logp = E.rearrange(logp, "(b NC)... -> b NC ...", NC=self.n_classes)

        weights = F.log_softmax(self.leaf_weights, dim=-3)
        weights = E.rearrange(weights, "... -> 1 ...")

        logp = torch.logsumexp(logp + weights, dim=-3)
        logp = E.rearrange(logp, "b NC ... -> (b NC) ...")

        return logp

    def compute_root(self, logp):
        logp = E.rearrange(logp, "(b NC)... -> b NC ...", NC=self.n_classes)

        weights = F.log_softmax(self.root_weights, dim=-1)
        weights = E.rearrange(weights, "... -> 1 ...")

        return torch.logsumexp(logp + weights, dim=-1)
