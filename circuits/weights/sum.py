import torch
import torch.nn.functional as F
import einops as E


class Sum(torch.nn.Module):
    def __init__(self, components, partitions, mixing):
        super().__init__()
        self.components = components
        self.partitions = partitions
        self.mixing = mixing

        self.init_weights()

    def init_weights(self):
        self.weights = torch.nn.Parameter(
            torch.rand(self.components, self.components, self.partitions)
        )

    def forward(self, logp):
        last_dim = logp.shape[-1]
        batches = logp.shape[0]
        weights = E.rearrange(self.weights, "... (r c)-> ... r c", c=last_dim)
        return E.repeat(weights, "... -> b ...", b=batches)
