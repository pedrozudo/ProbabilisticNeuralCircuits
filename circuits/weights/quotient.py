import torch
import torch.nn.functional as F
import einops as E


class Quotient(torch.nn.Module):
    def __init__(self, components, partitions, kernel_type):
        super().__init__()
        self.components = components
        self.partitions = partitions
        self.mixing = kernel_type

        k = kernel_type.split("-")

        self.kernel_type = kernel_type[0]
        self.kernel_shape = k[1]
        self.kernel_fields = int(k[2])

        self.init_kernel()
        self.init_weights()

    def init_weights(self):
        self._weights = torch.nn.Parameter(
            torch.rand(self.components, self.components, self.partitions)
        )

    def init_kernel(self):
        k = self.kernel_fields
        self._kernel_size = 2 * k - 1
        self._padding = k - 1

        kernel = torch.zeros((self._kernel_size, self._kernel_size), dtype=torch.float)
        for i in range(k):
            for j in range(k):
                if not (i == k - 1 and j == k - 1):
                    kernel[i, j] = 1.0

        kernel = E.rearrange(kernel, "r c  -> 1 1 r c")
        self.register_buffer(f"kernel", kernel)

    def forward(self, logp):
        last_dim = logp.shape[-1]

        logp = E.rearrange(logp, "b N ... -> (b N) 1 ...")
        x = F.conv2d(
            logp,
            self.kernel,
            bias=torch.tensor([0.0], device=logp.device),
            padding=self.kernel.size(-1) // 2,
        )
        x = E.rearrange(x, "(b N) 1 ... -> b 1 N ...", N=self.components)
        weights = E.rearrange(self._weights, "PN CN (r c) -> 1 PN CN r c", c=last_dim)

        return weights + x
