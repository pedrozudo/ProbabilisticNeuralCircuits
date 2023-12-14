import torch
import torch.nn.functional as F
import einops as E


class Neural(torch.nn.Module):
    def __init__(self, components, partitions, kernel_type):
        self.components = components
        self.mixing = kernel_type
        self.partitions = partitions

        k = kernel_type.split("-")

        self.kernel_type = kernel_type[0]
        self.kernel_shape = k[1]
        self.kernel_fields = int(k[2])

        super().__init__()

        self.init_conv()
        self.init_weights()

    def init_conv(self):
        self.init_kernel_params()

        self._conv = torch.nn.Conv2d(
            in_channels=self.components,
            out_channels=self.components**2,
            kernel_size=self._kernel_size,
            padding=self._padding,
        )

        with torch.no_grad():
            self._conv.weight = torch.nn.Parameter(
                self._conv.weight * self._kernel_mask
            )

    def init_kernel_params(self):
        if self.kernel_shape == "square":
            self.init_kernel_params_square()
        else:
            raise NotImplementedError

    def init_kernel_params_square(self):
        k = self.kernel_fields
        self._kernel_size = 2 * k - 1
        self._padding = k - 1

        mask = torch.zeros((self._kernel_size, self._kernel_size), dtype=torch.float)
        for i in range(k):
            for j in range(k):
                if not (i == k - 1 and j == k - 1):
                    mask[i, j] = 1.0

        mask = E.rearrange(mask, "... -> 1 1 ...")
        self.register_buffer("kernel_mask", mask)
        self._kernel_mask = mask

    def init_weights(self):
        self._weights = torch.nn.Parameter(
            torch.rand(self.components, self.components, self.partitions)
        )

    def forward(self, x):
        last_dim = x.shape[-1]
        x = self._conv(x)
        x = E.rearrange(x, "b (c1 c2) r c -> b c1 c2 (r c)", c1=self.components)
        x = torch.einsum("...ijk, ijk -> ...ijk", x, self._weights)
        x = E.rearrange(x, "... (r c) -> ... r c", c=last_dim)
        return x
