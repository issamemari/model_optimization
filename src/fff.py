import math

import torch
from torch.nn import Module, Parameter, init


class FastFeedForward(Module):
    def __init__(
        self,
        in_features: int,
        leaf_features: int,
        out_features: int,
        depth: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.n_leaves = 2**depth
        self.n_nodes = 2**depth - 1

        self.in_features = in_features
        self.leaf_features = leaf_features
        self.out_features = out_features
        self.depth = depth

        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device

        self.register_leaf_parameters()
        self.register_node_parameters()

        self.init_parameters()

    def register_leaf_parameters(self) -> None:
        self.leaf_weights1 = Parameter(
            torch.empty(
                (self.n_leaves, self.in_features, self.leaf_features),
                **self.factory_kwargs,
            )
        )
        self.leaf_biases1 = Parameter(
            torch.empty(
                (self.n_leaves, self.leaf_features),
                **self.factory_kwargs,
            )
        )
        self.leaf_weights2 = Parameter(
            torch.empty(
                (self.n_leaves, self.leaf_features, self.out_features),
                **self.factory_kwargs,
            )
        )
        self.leaf_biases2 = Parameter(
            torch.empty(
                (self.n_leaves, self.out_features),
                **self.factory_kwargs,
            )
        )

    def register_node_parameters(self) -> None:
        self.node_weights = Parameter(
            torch.empty(
                (self.n_nodes, self.in_features),
                **self.factory_kwargs,
            )
        )
        self.node_biases = Parameter(
            torch.empty(
                (self.n_nodes,),
                **self.factory_kwargs,
            )
        )

    def init_parameters(self) -> None:
        for weights, biases in zip(
            [self.leaf_weights1, self.leaf_weights2, self.node_weights],
            [self.leaf_biases1, self.leaf_biases2, self.node_biases],
        ):
            init.kaiming_uniform_(weights, a=math.sqrt(5))

            fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(biases, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaf_nodes = self.get_leaf_nodes(x)
        output = self.forward_leaf_nodes(x, leaf_nodes)
        return output

    def get_leaf_nodes(self, x):
        batch_size = x.shape[0]

        current_nodes = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        for _ in range(self.depth):
            node_weights = self.node_weights[current_nodes]  # (batch_size, in_features)
            node_biases = self.node_biases[current_nodes]  # (batch_size,)

            node_predictions = (
                torch.bmm(x.unsqueeze(1), node_weights.unsqueeze(2))
                .squeeze(1)
                .squeeze(1)
                + node_biases
            )

            choices = (node_predictions > 0).long()

            current_nodes = 2 * current_nodes + choices + 1

        current_nodes -= self.n_nodes

        return current_nodes

    def forward_leaf_nodes(self, x, leaf_nodes):
        leaf_weights1 = self.leaf_weights1[leaf_nodes]  # (batch, in, leaf)
        leaf_biases1 = self.leaf_biases1[leaf_nodes]  # (batch, leaf)

        hidden = torch.matmul(x.unsqueeze(1), leaf_weights1).squeeze(1) + leaf_biases1

        leaf_weights2 = self.leaf_weights2[leaf_nodes]  # (batch, leaf, out)
        leaf_biases2 = self.leaf_biases2[leaf_nodes]  # (batch, out)

        output = (
            torch.matmul(hidden.unsqueeze(1), leaf_weights2).squeeze(1) + leaf_biases2
        )

        return output
