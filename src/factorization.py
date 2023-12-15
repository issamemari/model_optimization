import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class MatrixFactorization(nn.Module):
    def __init__(self, rows, cols, n_factors):
        super().__init__()
        self.row_factors = nn.Parameter(torch.randn(rows, n_factors))
        self.col_factors = nn.Parameter(torch.randn(n_factors, cols))

    def forward(self):
        return self.row_factors @ self.col_factors


class FactorizedLinear(nn.Module):
    def __init__(
        self,
        row_factors: torch.Tensor,
        col_factors: torch.Tensor,
        bias: torch.Tensor = None,
    ):
        super().__init__()
        self.row_factors = row_factors
        self.col_factors = col_factors
        self.bias = bias

    def forward(self, x):
        x = F.linear(x, self.col_factors, bias=None)
        x = F.linear(x, self.row_factors, bias=self.bias)
        return x


def factorize(
    layer: nn.Linear,
    n_factors,
    n_epochs=10000,
    lr=0.01,
) -> Tuple[FactorizedLinear, torch.Tensor]:
    """
    Factorize a linear layer into two matrices of shape (rows, n_factors) and
    (n_factors, cols) respectively and create a FactorizedLinear layer from them.

    Args:
        layer (nn.Linear): The layer to be factorized.
        n_factors (int): The number of factors to use for factorization.
        n_epochs (int): The number of epochs to train the factorization model.
        lr (float): The learning rate to use for training the factorization model.

    Returns:
        Tuple[FactorizedLinear, torch.Tensor]: The factorized layer and the loss after training.
    """
    weights = layer.weight.data
    rows, cols = weights.size()

    model = MatrixFactorization(rows, cols, n_factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model()
        loss = criterion(output, weights)
        loss.backward()
        optimizer.step()

    return (
        FactorizedLinear(model.row_factors, model.col_factors, getattr(layer, "bias")),
        loss,
    )


def optimize(
    model: nn.Module,
    n_factors: int,
    n_epochs: int = 1000,
    lr: float = 0.01,
    names: Optional[List] = None,
) -> nn.Module:
    for name, module in model.named_children():
        if not isinstance(module, nn.Linear):
            optimize(module, n_factors, n_epochs, lr, names)
            continue

        if names is not None and name not in names:
            continue

        print(f"Factorizing {name}...")
        factorized_layer, loss = factorize(module, n_factors, n_epochs, lr)
        setattr(model, name, factorized_layer)
        print(f"Reconstruction MSE: {loss}")

    return model


def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def main():
    model_name = "prajjwal1/bert-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)

    model_optimized = AutoModel.from_pretrained(model_name)
    model_optimized = optimize(model_optimized, 300, 5000, 0.01, names=["dense"])

    input = tokenizer("Hello, my dog is cute", return_tensors="pt")

    iters = 1000
    total_time = 0
    total_time_optimized = 0
    for iter in range(iters):
        t1 = time.perf_counter()
        output = model(**input)
        t2 = time.perf_counter()
        total_time += t2 - t1

        t1 = time.perf_counter()
        output_optimized = model_optimized(**input)
        t2 = time.perf_counter()
        total_time_optimized += t2 - t1

        print(f"Average time: {total_time / (iter + 1)}")
        print(f"Average time optimized: {total_time_optimized / (iter + 1)}")

    print(f"Average time: {total_time / iters}")
    print(f"Average time optimized: {total_time_optimized / iters}")
    print(f"Speedup: {total_time / total_time_optimized}")

    print(
        f"Cosine similarity: {cosine_similarity(output.last_hidden_state[0][0], output_optimized.last_hidden_state[0][0])}"
    )

    import code

    code.interact(local=locals())


if __name__ == "__main__":
    main()
