import time

import torch
import torch.nn as nn

from fff import FastFeedForward


def count_model_parameters(model: nn.Module) -> int:
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()

    return n_params


def check_single_step_faster(m: int, n: int, k: int, d: int) -> bool:
    single_step_ops = m * (2 * n - 1)
    multi_step_ops = k * (2 * n - 1) + m * (2 * k - 1) + (d - 1) * (2 * n - 1)

    print(f"Single step ops: {single_step_ops}")
    print(f"Multi step ops: {multi_step_ops}")
    return single_step_ops < multi_step_ops


def main():
    in_features = 2000
    leaf_features = 50
    out_features = 2000
    depth = 4
    batch_size = 500
    max_iter = 1000

    model1 = FastFeedForward(in_features, leaf_features, out_features, depth)
    model2 = nn.Linear(in_features, out_features)

    model1 = torch.jit.script(model1)
    model2 = torch.jit.script(model2)

    print(f"FastFeedForward has {count_model_parameters(model1)} parameters")
    print(f"Linear has {count_model_parameters(model2)} parameters")

    single_step_faster = check_single_step_faster(
        out_features, in_features, leaf_features, depth
    )
    print(f"Single step should be faster: {single_step_faster}")

    model1_avg_time = 0
    model2_avg_time = 0

    iter = 0
    while iter < max_iter:
        iter += 1

        x = torch.rand([batch_size, in_features])

        t1 = time.perf_counter()
        with torch.jit.optimized_execution(True):
            out = model1(x)
        t2 = time.perf_counter()

        model1_avg_time = (model1_avg_time * (iter - 1) + (t2 - t1)) / iter

        t1 = time.perf_counter()
        with torch.jit.optimized_execution(True):
            out = model2(x)
        t2 = time.perf_counter()

        model2_avg_time = (model2_avg_time * (iter - 1) + (t2 - t1)) / iter

        print(f"FastFeedForward avg batch inference duration: {model1_avg_time}")
        print(f"Linear avg batch inference duration: {model2_avg_time}")
        print(f"Speedup: {model2_avg_time / model1_avg_time}")


if __name__ == "__main__":
    main()
