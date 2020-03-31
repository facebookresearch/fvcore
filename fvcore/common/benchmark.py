# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np


def timeit(
    num_iters: int = -1, warmup_iters: int = 0
) -> Callable[[], Callable[[], Dict[str, float]]]:
    """
    This is intened to be used as a decorator to time any function.

    Args:
        num_iters (int): number of iterations used to compute the average time
            (sec) required to run the function. If negative, the number of
            iterations is determined dynamically by running the function a few
            times to make sure the estimate is stable.
        warmup_iters (int): number of iterations used to warm up the function.
            This is useful for functions that exhibit poor performance during
            the first few times they run (due to caches, autotuning, etc).
    Returns:
        Dict[str, float]: dictionary of the aggregated timing estimates.
            "iterations": number of iterations used to compute the estimated
                          time.
            "mean": averate time (sec) used to run the function.
            "median": median time (sec) used to run the function.
            "min": minimal time (sec) used to run the function.
            "max": maximal time (sec) used to run the function.
            "stddev": standard deviation of the time (sec) used to run the
                      function.
    """

    # pyre-ignore
    def decorator(func: Callable[[], Any]) -> Callable[[], Dict[str, float]]:
        def decorated(*args: Any, **kwargs: Any) -> Dict[str, float]:
            # Warmup phase.
            for _ in range(warmup_iters):
                func(*args, **kwargs)

            # Estimate the run time of the function.
            total_time: float = 0
            count = 0
            run_times: List[float] = []
            max_num_iters = num_iters if num_iters > 0 else sys.maxsize
            for _ in range(max_num_iters):
                start_time = time.time()
                func(*args, **kwargs)
                run_time = time.time() - start_time

                run_times.append(run_time)
                total_time += run_time
                count += 1
                if num_iters < 0 and total_time >= 0.5:
                    # If num_iters is negative, run the function enough times so
                    # that we can have a more robust estimate of the average time.
                    break
            assert count == len(run_times)
            ret: Dict[str, float] = {}
            ret["iterations"] = count
            ret["mean"] = total_time / count
            ret["median"] = np.median(run_times)
            ret["min"] = np.min(run_times)
            ret["max"] = np.max(run_times)
            ret["stddev"] = np.std(run_times)
            return ret

        return decorated

    return decorator  # pyre-ignore


def benchmark(
    func: Callable[[], Any],  # pyre-ignore
    bm_name: str,
    kwargs_list: List[Any],  # pyre-ignore
    *,
    num_iters: int = -1,
    warmup_iters: int = 0
) -> None:
    """
    Benchmark the input function and print out the results.

    Args:
        func (callable): a closure that returns a function for benchmarking,
            where initialization can be done before the function to benchmark.
        bm_name (str): name of the benchmark to print out, e.g. "BM_UPDATE".
        kwargs_list (list): a list of argument dict to pass to the function. The
            intput function will be timed separately for each argument dict.
        num_iters (int): number of iterations to run. Defaults to run until 0.5s.
        warmup_iters (int): number of iterations used to warm up the function.

    Outputs:
        For each argument dict, print out the time (in microseconds) required
        to run the function along with the number of iterations used to get
        the timing estimate. Example output:

        Benchmark               Avg Time(μs)   Peak Time(μs)     Iterations
        -------------------------------------------------------------------
        BM_UPDATE_100                    820             914            610
        BM_UPDATE_1000                  7655            8709             66
        BM_UPDATE_10000                78062           81748              7
        -------------------------------------------------------------------
    """

    print("")
    outputs = []
    for kwargs in kwargs_list:
        func_bm = func(**kwargs)
        # pyre-ignore
        time_func = timeit(num_iters=num_iters, warmup_iters=warmup_iters)(func_bm)

        ret = time_func()
        name = bm_name
        if kwargs:
            name += "_" + "_".join(str(v) for k, v in kwargs.items())
        outputs.append(
            [
                name,
                str(ret["mean"] * 1000000),
                str(ret["max"] * 1000000),
                str(ret["iterations"]),
            ]
        )
    outputs = np.array(outputs)
    # Calculate column widths for metrics table.
    c1 = len(max(outputs[:, 0], key=len))
    c2 = len(max(outputs[:, 1], key=len))
    c3 = len(max(outputs[:, 2], key=len))
    c4 = len(max(outputs[:, 3], key=len))
    dash = "-" * 80
    print(
        "{:{}s} {:>{}s} {:>{}s} {:>{}s}".format(
            "Benchmark", c1, "Avg Time(μs)", c2, "Peak Time(μs)", c3, "Iterations", c4
        )
    )
    print(dash)
    for output in outputs:
        print(
            "{:{}s} {:15.0f} {:15.0f} {:14d}".format(
                output[0], c1, float(output[1]), float(output[2]), int(output[3])
            )
        )
    print(dash)
