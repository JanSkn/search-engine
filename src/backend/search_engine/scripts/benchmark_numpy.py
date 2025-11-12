import time
import numpy as np


def benchmark_operations(size: int) -> None:
    # python lists as in query_engine each numpy array must be converted first
    list1 = list(np.random.randint(0, size * 10, size))
    list2 = list(np.random.randint(0, size * 10, size))

    print(f"\n=== Benchmark for size = {size:,} ===")

    start = time.perf_counter()
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    np.union1d(arr1, arr2)
    np_union_time = time.perf_counter() - start

    start = time.perf_counter()
    sorted(set(list1) | set(list2))
    py_union_time = time.perf_counter() - start

    start = time.perf_counter()
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    np.setdiff1d(arr1, arr2)
    np_diff_time = time.perf_counter() - start

    start = time.perf_counter()
    sorted(set(list1) - set(list2))
    py_diff_time = time.perf_counter() - start

    start = time.perf_counter()
    arr1 = np.array(list1)
    np.sort(arr1)
    np_sort_time = time.perf_counter() - start

    start = time.perf_counter()
    sorted(list1)
    py_sort_time = time.perf_counter() - start

    print(
        f"Union:     NumPy={np_union_time:.6f}s | Python={py_union_time:.6f}s\n"
        f"SetDiff:   NumPy={np_diff_time:.6f}s | Python={py_diff_time:.6f}s\n"
        f"Sorting:   NumPy={np_sort_time:.6f}s | Python={py_sort_time:.6f}s"
    )


if __name__ == "__main__":
    for size in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        benchmark_operations(size)
