import time
import os
from concurrent.futures import ProcessPoolExecutor

def my_function(x):
    """
    This is the function you want to apply to each element in x_list.
    It simulates a CPU-bound computation.

    Args:
        x: The input parameter for this specific computation.

    Returns:
        A value representing the computational result.
    """
    # Simulate a CPU-intensive task
    result = 0
    for i in range(1_000_000): # A million iterations to make it noticeable
        result += (x * i) % 987654321
        result %= 1000000007 # Keep result within a manageable range

    # Add some identification for demonstration purposes
    return f"Input: {x}, Result: {result}, Process ID: {os.getpid()}"

if __name__ == "__main__":
    # --- 1. Define your input list ---
    x_list = [10, 20, 5, 15, 25, 8, 30, 12, 18, 22]

    # --- 2. Parallelized execution with ProcessPoolExecutor ---
    print("Starting parallel computations...")
    start_time_parallel = time.time()

    # Create a ProcessPoolExecutor.
    # By default, max_workers=None uses the number of CPU cores available.
    with ProcessPoolExecutor(max_workers=None) as executor:
        # executor.map() is perfect for this scenario:
        # - It applies my_function to each item in x_list.
        # - It distributes these tasks across the worker processes.
        # - It collects the results into an iterator, maintaining the input order.
        # - We convert it to a list to get all results immediately.
        y_list = list(executor.map(my_function, x_list))

    end_time_parallel = time.time()
    print("Parallel computations finished.")

    # --- 3. Print Results ---
    print("\n--- Parallel Results (y_list) ---")
    for i, result in enumerate(y_list):
        print(f"Original Input x_list[{i}] = {x_list[i]}, Result: {result}")

    print(f"\nTotal parallel time: {end_time_parallel - start_time_parallel:.4f} seconds")

    # --- Optional: Compare with Sequential Execution ---
    print("\nStarting sequential computations for comparison...")
    start_time_sequential = time.time()

    y_list_sequential = []
    for x_val in x_list:
        y_list_sequential.append(my_function(x_val))

    end_time_sequential = time.time()
    print("Sequential computations finished.")

    print(f"\nTotal sequential time: {end_time_sequential - start_time_sequential:.4f} seconds")

    # You'll likely see a significant speedup with the parallel version
    # if my_function is genuinely CPU-bound and your machine has multiple cores.
