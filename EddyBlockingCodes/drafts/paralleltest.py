import time
from concurrent.futures import ProcessPoolExecutor

# Simulate a time-consuming task (e.g., heavy computation)
def slow_square(x):
    time.sleep(0.1)  # Simulate a delay
    return x * x

# Serial computation
def serial_compute(data):
    results = []
    for x in data:
        results.append(slow_square(x))
    return results

# Parallel computation using multiprocessing
def parallel_compute(data):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(slow_square, data))
    return results

# Test data
data = list(range(20))  # 20 tasks

# Measure serial execution time
start = time.time()
serial_results = serial_compute(data)
serial_time = time.time() - start
print(f"Serial time: {serial_time:.2f} seconds")

# Measure parallel execution time
start = time.time()
parallel_results = parallel_compute(data)
parallel_time = time.time() - start
print(f"Parallel time: {parallel_time:.2f} seconds")