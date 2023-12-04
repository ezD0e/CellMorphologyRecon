import time
import numpy as np

def memory_efficiency_test():
    array_size = 100_000_000_000  # Size of the array. Adjust this based on your system's memory.
    iterations = 5  # How many times the operation will be repeated.

    # Create a large array of random numbers
    print(f"Creating an array with {array_size} elements.")
    start_time = time.time()
    large_array = np.random.rand(array_size)
    creation_time = time.time() - start_time
    print(f"Time taken to create the array: {creation_time:.4f} seconds.")

    # Sum the array, this is a memory-bound operation
    sum_times = []
    for i in range(iterations):
        start_time = time.time()
        total_sum = np.sum(large_array)
        end_time = time.time() - start_time
        sum_times.append(end_time)
        print(f"Iteration {i+1}: Time taken to sum the array: {end_time:.4f} seconds. Sum: {total_sum}")

    average_time = sum(sum_times) / len(sum_times)
    print(f"Average time taken to sum the array: {average_time:.4f} seconds.")

    return creation_time, average_time

# Run the memory efficiency test
creation_time, average_sum_time = memory_efficiency_test()



# CPU Benchmark: Calculate Fibonacci sequence
def cpu_benchmark():
    def fib(n):
        if n <= 1:
            return n
        else:
            return fib(n-1) + fib(n-2)

    start_time = time.time()
    fib(35)  # adjust the value for a longer or shorter benchmark
    end_time = time.time()

    return end_time - start_time

# Memory Benchmark: Allocate and deallocate large arrays
def memory_benchmark():
    start_time = time.time()
    for _ in range(10):  # adjust the range for a longer or shorter benchmark
        arr = np.random.rand(1000000)  # adjust the size for more or less memory
        arr *= 10
        del arr
    end_time = time.time()

    return end_time - start_time

cpu_time = cpu_benchmark()
memory_time = memory_benchmark()

print(f"CPU benchmark time: {cpu_time} seconds")
print(f"Memory benchmark time: {memory_time} seconds")
