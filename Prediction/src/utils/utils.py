import multiprocessing


def get_max_workers():
    # Get CPU count and set workers
    max_workers = multiprocessing.cpu_count()  # Gets all CPU cores
    # Or use a portion of available CPUs
    max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    max_workers = multiprocessing.cpu_count() // 2  # Use half of cores
    return max_workers
