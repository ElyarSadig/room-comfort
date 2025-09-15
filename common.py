import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import tracemalloc
import psutil
import os
import platform
import cpuinfo

def print_machine_info():
    # Get CPU info
    cpu = cpuinfo.get_cpu_info()
    cpu_name = cpu.get('brand_raw', 'Unknown CPU')
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_ghz = f"{cpu_freq.max / 1000:.2f} GHz" if cpu_freq else "Unknown"

    # Get RAM info
    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024 ** 3)

    # Get OS info
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    # Print in benchmark-style format
    print("=== Machine Info ===")
    print(f"OS:         {os_info}")
    print(f"CPU:        {cpu_name} ({cpu_cores} cores / {cpu_threads} threads @ {cpu_freq_ghz})")
    print(f"RAM:        {total_ram_gb:.2f} GB")
    print(f"Python:     {platform.python_version()}")
    print("=============================================")


def benchmark_pipeline(pipeline, X_train, y_train, X_test, y_test):
    process = psutil.Process(os.getpid())
    
    # Track memory
    tracemalloc.start()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    cpu_start = time.time()
    
    # Train
    pipeline.fit(X_train, y_train)
    train_time = time.time() - cpu_start
    mem_after = process.memory_info().rss / 1024 / 1024
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Training time: {train_time:.3f} s")
    print(f"Memory before training: {mem_before:.2f} MB")
    print(f"Memory after training: {mem_after:.2f} MB")
    print(f"Peak memory during training (tracemalloc): {peak / 1024 / 1024:.2f} MB")

    # Prediction
    cpu_start = time.time()
    y_pred = pipeline.predict(X_test)
    pred_time = time.time() - cpu_start

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.where(np.isclose(y_test, 0.0), 1e-8, y_test))) * 100

    print(f"Prediction time: {pred_time:.3f} s")
    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%\n")

    return {
        "r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "y_pred": y_pred,
        "train_time": train_time, "pred_time": pred_time,
        "memory_before_MB": mem_before, "memory_after_MB": mem_after, "peak_memory_MB": peak / 1024 / 1024
    }