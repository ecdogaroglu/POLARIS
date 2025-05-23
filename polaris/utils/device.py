"""
Device selection and management utilities for POLARIS.
"""

import os
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DeviceType = Literal["cuda", "mps", "cpu"]


def get_best_device() -> DeviceType:
    """Get the best available device (CUDA, MPS, or CPU).

    For MPS (Apple Silicon), we perform a quick benchmark to ensure it's actually
    faster than CPU, as there are known issues with MPS in some PyTorch versions.

    The device can be forced by setting the TORCH_DEVICE environment variable.
    """
    # Check if device is forced via environment variable
    forced_device = os.environ.get("TORCH_DEVICE")
    if forced_device:
        if forced_device.lower() in ["cuda", "mps", "cpu"]:
            print(
                f"Using device '{forced_device}' from TORCH_DEVICE environment variable"
            )
            return forced_device.lower()
        else:
            print(
                f"Warning: Invalid TORCH_DEVICE value '{forced_device}'. Must be 'cuda', 'mps', or 'cpu'."
            )

    # Auto-select the best device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if MPS is actually faster than CPU with a quick benchmark
        if is_mps_faster_than_cpu():
            return "mps"
        else:
            print(
                "MPS available but benchmark shows it's slower than CPU. Using CPU instead."
            )
            return "cpu"
    else:
        return "cpu"


def is_mps_faster_than_cpu(test_size=256, repeat=10):
    """Run a benchmark to check if MPS is faster than CPU for neural network operations.

    Args:
        test_size: Size of test matrices/tensors
        repeat: Number of times to repeat the test

    Returns:
        bool: True if MPS is faster, False otherwise
    """
    try:
        # Create a simple test network
        class TestNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(test_size, test_size)
                self.linear2 = nn.Linear(test_size, test_size)
                self.linear3 = nn.Linear(test_size, 1)

            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = torch.relu(self.linear2(x))
                return self.linear3(x)

        # Test data
        test_input = torch.randn(64, test_size)

        # Test CPU performance
        model_cpu = TestNetwork()
        model_cpu.eval()
        test_input_cpu = test_input.clone()

        start_time = time.time()
        for _ in range(repeat):
            with torch.no_grad():
                _ = model_cpu(test_input_cpu)
        cpu_time = time.time() - start_time

        # Test MPS performance
        model_mps = TestNetwork().to("mps")
        model_mps.eval()
        test_input_mps = test_input.to("mps")

        # Warm up MPS
        with torch.no_grad():
            _ = model_mps(test_input_mps)

        start_time = time.time()
        for _ in range(repeat):
            with torch.no_grad():
                _ = model_mps(test_input_mps)
        mps_time = time.time() - start_time

        # Return True if MPS is at least 10% faster
        return mps_time < cpu_time * 0.9

    except Exception as e:
        print(f"MPS benchmark failed with error: {e}")
        return False


def _benchmark_mps() -> bool:
    """Benchmark MPS vs CPU performance."""
    try:
        # Test network
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 256)
                self.fc2 = nn.Linear(256, 256)
                self.gru = nn.GRU(256, 128, batch_first=True)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = x.unsqueeze(1)
                x, _ = self.gru(x)
                return x

        # Test data
        batch_size = 32
        input_data = torch.randn(batch_size, 256)

        # CPU test
        cpu_model = TestNet().to("cpu")
        cpu_model.eval()

        # Warmup
        for _ in range(2):
            _ = cpu_model(input_data)

        # Benchmark
        cpu_start = time.time()
        for _ in range(10):
            _ = cpu_model(input_data)
        cpu_time = time.time() - cpu_start

        # MPS test
        mps_model = TestNet().to("mps")
        mps_model.eval()
        mps_input = input_data.to("mps")

        # Warmup
        for _ in range(2):
            _ = mps_model(mps_input)
            torch.mps.synchronize()

        # Benchmark
        mps_start = time.time()
        for _ in range(10):
            _ = mps_model(mps_input)
            torch.mps.synchronize()
        mps_time = time.time() - mps_start

        print(f"Benchmark - CPU: {cpu_time:.4f}s, MPS: {mps_time:.4f}s")

        # MPS should be at least 20% faster
        return mps_time < cpu_time * 0.8

    except Exception as e:
        print(f"Error during MPS benchmark: {e}")
        return False
