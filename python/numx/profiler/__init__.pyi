import numx.core


def enable_memory_profile() -> None:
    """Enable memory profiling"""

def enable_device_memory_profile(device_name: str) -> None:
    """Enable device memory profiling"""

def disable_memory_profile() -> None:
    """Disable memory profiling"""

def disable_device_memory_profile(device_name: str) -> None:
    """Disable device memory profiling"""

def print_memory_profile() -> None:
    """Print memory profile to the console"""

def print_device_memory_profile(device_name: str) -> bool:
    """Print device memory profile to the console"""

def print_graph_profile(array: numx.core.Array) -> None:
    """Print graph profile to the console"""

def save_memory_profile(file_name: str) -> None:
    """Save memory profile to a file"""

def save_device_memory_profile(device_name: str, file_name: str) -> bool:
    """Save device memory profile to a file"""

def save_graph_profile(array: numx.core.Array, file_name: str) -> None:
    """Save graph profile to a file"""
