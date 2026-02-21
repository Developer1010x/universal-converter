"""Platform utilities for finding executables"""

import os
import shutil
import subprocess
from typing import Optional, List


def find_executable(name: str) -> Optional[str]:
    """Find an executable in system PATH"""
    return shutil.which(name)


def run_command(cmd: List[str], timeout: int = 300, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command with timeout"""
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=capture_output,
        check=False
    )


def get_platform_info() -> dict:
    """Get platform information"""
    import platform
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }
