"""Platform utilities for locating and running external tools."""

import shutil
import subprocess
from typing import List, Optional


def find_executable(name: str) -> Optional[str]:
    """Return the absolute path of ``name`` on PATH, or ``None``."""
    return shutil.which(name)


def run_command(
    cmd: List[str], timeout: int = 300, capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run ``cmd`` and return the :class:`subprocess.CompletedProcess`.

    ``text=True`` is always passed so ``.stdout``/``.stderr`` are ``str`` and
    can be parsed directly (for example with :func:`json.loads`) rather than
    coming back as ``bytes``.
    """
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=capture_output,
        text=True,
        check=False,
    )


def get_platform_info() -> dict:
    """Return a dict describing the host platform."""
    import platform

    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }
