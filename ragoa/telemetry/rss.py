"""Peak resident memory. Unix uses resource; Windows uses the working set."""

from __future__ import annotations

import sys


def peak_rss_mb() -> float:
    if sys.platform == "win32":
        return _peak_rss_mb_windows()
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _peak_rss_mb_windows() -> float:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_info = psapi.GetProcessMemoryInfo
    get_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    get_info.restype = wintypes.BOOL
    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return 0.0
    return counters.PeakWorkingSetSize / 1e6
