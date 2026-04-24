"""Build a seccomp BPF filter via libseccomp (ctypes).

No pip dependency — uses libseccomp.so from the host OS.
Returns a readable fd containing the compiled BPF; caller passes it to
bwrap via --add-seccomp-fd and must close it after Popen().

Blocked syscalls (SCMP_ACT_ERRNO(EPERM)):
  - namespace: unshare, setns, mount, umount2, pivot_root
  - tracing/debug: ptrace, kcmp, perf_event_open
  - kernel keyring: keyctl, add_key, request_key
  - eBPF: bpf
  - exploit primitives: userfaultfd
  - privilege: acct, syslog
  - file-handle escape: open_by_handle_at, name_to_handle_at
  - notification: fanotify_init

Intentionally NOT blocked: clone/clone3/fork (threads/subprocesses),
socket (network handled at bwrap --unshare-net level).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os

logger = logging.getLogger(__name__)

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO_EPERM = 0x00050000 | 1  # EPERM

_BLOCKED_SYSCALLS = [
    "unshare",
    "setns",
    "mount",
    "umount2",
    "pivot_root",
    "ptrace",
    "kcmp",
    "perf_event_open",
    "keyctl",
    "add_key",
    "request_key",
    "bpf",
    "userfaultfd",
    "acct",
    "syslog",
    "open_by_handle_at",
    "name_to_handle_at",
    "fanotify_init",
]


def _load_lib() -> ctypes.CDLL | None:
    name = ctypes.util.find_library("seccomp")
    if not name:
        return None
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return None

    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_release.restype = None
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_export_bpf.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    return lib


_lib: ctypes.CDLL | None | bool = False  # False = not yet loaded


def _get_lib() -> ctypes.CDLL | None:
    global _lib
    if _lib is False:
        _lib = _load_lib()
        if _lib is None:
            logger.warning("libseccomp not found — seccomp filter disabled")
    return _lib  # type: ignore[return-value]


def build_filter_fd() -> int | None:
    """Return readable fd with compiled BPF, or None if unavailable."""
    lib = _get_lib()
    if lib is None:
        return None

    ctx = lib.seccomp_init(_SCMP_ACT_ALLOW)
    if not ctx:
        logger.warning("seccomp_init failed")
        return None

    ctx_ptr = ctypes.c_void_p(ctx)
    skipped = []
    for name in _BLOCKED_SYSCALLS:
        nr = lib.seccomp_syscall_resolve_name(name.encode())
        if nr < 0:
            skipped.append(name)
            continue
        ret = lib.seccomp_rule_add(ctx_ptr, _SCMP_ACT_ERRNO_EPERM, nr, 0)
        if ret != 0:
            logger.warning("seccomp_rule_add %s failed: %d", name, ret)

    if skipped:
        logger.debug("seccomp: unknown syscalls on this arch: %s", skipped)

    r_fd, w_fd = os.pipe()
    ret = lib.seccomp_export_bpf(ctx_ptr, w_fd)
    os.close(w_fd)
    lib.seccomp_release(ctx_ptr)

    if ret != 0:
        logger.warning("seccomp_export_bpf failed: %d", ret)
        os.close(r_fd)
        return None

    return r_fd
