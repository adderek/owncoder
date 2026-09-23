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
  - new mount API: open_tree, move_mount, fs{open,config,mount,pick}, mount_setattr
  - kernel modules / kexec
  - io_uring_* (ENOSYS, so runtimes fall back)

Partially blocked: clone is allowed only when no namespace flag is set
(threads/subprocesses); namespace-creating clone and all clone3 are denied.
fork/vfork and socket stay unblocked (network handled at bwrap
--unshare-net level).

BACKEND ASYMMETRY (P3):
  This filter is only applied on the bwrap path (--add-seccomp-fd).
  The firejail backend uses firejail's own built-in default seccomp filter
  (--seccomp flag) which covers a broader but different syscall set and is
  not controlled by _BLOCKED_SYSCALLS here. The curated list above documents
  our threat model; firejail's filter provides overlapping but not identical
  coverage. Prefer bwrap for stronger, reproducible seccomp guarantees.
  To apply _BLOCKED_SYSCALLS on firejail too, pass --seccomp=<comma-list>
  once firejail compatibility can be verified on a firejail-equipped system.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os

logger = logging.getLogger(__name__)

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO_EPERM = 0x00050000 | 1  # EPERM
_SCMP_ACT_ERRNO_ENOSYS = 0x00050000 | 38  # ENOSYS — lets glibc fall back from clone3
_SCMP_CMP_MASKED_EQ = 7


class _ScmpArgCmp(ctypes.Structure):
    """Mirror of struct scmp_arg_cmp from libseccomp."""
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


# Namespace-creating clone flags. Plain clone (threads/subprocesses) must stay
# allowed, so we deny only these bits — a nested userns+mountns is a kernel
# exploit primitive and `unshare` alone is not enough to close it.
_CLONE_NS_FLAGS = [
    0x00000080,  # CLONE_NEWTIME
    0x00020000,  # CLONE_NEWNS
    0x02000000,  # CLONE_NEWCGROUP
    0x04000000,  # CLONE_NEWUTS
    0x08000000,  # CLONE_NEWIPC
    0x10000000,  # CLONE_NEWUSER
    0x20000000,  # CLONE_NEWPID
    0x40000000,  # CLONE_NEWNET
]

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
    # New mount API: same power as mount(2), separate syscalls.
    "open_tree",
    "move_mount",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "mount_setattr",
    # Kernel code loading / replacement.
    "init_module",
    "finit_module",
    "delete_module",
    "kexec_load",
    "kexec_file_load",
]

# io_uring: a recurring kernel-exploit surface, and its ops bypass the
# per-syscall filter above. ENOSYS, not EPERM, so runtimes that probe it
# (libuv, tokio) fall back to plain syscalls.
_ENOSYS_SYSCALLS = [
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
]


def _load_lib() -> ctypes.CDLL | None:
    name = ctypes.util.find_library("seccomp")
    if not name:
        return None
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return None
    if not hasattr(lib, "seccomp_rule_add_array"):
        return None  # libseccomp too old for arg-filtered rules

    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_release.restype = None
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCmp),
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
            logger.warning(
                "libseccomp not found — sandbox will run WITHOUT syscall "
                "filtering (install libseccomp for full isolation)"
            )
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
        ret = lib.seccomp_rule_add_array(ctx_ptr, _SCMP_ACT_ERRNO_EPERM, nr, 0, None)
        if ret != 0:
            logger.warning("seccomp_rule_add %s failed: %d", name, ret)

    for name in _ENOSYS_SYSCALLS:
        nr = lib.seccomp_syscall_resolve_name(name.encode())
        if nr < 0:
            skipped.append(name)
            continue
        ret = lib.seccomp_rule_add_array(ctx_ptr, _SCMP_ACT_ERRNO_ENOSYS, nr, 0, None)
        if ret != 0:
            logger.warning("seccomp_rule_add %s failed: %d", name, ret)

    # clone: deny only namespace-creating invocations (one masked-equality rule
    # per flag, since libseccomp cannot express "any of these bits set").
    clone_nr = lib.seccomp_syscall_resolve_name(b"clone")
    if clone_nr >= 0:
        for flag in _CLONE_NS_FLAGS:
            cmp = _ScmpArgCmp(0, _SCMP_CMP_MASKED_EQ, flag, flag)
            ret = lib.seccomp_rule_add_array(
                ctx_ptr, _SCMP_ACT_ERRNO_EPERM, clone_nr, 1, ctypes.byref(cmp)
            )
            if ret != 0:
                logger.warning("seccomp_rule_add clone flag %#x failed: %d", flag, ret)

    # clone3 carries its flags in a struct we cannot inspect; deny it with
    # ENOSYS so glibc falls back to clone (which is flag-filtered above).
    clone3_nr = lib.seccomp_syscall_resolve_name(b"clone3")
    if clone3_nr >= 0:
        ret = lib.seccomp_rule_add_array(ctx_ptr, _SCMP_ACT_ERRNO_ENOSYS, clone3_nr, 0, None)
        if ret != 0:
            logger.warning("seccomp_rule_add clone3 failed: %d", ret)

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
