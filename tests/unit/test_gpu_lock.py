"""Cross-process GPU semaphore: bounds concurrency, releases on close/crash."""
from __future__ import annotations

import asyncio
import os

import pytest

from agent.core.gpu_lock import CrossProcessSemaphore, resolve_lock_dir


def test_resolve_lock_dir_modes(tmp_path):
    assert resolve_lock_dir("off", "k") is None
    assert resolve_lock_dir("none", "k") is None
    explicit = resolve_lock_dir(str(tmp_path / "x"), "k")
    assert explicit == (tmp_path / "x")
    # auto: same key -> same dir; different key -> different dir
    a1 = resolve_lock_dir("", "http://localhost:8081/v1")
    a2 = resolve_lock_dir("", "http://localhost:8081/v1")
    b = resolve_lock_dir("", "http://localhost:9000/v1")
    assert a1 == a2 and a1 != b
    assert "owncoder-gpu" in str(a1)


def test_one_slot_excludes_second_acquire(tmp_path):
    sem = CrossProcessSemaphore(slots=1, lock_dir=tmp_path)
    fd = sem._try_acquire()
    assert fd is not None
    # all slots taken -> next attempt fails (separate fd, contends even in-proc)
    assert sem._try_acquire() is None
    os.close(fd)  # release
    fd2 = sem._try_acquire()
    assert fd2 is not None
    os.close(fd2)


def test_n_slots_capacity(tmp_path):
    sem = CrossProcessSemaphore(slots=2, lock_dir=tmp_path)
    a, b = sem._try_acquire(), sem._try_acquire()
    assert a is not None and b is not None
    assert sem._try_acquire() is None  # cap reached
    os.close(a)
    assert (c := sem._try_acquire()) is not None
    for fd in (b, c):
        os.close(fd)


async def test_acquire_blocks_until_release(tmp_path):
    sem = CrossProcessSemaphore(slots=1, lock_dir=tmp_path, poll_s=0.01)
    order: list[str] = []

    async with sem.acquire():
        order.append("first-in")

        async def second():
            async with sem.acquire():
                order.append("second-in")

        task = asyncio.create_task(second())
        await asyncio.sleep(0.05)
        # second must still be waiting — slot held
        assert "second-in" not in order
        order.append("first-out")
    await asyncio.wait_for(task, 2)
    assert order == ["first-in", "first-out", "second-in"]


def test_init_gpu_semaphore_wires_cross_process(tmp_path):
    from agent.core import model_status as ms
    try:
        ms.init_gpu_semaphore(2, tmp_path)
        assert ms.get_gpu_semaphore() is not None
        assert ms._gpu_xsem is not None
        # disabled path
        ms.init_gpu_semaphore(2, None)
        assert ms._gpu_xsem is None
    finally:
        ms._gpu_sem = None
        ms._gpu_xsem = None


def test_crash_releases_slot(tmp_path):
    """A held slot fd closed (as on process death) frees the slot."""
    sem = CrossProcessSemaphore(slots=1, lock_dir=tmp_path)
    fd = sem._try_acquire()
    assert fd is not None and sem._try_acquire() is None
    os.close(fd)  # simulates fd cleanup on process exit
    # a fresh semaphore over the same dir (new "process") can acquire
    sem2 = CrossProcessSemaphore(slots=1, lock_dir=tmp_path)
    fd2 = sem2._try_acquire()
    assert fd2 is not None
    os.close(fd2)
