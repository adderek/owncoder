"""Background job registry + HTML UI ask-answer bridge."""
import asyncio

import pytest

from agent.core import background


async def test_register_list_cancel_task():
    async def sleeper():
        await asyncio.sleep(30)

    t = asyncio.ensure_future(sleeper())
    background.register_task(t, "test-sleeper", "test")
    jobs = [j for j in background.jobs() if j["label"] == "test-sleeper"]
    assert len(jobs) == 1
    assert jobs[0]["killable"]
    assert background.cancel(jobs[0]["id"])
    await asyncio.sleep(0.05)
    assert not [j for j in background.jobs() if j["label"] == "test-sleeper"]


async def test_external_job_visible_not_killable():
    jid = background.register_external("sched:test (every 6h)", "scheduler")
    try:
        jobs = {j["id"]: j for j in background.jobs()}
        assert jid in jobs
        assert not jobs[jid]["killable"]
        assert not background.cancel(jid)
    finally:
        background.unregister(jid)
    assert jid not in {j["id"] for j in background.jobs()}


def test_cancel_unknown_job():
    assert not background.cancel(999999)


async def test_answer_latest_unblocks_broker_ask():
    """UI answer resolves a turn blocked in NotifyBroker.ask (remote_answers +
    on_timeout='wait') instead of deadlocking the session."""
    from agent.notify.broker import NotifyBroker
    from agent.notify.messages import Question

    class FakeNotifyCfg:
        enabled = True
        channels = []
        events = ["ask_user"]
        remote_answers = True
        answer_timeout_s = 0
        on_timeout = "wait"
        relay_responses = True

    class FakeCfg:
        notify = FakeNotifyCfg()

    class FakeChan:
        name = "fake"
        capability = "chat"

        async def send(self, msg):
            pass

    broker = NotifyBroker(FakeCfg())
    broker._channels = [FakeChan()]
    q = Question(kind="ask_user", text="test?", free_text=True)
    ask_task = asyncio.ensure_future(broker.ask(q))
    await asyncio.sleep(0.05)
    assert broker.answer_latest("ui answer")
    ans = await asyncio.wait_for(ask_task, 2)
    assert ans is not None and ans.text == "ui answer" and ans.source == "ui"
    assert not broker.answer_latest("nothing pending")
