"""The browser can see the plan and the goal.

Both existed server-side and were reachable only by typing /plan or /goal and
reading printed text back — so the agent's current multi-step state, the one
thing worth having on screen, was the one thing that could not be shown.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.planning import plan as plan_mod
from agent.ui.http_loop import _HttpUI, _PAGE

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class _Server:
    def __init__(self, agent=None, goal=""):
        self._agent = agent
        self._goal = goal

    def get_goal(self):
        return self._goal


def _ui(server):
    ui = _HttpUI.__new__(_HttpUI)
    ui.server = server
    ui.session = None
    return ui


@pytest.fixture
def plans(tmp_path):
    plan_mod.configure(str(tmp_path), ".agent")
    yield plan_mod


class TestEndpoint:
    def test_no_plan_is_not_an_error(self, plans):
        out = _ui(_Server(agent=SimpleNamespace(), goal="")).plan_info()
        assert out == {"goal": "", "plan": None}

    def test_the_goal_comes_through_without_a_plan(self, plans):
        out = _ui(_Server(agent=SimpleNamespace(), goal="ship it")).plan_info()
        assert out["goal"] == "ship it"

    def test_a_remote_backend_has_no_plan_to_read(self, plans):
        """_agent_of returns None for the relay bridge; that is not a crash."""
        out = _ui(_Server(agent=None)).plan_info()
        assert out["plan"] is None

    def test_an_active_plan_is_reported_with_its_steps(self, plans):
        p = plans.create_plan("rewrite the parser", session_id="S1",
                              steps=["read grammar", "write lexer", "wire up"])
        plans.update_step(p, "s1", status="completed")
        plans.update_step(p, "s2", status="in_progress")

        out = _ui(_Server(agent=SimpleNamespace(session=SimpleNamespace(id="S1")))).plan_info()
        got = out["plan"]
        assert got["id"] == p.id and got["goal"] == "rewrite the parser"
        assert (got["done"], got["total"]) == (1, 3)
        assert got["current"] == "s2"
        assert [s["status"] for s in got["steps"]] == \
            ["completed", "in_progress", "pending"]

    def test_a_ready_step_is_flagged(self, plans):
        p = plans.create_plan("g", steps=["a", "b"])
        out = _ui(_Server(agent=SimpleNamespace())).plan_info()
        steps = out["plan"]["steps"]
        assert steps[0]["ready"] is True

    def test_notes_are_bounded(self, plans):
        """A step note can be an essay; the drawer row is one line."""
        p = plans.create_plan("g", steps=["a"])
        plans.update_step(p, "s1", notes="x" * 900)
        out = _ui(_Server(agent=SimpleNamespace())).plan_info()
        assert len(out["plan"]["steps"][0]["notes"]) == 200

    def test_the_route_exists(self):
        assert 'elif self.path == "/api/plan":' in HTTP_LOOP


class TestPanel:
    def test_the_page_has_a_chip_and_a_fold(self):
        assert 'id="planchip"' in _PAGE and 'id="planfold"' in _PAGE

    def test_the_chip_hides_when_there_is_no_plan(self):
        i = APP_JS.index("async function loadPlan()")
        body = APP_JS[i:i + 1200]
        assert "chip.style.display = 'none';" in body

    def test_the_panel_is_not_rendered_into_a_closed_drawer(self):
        i = APP_JS.index("async function loadPlan()")
        body = APP_JS[i:i + 1400]
        assert "foldOpen('planfold')" in body

    def test_it_refreshes_when_a_turn_ends(self):
        """Steps advance while the agent works, not while you watch."""
        i = APP_JS.index("ev.type === 'state'")
        assert "if (wasBusy) loadPlan();" in APP_JS[i:APP_JS.index("ev.type === 'switched'")]

    def test_only_the_step_that_moved_is_highlighted(self):
        """The fold re-renders wholesale; flashing all of it would be noise."""
        i = APP_JS.index("function planChanged(")
        body = APP_JS[i:i + 600]
        assert "planSeen[st.id] !== st.status" in body
        assert "planSeen === null ? []" in body      # first render marks nothing

    def test_a_new_plan_does_not_read_as_all_changed(self):
        i = APP_JS.index("async function loadPlan()")
        body = APP_JS[i:i + 1600]
        assert "planSeen = null;" in body

    def test_the_highlight_expires(self):
        i = APP_JS.index("classList.add('step-moved')")
        assert "remove('step-moved')" in APP_JS[i:i + 200]

    def test_the_highlight_does_not_move_anything(self):
        """Colour only, so it needs no reduced-motion exemption."""
        css = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
               ).read_text(encoding="utf-8")
        i = css.index("@keyframes stepmoved")
        block = css[i:css.index("\n}", i)]
        assert "transform" not in block and "translate" not in block

    def test_the_step_status_marks_match_the_terminal_renderer(self):
        import inspect

        from agent.ui import slash_plan

        src = inspect.getsource(slash_plan._render_plan)
        i = APP_JS.index("const STEP_MARK = {")
        marks = APP_JS[i:APP_JS.index("}", i)]
        for status in ("pending", "in_progress", "completed", "failed",
                       "skipped", "blocked"):
            assert status in marks and '"%s"' % status in src, status
