"""Unit tests for agent/tools/analyze_deps/ — dependency hygiene report."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.analyze_deps.main import (
    _collect_imports,
    _find_unused,
    _idle_dep_hygiene,
    _pip_outdated,
    _venv_dist_modules,
    analyze_dependencies,
    setup,
)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    import agent.tools.analyze_deps.main as mod
    monkeypatch.setattr(mod, "_config", None)
    yield


def _make_config(root: Path, airgap: bool = False):
    cfg = MagicMock()
    cfg.tools.working_dir = str(root)
    cfg.security.airgap = airgap
    return cfg


class _Comp:
    def __init__(self, name, source="requirements.txt", ecosystem="pypi", pinned=True):
        self.name = name
        self.source = source
        self.ecosystem = ecosystem
        self.pinned = pinned
        self.version = "1.0" if pinned else ""


class TestCollectImports:
    def test_finds_import_and_from(self, tmp_path):
        (tmp_path / "a.py").write_text("import requests\nfrom yaml import safe_load\n")
        imports = _collect_imports(str(tmp_path))
        assert {"requests", "yaml"} <= imports

    def test_skips_venv(self, tmp_path):
        d = tmp_path / ".venv" / "lib"
        d.mkdir(parents=True)
        (d / "x.py").write_text("import secretmod\n")
        assert "secretmod" not in _collect_imports(str(tmp_path))


class TestVenvDistModules:
    def test_reads_top_level_txt(self, tmp_path):
        info = tmp_path / "lib" / "python3.12" / "site-packages" / "PyYAML-6.0.dist-info"
        info.mkdir(parents=True)
        (info / "top_level.txt").write_text("yaml\n_yaml\n")
        m = _venv_dist_modules(str(tmp_path))
        assert m["pyyaml"] == {"yaml", "_yaml"}

    def test_record_fallback(self, tmp_path):
        info = tmp_path / "lib" / "python3.12" / "site-packages" / "bs4-4.12.dist-info"
        info.mkdir(parents=True)
        (info / "RECORD").write_text("bs4/__init__.py,sha,1\nbs4-4.12.dist-info/RECORD,,\n")
        m = _venv_dist_modules(str(tmp_path))
        assert "bs4" in m["bs4"]


class TestFindUnused:
    def test_used_via_venv_metadata(self):
        unused = _find_unused([_Comp("pyyaml")], {"yaml"}, {"pyyaml": {"yaml"}})
        assert unused == []

    def test_unused_venv_verified(self):
        unused = _find_unused([_Comp("requests")], {"os"}, {"requests": {"requests"}})
        assert unused[0]["name"] == "requests"
        assert unused[0]["confidence"] == "venv-verified"

    def test_heuristic_alias(self):
        # No venv metadata: pillow → PIL known alias, counts as used.
        assert _find_unused([_Comp("pillow")], {"PIL"}, {}) == []

    def test_heuristic_unused_flagged(self):
        unused = _find_unused([_Comp("leftpad")], {"os"}, {})
        assert unused[0]["confidence"] == "heuristic"

    def test_underscore_dash_normalization(self):
        assert _find_unused([_Comp("typing_extensions")], {"typing_extensions"}, {}) == []

    def test_indirect_plugins_skipped(self):
        comps = [_Comp("pytest-asyncio"), _Comp("pre-commit"), _Comp("coverage")]
        assert _find_unused(comps, set(), {}) == []


class TestPipOutdated:
    def test_parses_json(self):
        out = '[{"name": "requests", "version": "2.0", "latest_version": "2.32"}]'
        with patch("agent.tools.analyze_deps.main._run_pip", return_value=(0, out)):
            rows, err = _pip_outdated("/fake")
        assert err is None
        assert rows == [{"name": "requests", "installed": "2.0", "latest": "2.32"}]

    def test_failure_reported(self):
        with patch("agent.tools.analyze_deps.main._run_pip", return_value=(1, "boom")):
            rows, err = _pip_outdated("/fake")
        assert rows == [] and "boom" in err


class TestAnalyzeDependenciesTool:
    def test_no_manifests(self, tmp_path):
        setup(_make_config(tmp_path))
        r = asyncio.run(analyze_dependencies(path=str(tmp_path)))
        assert "error" in r

    def test_report_shape(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.0\nleftpad>=1.0\n")
        (tmp_path / "app.py").write_text("import requests\n")
        setup(_make_config(tmp_path))
        r = asyncio.run(analyze_dependencies(path=str(tmp_path), check_outdated=False))
        assert r["components"] == {"pypi": 2}
        assert [u["name"] for u in r["possibly_unused"]] == ["leftpad"]
        assert any(u["name"] == "leftpad" for u in r["unpinned"])
        # possibly_unused is informational — it must NOT flip ok to False.
        assert r["ok"] is True
        assert any("no project venv" in n for n in r["notes"])

    def test_airgap_skips_outdated(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.0\n")
        (tmp_path / "app.py").write_text("import requests\n")
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py = venv_bin / "python"
        py.write_text("#!/bin/sh\nexit 0\n")
        py.chmod(0o755)
        setup(_make_config(tmp_path, airgap=True))
        with patch("agent.tools.analyze_deps.main._pip_conflicts", return_value=([], None)):
            r = asyncio.run(analyze_dependencies(path=str(tmp_path)))
        assert any("air-gapped" in n for n in r["notes"])
        assert r["outdated"] == []

    def test_conflicts_surfaced(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.0\n")
        (tmp_path / "app.py").write_text("import requests\n")
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py = venv_bin / "python"
        py.write_text("#!/bin/sh\nexit 0\n")
        py.chmod(0o755)
        setup(_make_config(tmp_path))
        with patch("agent.tools.analyze_deps.main._pip_conflicts",
                   return_value=(["a 1.0 requires b>=2, but you have b 1.0"], None)):
            r = asyncio.run(analyze_dependencies(path=str(tmp_path), check_outdated=False))
        assert len(r["conflicts"]) == 1
        assert r["ok"] is False


class TestIdleDepHygiene:
    def _agent(self, tmp_path):
        agent = MagicMock()
        agent.config.tools.working_dir = str(tmp_path)
        agent.config.tools.agent_dir = ".agent"
        return agent

    def test_files_idea_on_conflicts(self, tmp_path):
        agent = self._agent(tmp_path)
        report = {"ok": False, "conflicts": ["a requires b>=2"], "vulnerabilities": []}
        with patch("agent.tools.analyze_deps.main.analyze_dependencies",
                   new=AsyncMock(return_value=report)), \
             patch("agent.tools.ideas.main.submit_idea",
                   return_value={"saved": True}) as si:
            did = asyncio.run(_idle_dep_hygiene(agent))
        assert did is True
        si.assert_called_once()
        assert "1 conflicts" in si.call_args.kwargs["title"]
        assert (tmp_path / ".agent" / "dep_hygiene.stamp").exists()

    def test_clean_report_files_nothing(self, tmp_path):
        agent = self._agent(tmp_path)
        report = {"ok": True, "conflicts": [], "vulnerabilities": [],
                  "possibly_unused": [{"name": "x"}]}
        with patch("agent.tools.analyze_deps.main.analyze_dependencies",
                   new=AsyncMock(return_value=report)), \
             patch("agent.tools.ideas.main.submit_idea") as si:
            did = asyncio.run(_idle_dep_hygiene(agent))
        assert did is False
        si.assert_not_called()

    def test_stamp_throttles(self, tmp_path):
        agent = self._agent(tmp_path)
        d = tmp_path / ".agent"
        d.mkdir()
        (d / "dep_hygiene.stamp").touch()
        with patch("agent.tools.analyze_deps.main.analyze_dependencies") as ad:
            did = asyncio.run(_idle_dep_hygiene(agent))
        assert did is False
        ad.assert_not_called()
