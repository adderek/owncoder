"""Tests for SBOM + dependency vuln audit (agent.security.sbom)."""
from __future__ import annotations

import json
import types

from agent.security import sbom


def _cfg(tmp_path):
    return types.SimpleNamespace(
        tools=types.SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent")
    )


def test_requirements_pinned_vs_floating(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "# comment\nrequests==2.20.0\nflask>=2.0\n-e .\n"
    )
    comps = sbom.build_sbom(str(tmp_path))
    by = {c.name: c for c in comps}
    assert by["requests"].version == "2.20.0" and by["requests"].pinned
    assert by["flask"].version == "" and not by["flask"].pinned


def test_package_lock_v3(tmp_path):
    (tmp_path / "package-lock.json").write_text(json.dumps({
        "packages": {
            "": {"name": "app"},
            "node_modules/lodash": {"version": "4.17.10"},
            "node_modules/left-pad": {"version": "1.3.0"},
        }
    }))
    comps = sbom.build_sbom(str(tmp_path))
    npm = {c.name: c.version for c in comps if c.ecosystem == "npm"}
    assert npm == {"lodash": "4.17.10", "left-pad": "1.3.0"}


def test_cargo_lock(tmp_path):
    (tmp_path / "Cargo.lock").write_text(
        '[[package]]\nname = "serde"\nversion = "1.0.100"\n\n'
        '[[package]]\nname = "tokio"\nversion = "1.20.0"\n'
    )
    comps = sbom.build_sbom(str(tmp_path))
    crates = {c.name: c.version for c in comps if c.ecosystem == "crates"}
    assert crates == {"serde": "1.0.100", "tokio": "1.20.0"}


def test_go_mod(tmp_path):
    (tmp_path / "go.mod").write_text(
        "module x\n\ngo 1.21\n\nrequire (\n\tgithub.com/foo/bar v1.2.3\n)\n"
    )
    comps = sbom.build_sbom(str(tmp_path))
    go = {c.name: c.version for c in comps if c.ecosystem == "go"}
    assert go == {"github.com/foo/bar": "1.2.3"}


def test_version_range_matching():
    adv = {"introduced": "2.0.0", "fixed": "2.20.1"}
    assert sbom._affected("2.20.0", adv)      # in range
    assert not sbom._affected("2.20.1", adv)  # at fix
    assert not sbom._affected("1.9.0", adv)   # before introduced
    assert not sbom._affected("", adv)        # unknown version


def test_vuln_match_against_local_db(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "requirements.txt").write_text("requests==2.20.0\nsafe-pkg==9.9.9\n")
    vdb = tmp_path / ".agent" / "vulndb"
    vdb.mkdir(parents=True)
    (vdb / "pypi.json").write_text(json.dumps({
        "requests": [{"id": "CVE-2018-18074", "severity": "high",
                      "introduced": "0", "fixed": "2.20.1",
                      "summary": "credential leak on redirect"}]
    }))
    comps = sbom.build_sbom(str(tmp_path))
    hits = sbom.match_vulns(comps, sbom.load_vulndb(cfg))
    assert len(hits) == 1
    assert hits[0]["id"] == "CVE-2018-18074"
    assert hits[0]["name"] == "requests"


def test_command_no_db_is_sbom_only(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
    out = sbom.run_sbom_command(cfg, str(tmp_path))
    assert "components: 1" in out
    assert "vuln DB: none installed" in out


def test_command_no_manifests(tmp_path):
    out = sbom.run_sbom_command(_cfg(tmp_path), str(tmp_path))
    assert "No dependency manifests" in out


def test_skips_node_modules(tmp_path):
    nm = tmp_path / "node_modules" / "x"
    nm.mkdir(parents=True)
    (nm / "package.json").write_text(json.dumps({"dependencies": {"evil": "1.0.0"}}))
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"good": "2.0.0"}}))
    comps = sbom.build_sbom(str(tmp_path))
    names = {c.name for c in comps}
    assert "good" in names and "evil" not in names
