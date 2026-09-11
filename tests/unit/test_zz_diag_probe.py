"""TEMPORARY — delete me.

This module was a diagnostic probe used while run_argv was blocked by the
sandbox mask-scan cap (run_tests drops tracebacks). The failure causes it was
created to find are now fixed; it is parked as a module-level skip only because
this session has no shell access to remove the file.

Delete with:  rm agent/tests/unit/test_zz_diag_probe.py agent/tests/_diag_out.json
"""
import pytest

pytest.skip("temporary diagnostic probe — safe to delete", allow_module_level=True)
