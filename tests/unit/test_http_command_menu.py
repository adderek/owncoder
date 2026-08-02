"""Commands you can find without knowing their names.

The '/' palette completes a name you already have. Everything else in the
product was reachable only by knowing the word to type — which is the same
as not being reachable.
"""
from pathlib import Path

from agent.ui.http_loop import _PAGE, _TERMINAL_ONLY, _slash_catalog
from agent.ui.slash import _GROUP_ORDER, _GROUPS, _PRESETS, _SLASH_COMMANDS

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")

CATALOG = {c["name"]: c for c in _slash_catalog()}


class TestGrouping:
    def test_every_offered_command_lands_in_a_named_group(self):
        """'other' is the fallback, not the plan — a command with no home is
        one nobody will find."""
        homeless = [n for n, c in CATALOG.items() if c["group"] == "other"]
        assert homeless == [], homeless

    def test_group_names_are_all_in_the_declared_order(self):
        assert set(_GROUPS) == set(_GROUP_ORDER)

    def test_groups_only_name_real_commands(self):
        known = {c[0] for c in _SLASH_COMMANDS}
        for group, names in _GROUPS.items():
            for name in names:
                assert name in known, f"{group} lists unknown {name}"

    def test_a_command_is_not_filed_in_two_places(self):
        seen: dict[str, str] = {}
        for group, names in _GROUPS.items():
            for name in names:
                assert name not in seen, f"{name} in {seen.get(name)} and {group}"
                seen[name] = group

    def test_the_order_index_matches_the_declared_order(self):
        for c in CATALOG.values():
            assert _GROUP_ORDER[c["group_order"]] == c["group"]

    def test_terminal_only_commands_stay_out_of_the_menu(self):
        assert not (set(CATALOG) & _TERMINAL_ONLY)


class TestPresets:
    def test_presets_only_hang_off_commands_that_take_an_argument(self):
        """A preset on a no-argument command would send it junk."""
        for name in _PRESETS:
            assert CATALOG.get(name) is None or CATALOG[name]["arg"], name

    def test_presets_name_real_commands(self):
        known = {c[0] for c in _SLASH_COMMANDS}
        for name in _PRESETS:
            assert name in known, name

    def test_the_catalogue_carries_them(self):
        sec = CATALOG["/security"]
        labels = [p["label"] for p in sec["presets"]]
        assert "scan the project" in labels
        assert all("arg" in p and "label" in p for p in sec["presets"])

    def test_the_long_jobs_are_one_click_away(self):
        """The point of the menu: the things people did not know existed."""
        for name in ("/security", "/analyze-asm", "/memory"):
            assert CATALOG[name]["presets"], name


class TestPage:
    def test_the_menu_is_in_the_page(self):
        for el in ('id="cmdmenu"', 'id="cmdsheet"', 'id="cmdfilter"',
                   'id="cmdlist"', 'id="cmdclose"'):
            assert el in _PAGE, el

    def test_the_browser_builds_it_from_the_catalogue(self):
        assert "cmdRender" in APP_JS and "loadSlashCmds()" in APP_JS
        assert "group_order" in APP_JS

    def test_an_argument_command_is_filled_in_not_fired(self):
        """Running '/save' bare would name the session an empty string."""
        fn = APP_JS[APP_JS.index("function cmdRun("):APP_JS.index("async function cmdRender(")]
        assert "cmdFill" in APP_JS
        assert "c.arg ? cmdFill(c.name) : cmdRun(c.name)" in APP_JS
        assert "send()" in fn

    def test_it_opens_by_click_and_by_key(self):
        assert "getElementById('cmdmenu')" in APP_JS
        assert "e.key.toLowerCase() === 'k'" in APP_JS

    def test_escape_closes_the_menu_before_anything_else(self):
        esc = APP_JS[APP_JS.index("if (e.key === 'Escape') {"):]
        assert esc.index("cmdMenuOpen(false)") < esc.index("helpClose()")

    def test_it_is_styled_for_both_themes(self):
        """Hard-coded colours here would be invisible in the light theme."""
        block = APP_CSS[APP_CSS.index("#cmdsheet"):APP_CSS.index("/* Memory browser")]
        assert "var(--border)" in block and "var(--accent)" in block
