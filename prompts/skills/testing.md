# Testing guidelines
---
description: pytest conventions, test structure, what to assert
---

- One test file per module under `tests/unit/` mirroring the source path.
- Test function names: `test_<what>_<condition>_<expected>`.
- Use `pytest.fixture` for shared setup; `autouse=True` only for mandatory env wiring.
- Assert on observable outcomes, not implementation details.
- Parametrize with `@pytest.mark.parametrize` when the same logic runs over multiple inputs.
- No `time.sleep` in tests — use monkeypatching or async fixtures.
- Mark slow/external tests with `@pytest.mark.e2e`; keep unit tests fast and hermetic.
- Each test should set up its own state; don't rely on test ordering.
