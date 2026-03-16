"""
tests/test_app_compile.py
-------------------------
Verifies that app.py is syntactically valid and references the correct
internal modules — no Streamlit or Plotly installation required.

Tests
~~~~~
1. app.py parses without SyntaxError (ast.parse)
2. app.py imports from the correct internal modules
3. SimConfig, simulate, model classes, and metric functions referenced
4. requirements.txt lists all required packages
5. No reimplementation of core logic (classes imported, not defined)
6. Reset button: st.session_state.clear() and st.rerun() present
7. Results stored in and read from st.session_state["results"]
8. Run button and Reset button both present in sidebar
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

APP_PATH = _ROOT / "app.py"
REQ_PATH = _ROOT / "requirements.txt"


def _load_source() -> str:
    return APP_PATH.read_text(encoding="utf-8")


def _parse_tree() -> ast.Module:
    return ast.parse(_load_source(), filename=str(APP_PATH))


def _collect_imports(tree: ast.Module) -> dict[str, list[str]]:
    """Return {module: [names]} for all `from module import name` statements."""
    imports: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in node.names]
            imports.setdefault(node.module, []).extend(names)
    return imports


def _collect_defined_classes(tree: ast.Module) -> list[str]:
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _collect_defined_functions(tree: ast.Module) -> list[str]:
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


# ---------------------------------------------------------------------------
# 1. Syntax
# ---------------------------------------------------------------------------

class TestAppSyntax(unittest.TestCase):

    def test_app_file_exists(self) -> None:
        self.assertTrue(APP_PATH.exists(), f"app.py not found at {APP_PATH}")

    def test_app_parses_without_syntax_error(self) -> None:
        source = _load_source()
        try:
            ast.parse(source, filename=str(APP_PATH))
        except SyntaxError as exc:
            self.fail(f"app.py has a SyntaxError: {exc}")

    def test_app_is_non_empty(self) -> None:
        self.assertGreater(len(_load_source()), 500, "app.py appears to be nearly empty.")


# ---------------------------------------------------------------------------
# 2. Imports
# ---------------------------------------------------------------------------

class TestAppImports(unittest.TestCase):

    def setUp(self) -> None:
        self.tree = _parse_tree()
        self.imports = _collect_imports(self.tree)

    def test_imports_simulate_regimes(self) -> None:
        self.assertIn("data.simulate_regimes", self.imports)

    def test_imports_simconfig(self) -> None:
        self.assertIn("SimConfig", self.imports.get("data.simulate_regimes", []))

    def test_imports_simulate_function(self) -> None:
        self.assertIn("simulate", self.imports.get("data.simulate_regimes", []))

    def test_imports_static_model(self) -> None:
        self.assertIn("StaticModel", self.imports.get("models.static_model", []))

    def test_imports_rolling_model(self) -> None:
        self.assertIn("RollingModel", self.imports.get("models.rolling_model", []))

    def test_imports_unlearning_model(self) -> None:
        self.assertIn("UnlearningModel", self.imports.get("models.unlearning_model", []))

    def test_imports_metrics(self) -> None:
        self.assertIn("evaluation.metrics", self.imports)

    def test_imports_rolling_rmse(self) -> None:
        self.assertIn("rolling_rmse_vectorised", self.imports.get("evaluation.metrics", []))

    def test_imports_recovery_time(self) -> None:
        self.assertIn("recovery_time", self.imports.get("evaluation.metrics", []))

    def test_imports_detection_lag(self) -> None:
        self.assertIn("detection_lag", self.imports.get("evaluation.metrics", []))

    def test_imports_streamlit(self) -> None:
        all_modules = {
            node.module
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
        }
        plain_imports = {
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(
            "streamlit" in all_modules or "streamlit" in plain_imports,
            "app.py must import streamlit",
        )

    def test_imports_plotly(self) -> None:
        all_modules = {
            node.module or ""
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertTrue(any("plotly" in m for m in all_modules))


# ---------------------------------------------------------------------------
# 3. No reimplementation
# ---------------------------------------------------------------------------

class TestNoReimplementation(unittest.TestCase):

    def setUp(self) -> None:
        self.tree = _parse_tree()
        self.defined_classes = _collect_defined_classes(self.tree)

    def test_static_model_not_redefined(self) -> None:
        self.assertNotIn("StaticModel", self.defined_classes)

    def test_rolling_model_not_redefined(self) -> None:
        self.assertNotIn("RollingModel", self.defined_classes)

    def test_unlearning_model_not_redefined(self) -> None:
        self.assertNotIn("UnlearningModel", self.defined_classes)

    def test_simconfig_not_redefined(self) -> None:
        self.assertNotIn("SimConfig", self.defined_classes)


# ---------------------------------------------------------------------------
# 4. Key UI elements
# ---------------------------------------------------------------------------

class TestKeyUiElements(unittest.TestCase):

    def setUp(self) -> None:
        self.source = _load_source()

    def test_has_sidebar(self) -> None:
        self.assertIn("st.sidebar", self.source)

    def test_has_run_button(self) -> None:
        self.assertIn("st.sidebar.button", self.source,
            "Run button should be in the sidebar")

    def test_has_plotly_chart(self) -> None:
        self.assertIn("st.plotly_chart", self.source)

    def test_has_subplots(self) -> None:
        self.assertIn("make_subplots", self.source)

    def test_has_two_subplot_rows(self) -> None:
        self.assertIn("rows=2", self.source)

    def test_has_spinner(self) -> None:
        self.assertIn("st.spinner", self.source)

    def test_has_metrics_display(self) -> None:
        has_display = "st.dataframe" in self.source or "st.metric" in self.source
        self.assertTrue(has_display)

    def test_regime_boundaries_in_plot(self) -> None:
        has_boundary = "add_vline" in self.source or "add_vrect" in self.source
        self.assertTrue(has_boundary)

    def test_all_three_models_instantiated(self) -> None:
        self.assertIn("StaticModel(", self.source)
        self.assertIn("RollingModel(", self.source)
        self.assertIn("UnlearningModel(", self.source)

    def test_online_loop_predict_before_update(self) -> None:
        pred_pos   = self.source.find(".predict(")
        update_pos = self.source.find(".update(")
        self.assertGreater(pred_pos, 0, "No .predict() call found")
        self.assertGreater(update_pos, 0, "No .update() call found")
        self.assertLess(pred_pos, update_pos,
            ".predict() must appear before .update() in the source")


# ---------------------------------------------------------------------------
# 5. Reset button
# ---------------------------------------------------------------------------

class TestResetButton(unittest.TestCase):

    def setUp(self) -> None:
        self.source = _load_source()
        self.tree   = _parse_tree()

    def test_reset_button_present(self) -> None:
        self.assertIn("Reset", self.source,
            "A 'Reset' button label must appear in app.py")

    def test_session_state_clear_called(self) -> None:
        self.assertIn("st.session_state.clear()", self.source,
            "Reset must call st.session_state.clear()")

    def test_st_rerun_called(self) -> None:
        self.assertIn("st.rerun()", self.source,
            "Reset must call st.rerun()")

    def test_reset_helper_function_defined(self) -> None:
        fns = _collect_defined_functions(self.tree)
        self.assertIn("_reset", fns,
            "A _reset() helper function should be defined in app.py")

    def test_reset_button_in_sidebar(self) -> None:
        # The reset button call should be via st.sidebar.button
        self.assertIn("st.sidebar.button", self.source,
            "Reset button should be placed in st.sidebar")

    def test_clear_precedes_rerun(self) -> None:
        # Use rfind so we target the actual call site, not any docstring mention
        clear_pos = self.source.rfind("st.session_state.clear()")
        rerun_pos = self.source.rfind("st.rerun()")
        self.assertGreater(clear_pos, 0, "st.session_state.clear() not found")
        self.assertGreater(rerun_pos, 0, "st.rerun() not found")
        self.assertLess(clear_pos, rerun_pos,
            "st.session_state.clear() must appear before st.rerun()")


# ---------------------------------------------------------------------------
# 6. Session-state result storage
# ---------------------------------------------------------------------------

class TestSessionStateUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.source = _load_source()

    def test_results_stored_in_session_state(self) -> None:
        self.assertIn('st.session_state["results"]', self.source,
            'Results must be stored under st.session_state["results"]')

    def test_results_key_checked_before_render(self) -> None:
        self.assertIn('"results" in st.session_state', self.source,
            'App must guard rendering with: if "results" in st.session_state')

    def test_params_snapshot_stored(self) -> None:
        self.assertIn('"params"', self.source,
            'A "params" snapshot should be stored alongside results')

    def test_session_state_not_used_for_sliders_directly(self) -> None:
        # Sliders should use key= parameter, not manual session_state assignment
        # Check that we DON'T manually assign slider values into session_state
        # (which would fight Streamlit's own widget state management)
        self.assertNotIn(
            'st.session_state["dur0"]',
            self.source,
            "Slider values should not be manually assigned into session_state",
        )


# ---------------------------------------------------------------------------
# 7. DEFAULT dict and key constants
# ---------------------------------------------------------------------------

class TestDefaults(unittest.TestCase):

    def setUp(self) -> None:
        self.source = _load_source()

    def test_default_dict_defined(self) -> None:
        self.assertIn("DEFAULT", self.source,
            "A DEFAULT dict of parameter defaults should be defined")

    def test_rolling_rmse_window_constant(self) -> None:
        self.assertIn("ROLLING_RMSE_WINDOW", self.source)

    def test_recovery_tolerance_constant(self) -> None:
        self.assertIn("RECOVERY_TOLERANCE", self.source)

    def test_model_colors_defined(self) -> None:
        self.assertIn("MODEL_COLORS", self.source)


# ---------------------------------------------------------------------------
# 8. requirements.txt
# ---------------------------------------------------------------------------

class TestRequirementsTxt(unittest.TestCase):

    def setUp(self) -> None:
        self.assertTrue(REQ_PATH.exists(), f"requirements.txt not found at {REQ_PATH}")
        self.reqs = REQ_PATH.read_text(encoding="utf-8").lower()

    def test_has_numpy(self) -> None:
        self.assertIn("numpy", self.reqs)

    def test_has_pandas(self) -> None:
        self.assertIn("pandas", self.reqs)

    def test_has_scikit_learn(self) -> None:
        self.assertIn("scikit-learn", self.reqs)

    def test_has_streamlit(self) -> None:
        self.assertIn("streamlit", self.reqs)

    def test_has_plotly(self) -> None:
        self.assertIn("plotly", self.reqs)

    def test_has_pytest(self) -> None:
        self.assertIn("pytest", self.reqs)

    def test_all_packages_have_version_pins(self) -> None:
        for line in REQ_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self.assertTrue(
                ">=" in line or "==" in line or "~=" in line,
                f"Package line missing version constraint: '{line}'",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
