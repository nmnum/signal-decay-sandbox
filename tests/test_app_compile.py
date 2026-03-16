"""
tests/test_app_compile.py
-------------------------
Verifies that app.py is syntactically valid and references the correct
internal modules — no Streamlit or Plotly installation required.

Tests
~~~~~
1. app.py parses without SyntaxError (ast.parse)
2. app.py imports from the correct internal modules
3. SimConfig, simulate, model classes, and metric functions are all
   referenced by name in the source
4. requirements.txt lists all required packages
5. No reimplementation of core logic (key class names are imported,
   not defined, in app.py)
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


# ---------------------------------------------------------------------------
# Tests
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


class TestAppImports(unittest.TestCase):

    def setUp(self) -> None:
        self.tree = _parse_tree()
        self.imports = _collect_imports(self.tree)

    def test_imports_simulate_regimes(self) -> None:
        self.assertIn(
            "data.simulate_regimes", self.imports,
            "app.py must import from data.simulate_regimes",
        )

    def test_imports_simconfig(self) -> None:
        names = self.imports.get("data.simulate_regimes", [])
        self.assertIn("SimConfig", names, "SimConfig must be imported from data.simulate_regimes")

    def test_imports_simulate_function(self) -> None:
        names = self.imports.get("data.simulate_regimes", [])
        self.assertIn("simulate", names, "simulate must be imported from data.simulate_regimes")

    def test_imports_static_model(self) -> None:
        self.assertIn(
            "models.static_model", self.imports,
            "app.py must import from models.static_model",
        )
        names = self.imports.get("models.static_model", [])
        self.assertIn("StaticModel", names)

    def test_imports_rolling_model(self) -> None:
        self.assertIn(
            "models.rolling_model", self.imports,
            "app.py must import from models.rolling_model",
        )
        names = self.imports.get("models.rolling_model", [])
        self.assertIn("RollingModel", names)

    def test_imports_unlearning_model(self) -> None:
        self.assertIn(
            "models.unlearning_model", self.imports,
            "app.py must import from models.unlearning_model",
        )
        names = self.imports.get("models.unlearning_model", [])
        self.assertIn("UnlearningModel", names)

    def test_imports_metrics(self) -> None:
        self.assertIn(
            "evaluation.metrics", self.imports,
            "app.py must import from evaluation.metrics",
        )

    def test_imports_rolling_rmse(self) -> None:
        names = self.imports.get("evaluation.metrics", [])
        self.assertIn(
            "rolling_rmse_vectorised", names,
            "rolling_rmse_vectorised must be imported from evaluation.metrics",
        )

    def test_imports_recovery_time(self) -> None:
        names = self.imports.get("evaluation.metrics", [])
        self.assertIn(
            "recovery_time", names,
            "recovery_time must be imported from evaluation.metrics",
        )

    def test_imports_detection_lag(self) -> None:
        names = self.imports.get("evaluation.metrics", [])
        self.assertIn(
            "detection_lag", names,
            "detection_lag must be imported from evaluation.metrics",
        )

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
        has_st = "streamlit" in all_modules or "streamlit" in plain_imports
        self.assertTrue(has_st, "app.py must import streamlit")

    def test_imports_plotly(self) -> None:
        all_modules = {
            node.module or ""
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
        }
        has_plotly = any("plotly" in m for m in all_modules)
        self.assertTrue(has_plotly, "app.py must import from plotly")


class TestNoReimplementation(unittest.TestCase):
    """Core model classes must be imported, not redefined in app.py."""

    def setUp(self) -> None:
        self.tree = _parse_tree()
        self.defined_classes = _collect_defined_classes(self.tree)

    def test_static_model_not_redefined(self) -> None:
        self.assertNotIn(
            "StaticModel", self.defined_classes,
            "StaticModel must not be redefined in app.py — import it.",
        )

    def test_rolling_model_not_redefined(self) -> None:
        self.assertNotIn(
            "RollingModel", self.defined_classes,
            "RollingModel must not be redefined in app.py — import it.",
        )

    def test_unlearning_model_not_redefined(self) -> None:
        self.assertNotIn(
            "UnlearningModel", self.defined_classes,
            "UnlearningModel must not be redefined in app.py — import it.",
        )

    def test_simconfig_not_redefined(self) -> None:
        self.assertNotIn(
            "SimConfig", self.defined_classes,
            "SimConfig must not be redefined in app.py — import it.",
        )


class TestKeyUiElements(unittest.TestCase):
    """Verify that key Streamlit UI elements are present in the source."""

    def setUp(self) -> None:
        self.source = _load_source()

    def test_has_sidebar(self) -> None:
        self.assertIn("st.sidebar", self.source, "app.py must use st.sidebar")

    def test_has_run_button(self) -> None:
        self.assertIn("st.button", self.source, "app.py must have a st.button")

    def test_has_plotly_chart(self) -> None:
        self.assertIn(
            "st.plotly_chart", self.source,
            "app.py must render a chart with st.plotly_chart",
        )

    def test_has_subplots(self) -> None:
        self.assertIn(
            "make_subplots", self.source,
            "app.py must use make_subplots for the two-panel figure",
        )

    def test_has_two_subplot_rows(self) -> None:
        self.assertIn("rows=2", self.source, "Figure must have 2 subplot rows")

    def test_has_spinner(self) -> None:
        self.assertIn("st.spinner", self.source, "app.py should use st.spinner for loading feedback")

    def test_has_metrics_display(self) -> None:
        # Should display recovery metrics as a dataframe or metric widget
        has_display = "st.dataframe" in self.source or "st.metric" in self.source
        self.assertTrue(has_display, "app.py must display metrics (st.dataframe or st.metric)")

    def test_regime_boundaries_in_plot(self) -> None:
        # Should add vertical lines or vrect for regime boundaries
        has_boundary = "add_vline" in self.source or "add_vrect" in self.source
        self.assertTrue(has_boundary, "Plot must mark regime boundaries")

    def test_all_three_models_instantiated(self) -> None:
        self.assertIn("StaticModel(", self.source)
        self.assertIn("RollingModel(", self.source)
        self.assertIn("UnlearningModel(", self.source)

    def test_online_loop_structure(self) -> None:
        # Online loop: predict called before update
        pred_pos = self.source.find(".predict(")
        update_pos = self.source.find(".update(")
        self.assertGreater(pred_pos, 0, "No .predict() call found")
        self.assertGreater(update_pos, 0, "No .update() call found")
        self.assertLess(
            pred_pos, update_pos,
            ".predict() must appear before .update() in app.py",
        )


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
