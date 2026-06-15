import os
from pathlib import Path
import subprocess
import sys

import my_krml_24999690 as package


def test_supported_helpers_are_exported():
    expected = {
        "balance_classes",
        "download_canvas_courses",
        "plot_confusion_matrix",
        "plot_feature_importance",
        "plot_roc_curve",
        "summarize_classification_result",
        "tune_hyperparameters",
    }

    assert expected.issubset(package.__all__)
    assert all(callable(getattr(package, name)) for name in expected)


def test_base_package_import_does_not_require_optional_dependencies():
    code = """
import importlib.abc
import sys

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    blocked = {"bs4", "canvasapi", "imblearn", "requests"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockOptionalImports())
import my_krml_24999690
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "ok"
