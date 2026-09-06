"""
Architectural test enforcing Reviewer-2 attack #11 (main_v2.tex): the
oracle must never import the estimator it is meant to validate.
"""
import ast
from pathlib import Path


def test_oracle_module_does_not_import_estimator():
    oracle_path = Path(__file__).parent.parent / "src" / "ccce" / "oracle" / "analytical_oracle.py"
    tree = ast.parse(oracle_path.read_text())
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)
    for mod in imported_modules:
        assert "estimator" not in mod, (
            f"oracle module imports '{mod}', violating the required "
            "oracle/estimator independence (Reviewer-2 attack #11)."
        )
