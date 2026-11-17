"""Module-boundary regression tests for the benchmark runners.

``android_world``, ``mobile_world`` and ``mobile_gym`` are sibling runners: they
may share ``mobilerun.benchmark.common`` but must never import each other.
"""

import ast
import inspect
from pathlib import Path

import mobilerun.benchmark.android_world as aw
import mobilerun.benchmark.common as common
import mobilerun.benchmark.mobile_gym as mg
import mobilerun.benchmark.mobile_world as mw

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "mobilerun" / "benchmark"
RUNNERS = ("android_world", "mobile_world", "mobile_gym")


def _imported_modules(path: Path) -> set[str]:
    """Every module path imported anywhere in *path* (including function bodies)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = {1: "mobilerun.benchmark", 2: "mobilerun"}.get(node.level, "")
                module = f"{base}.{node.module}" if node.module else base
                if module:
                    modules.add(module)
            elif node.module:
                modules.add(node.module)
                if node.module == "mobilerun.benchmark":
                    modules.update(
                        f"mobilerun.benchmark.{alias.name}" for alias in node.names
                    )
    return modules


def _imports(modules: set[str], target: str) -> bool:
    prefix = f"mobilerun.benchmark.{target}"
    return any(
        module == prefix or module.startswith(prefix + ".") for module in modules
    )


def test_benchmark_runners_do_not_import_each_other():
    for name in RUNNERS:
        modules = _imported_modules(BENCHMARK_DIR / f"{name}.py")
        for other in RUNNERS:
            if other == name:
                continue
            assert not _imports(modules, other), f"{name}.py imports {other}.py"


def test_common_is_a_leaf_for_runners_and_harness():
    modules = _imported_modules(BENCHMARK_DIR / "common.py")
    for name in RUNNERS:
        assert not _imports(modules, name), f"common.py imports {name}.py"
    assert not any(
        module == "mobilerun.harness" or module.startswith("mobilerun.harness.")
        for module in modules
    ), "common.py must not import the harness at module scope"


def test_android_world_reexports_the_shared_surface():
    assert aw.TaskResult is common.TaskResult
    assert aw._write_results is common._write_results
    assert aw._load_results is common._load_results
    assert aw._aggregate_times is common._aggregate_times


def test_siblings_share_the_same_result_objects():
    for module in (mw, mg):
        assert module.TaskResult is common.TaskResult
        assert module._write_results is common._write_results


def test_harness_runner_uses_common_not_a_benchmark():
    runner_path = Path(aw.__file__).resolve().parents[1] / "harness" / "runner.py"
    modules = _imported_modules(runner_path)
    assert "mobilerun.benchmark.common" in modules
    for name in RUNNERS:
        assert not _imports(modules, name), f"harness/runner.py imports {name}.py"


def test_print_summary_requires_an_explicit_title():
    parameter = inspect.signature(common._print_summary).parameters["title"]
    assert parameter.default is inspect.Parameter.empty
