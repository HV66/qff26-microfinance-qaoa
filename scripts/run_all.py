"""Reproduce every result from scratch with one command.

Run from the repo root:  python scripts/run_all.py   (or python -m scripts.run_all)

Steps run in order via subprocess. Mandatory steps (tests, lambda sweep, QAOA
runs) stop the pipeline with a nonzero exit on failure. Optional steps whose
script does not exist yet are reported as SKIPPED (missing).

Real hardware submission (hardware_run.py) is NOT part of this pipeline because
it needs an IBM Quantum token. Run it manually, see the note printed at the end.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    """One pipeline step.

    Attributes:
        name: Banner text.
        args: Arguments after the Python interpreter.
        script: Repo-relative script that must exist for the step to run, or
            None when the step runs a module/tool that is always present.
        mandatory: A failure of a mandatory step aborts the pipeline.
    """

    name: str
    args: tuple[str, ...]
    script: str | None
    mandatory: bool


def build_steps(skip_tests: bool = False) -> list[Step]:
    """Return the ordered list of pipeline steps."""
    steps: list[Step] = []
    if not skip_tests:
        steps.append(Step("pytest", ("-m", "pytest", "-q"), None, True))
    steps += [
        Step("lambda sweep", ("-m", "scripts.lambda_sweep"), None, True),
        Step(
            "QAOA seed 0, p=1..4",
            ("qaoa_run.py", "--seeds", "0", "--ps", "1", "2", "3", "4"),
            "qaoa_run.py",
            True,
        ),
        Step(
            "QAOA seeds 1-4, p=1..3",
            ("qaoa_run.py", "--seeds", "1", "2", "3", "4", "--ps", "1", "2", "3"),
            "qaoa_run.py",
            True,
        ),
        Step("noise simulation", ("noise_run.py",), "noise_run.py", False),
        Step("classical comparison", ("compare.py",), "compare.py", False),
        Step(
            "results.md",
            ("scripts/make_results_md.py",),
            "scripts/make_results_md.py",
            False,
        ),
        Step(
            "independent verification",
            ("tests/independent/verify_independent.py",),
            "tests/independent/verify_independent.py",
            False,
        ),
        Step("demo (--quick)", ("demo.py", "--quick"), "demo.py", False),
    ]
    return steps


def is_runnable(step: Step, root: Path = ROOT) -> bool:
    """Return True if the step's script exists (or it needs no script)."""
    return step.script is None or (root / step.script).is_file()


def run_step(step: Step, root: Path = ROOT) -> int:
    """Run one step with the current interpreter; return its exit code."""
    cmd = [sys.executable, *step.args]
    return subprocess.run(cmd, cwd=root, check=False).returncode


def run_pipeline(steps: list[Step], root: Path = ROOT) -> int:
    """Run steps in order; return 0 on success or the first mandatory failure code."""
    failed_optional: list[str] = []
    for i, step in enumerate(steps, 1):
        banner = f"=== [{i}/{len(steps)}] {step.name} ==="
        print(f"\n{banner}", flush=True)
        if not is_runnable(step, root):
            print(f"SKIPPED (missing): {step.script}", flush=True)
            continue
        t0 = time.perf_counter()
        code = run_step(step, root)
        elapsed = time.perf_counter() - t0
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"--- {step.name}: {status} in {elapsed:.1f} s", flush=True)
        if code != 0:
            if step.mandatory:
                return code
            failed_optional.append(step.name)
    if failed_optional:
        print(f"\nOptional steps failed: {', '.join(failed_optional)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest step")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    code = run_pipeline(build_steps(skip_tests=args.skip_tests))
    print(f"\nTotal: {time.perf_counter() - t0:.1f} s, exit code {code}")
    print(
        "Note: real hardware is not run here (needs an IBM Quantum token). "
        "Set the token in your environment, then run: python hardware_run.py"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
