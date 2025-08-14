# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from io import TextIOBase
from pathlib import Path
from subprocess import CompletedProcess

from tqdm.auto import tqdm

import evaluation as evaluation_imp
from evaluation.code_problem import CodeProblem

# Number of seconds given to warmup a solution.
WARMUP_TIMEOUT = 1

# Names of the test suites to be tested.
SUITE_NAMES = [
    "correctness_tests",
    "consistency_tests",
    "consistency_without_correctness_tests",
    "efficiency_tests",
]


def get_repo_root() -> Path:
    """
    @brief Returns the absolute path to the repository root directory.
    @return The absoluate path to the root directory.
    """
    return Path(__file__).resolve().parents[2]


def execute_harness(
    code_dir: str,
    read_only_stdin: TextIOBase,
    timeout: float,
    use_sandbox: bool,
) -> CompletedProcess[str]:
    """
    @brief Executes the (prepared) harness at the given code directory.
    @param code_dir The directory in which the prepared solution resides.
    @param read_only_stdin Handle to the stdin file.
    @param timeout The timeout for the harness, on the given input, in seconds.
    @param use_sandbox Whether to run the code inside a sandbox.
    @return The CompletedProcess.
    """

    if not use_sandbox:
        harness_path = Path(code_dir) / "evaluation" / "harness.py"
        return subprocess.run(
            [sys.executable, str(harness_path)],
            stdin=read_only_stdin.fileno(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            env={"PYTHONPATH": code_dir},
        )
    else:
        # Executing the harness using nsjail.
        command = (
            [
                "nsjail",
                "-Mo",
            ]
            # Limit maximal written file size to 2GB.
            + [
                "--rlimit_fsize",
                "2048",
            ]
            # Prevent nsjail from setting niceness to a low value!
            + [
                "--nice_level",
                "0",
            ]
            # Prevent HPC libraries from utilizing more than one core.
            # This is needed since we are limiting the addressable memory, and if we do
            # not configure the libraries' concurrency, then they spawn multiple threads
            # that consume the entire VAS.
            + [
                "--env",
                "MKL_NUM_THREADS=1",
                "--env",
                "OPENBLAS_NUM_THREADS=1",
                "--env",
                "NUMEXPR_NUM_THREADS=1",
                "--env",
                "OMP_NUM_THREADS=1",
            ]
            # Allowing read access to rootfs.
            + ["--chroot", "/"]
            # Running as 'nobody'.
            + ["--user", "nobody"]
            # Adding the PYTHONPATH to the env
            + ["--env", f"PYTHONPATH={code_dir}"]
            # Enforce the time limit.
            + ["--time_limit", f"{timeout}"]
            # Setting the CWD.
            + ["--cwd", code_dir]
            + ["--", sys.executable, "-m", "evaluation.harness"]
        )
        return subprocess.run(
            command,
            stdin=read_only_stdin.fileno(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )


def prepare_solution(
    candidate_solution_path: str,
    configuration: dict[str, int | bool | str],
    use_sandbox: bool,
    warmup: bool = True,
) -> str:
    """
    @brief Prepares the given candidate solution for execution, by generating
           a temporary directory in which the solution's execution environment
           will be placed.
    @param candidate_solution_path The path of the candidate solution shim.
    @param configuration The global configuration to be used when evaluating the solution.
    @param use_sandbox Whether the code should be executed in a sandbox, in warmup is enabled.
    @param warmup Whether to 'warmup' the solution by executing it once.
    @return The path of the directory in which the prepared solution is placed.
    """
    with tempfile.TemporaryDirectory(delete=False) as temp_dir:

        # Copy 'evaluation/' module to temp directory.
        temp_code_dir = Path(temp_dir)
        evaluation_src = os.path.dirname(evaluation_imp.__file__)
        evaluation_dest = temp_code_dir / "evaluation"
        shutil.copytree(evaluation_src, evaluation_dest)

        # Override 'shim.py' with candidate solution.
        shim_path = evaluation_dest / "shim.py"
        shutil.copy2(candidate_solution_path, shim_path)

        # Overwrite the configuration file.
        with open(os.path.join(evaluation_dest, "config.py"), "w") as f:
            f.write(f"GLOBAL_CONFIG={repr(configuration)}")

        if warmup:

            stdin_path = os.path.join(evaluation_dest, "warmup_input.txt")
            with open(stdin_path, "w") as f:
                f.write("WARMUP_INPUT")

            try:
                with open(stdin_path, "r") as read_only_stdin:
                    execute_harness(
                        code_dir=str(temp_code_dir),
                        read_only_stdin=read_only_stdin,
                        timeout=WARMUP_TIMEOUT,
                        use_sandbox=use_sandbox,
                    )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass  # The warmup is given an invalid input, so exception may be raised.

        return temp_dir


def run_test(
    solution_directory: str,
    stdin_string: str,
    timeout: float,
    use_sandbox: bool,
) -> tuple[str, float]:
    """
    @brief Executes the given candidate solution on the given stdin string.
    @param solution_directory Path to directory containing prepared solution.
    @param stdin_string The string to be provided as standard input to the executed code.
    @param timeout The maximum allotted running time, in seconds.
    @param use_sandbox Whether to run the code in a sandbox or not.
    @return (stdout, execution_time).
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=True) as temp_stdin_file:

        # Write the stdin string
        stdin_path = temp_stdin_file.name
        temp_stdin_file.write(stdin_string)
        temp_stdin_file.flush()

        with open(stdin_path, "r", encoding="utf-8") as read_only_stdin_handle:
            try:
                execution_start_time = time.perf_counter()
                res = execute_harness(
                    code_dir=solution_directory,
                    read_only_stdin=read_only_stdin_handle,
                    timeout=timeout,
                    use_sandbox=use_sandbox,
                )
                execution_time = time.perf_counter() - execution_start_time
                return res.stdout, execution_time
            except subprocess.TimeoutExpired as e:
                raise TimeoutError(f"Script timed out after {timeout} seconds.") from e
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Script from directory {solution_directory}\n"
                    + f"Failed with exit code {e.returncode}.\n"
                    + f"Input:\n{stdin_string}\n-----\n"
                    + f"Output:\n{e.output}\n-----\n"
                    + f"Error:\n{e.stderr}"
                ) from e


def test_solution(
    candidate_solution_path: str,
    code_problem: CodeProblem,
    use_sandbox: bool,
) -> dict[str, tuple[float, ...]]:
    """
    @brief Runs the candidate solution against the tests in the given code problem.

    @param candidate_solution_path Path to the candidate Python script with callback and result extraction functions.
    @param code_problem The problem instance to test against.
    @param run_script_args Optional dictionary of arguments for the run script.
    @param use_sandbox Whether to run the code inside a sandbox.
    @raises AssertionError If the output of the candidate solution does not match the expected output for any of the
                           correctness tests.
    @raises RuntimeError If the candidate solution fails to execute properly due to an error or exception.
    @raises TimeoutError If the execution of the candidate solution exceeds the allowed time.
    @return execution_times_per_test A dictionary mapping each test suite name (e.g., "correctness_tests") to a tuple
                                     of execution times (in seconds) for each test case. Only returned if all tests
                                     pass. Although metrics are currently binary (pass/fail), this timing information
                                     can serve as a useful supervision signal for further research.
    """

    time_limit_seconds = code_problem.time_limit_in_seconds
    prepared_solution_dir = prepare_solution(
        candidate_solution_path=candidate_solution_path,
        configuration=code_problem.configuration,
        use_sandbox=use_sandbox,
    )
    execution_times_per_test: dict[str, tuple[float, ...]] = {}
    try:

        for suite_name in SUITE_NAMES:
            assert hasattr(code_problem, suite_name), f"Missing test suite: {suite_name}"

            suite_outputs = []
            suite_execution_times = []
            test_suite = getattr(code_problem, suite_name)
            for test in tqdm(test_suite, desc=f"Testing {suite_name}", total=len(test_suite)):
                actual_output, execution_time = run_test(
                    solution_directory=prepared_solution_dir,
                    stdin_string=test.input,
                    timeout=time_limit_seconds,
                    use_sandbox=use_sandbox,
                )
                actual_output = actual_output.strip()
                if test.expected_output is not None:
                    if actual_output != test.expected_output.strip():
                        raise AssertionError(
                            f"Got:\nOutput:\n{actual_output}\n"
                            + f"Excepted:\n{test.expected_output}\n"
                            + f"Input:\n{test.input[:1000]}"
                        )

                suite_outputs.append(actual_output)
                suite_execution_times.append(execution_time)

            if suite_name == "consistency_without_correctness_tests":
                assert len(suite_outputs) > 0
                if not all(result == suite_outputs[0] for result in suite_outputs):
                    raise AssertionError(
                        f"Failed consistency without correctness! Got the following outputs: {suite_outputs}."
                    )

        execution_times_per_test[suite_name] = tuple(suite_execution_times)

    finally:
        shutil.rmtree(prepared_solution_dir)  # Cleanup.

    return execution_times_per_test


if __name__ == "__main__":

    # Run tests for the example warmup problem.
    import example.wmc_dominating_set as wmc_dominating_set_imp

    root = get_repo_root()
    json_str = (root / "dataset" / "public" / "formulaone_warmup" / "142.json").read_text(encoding="utf-8")
    code_problem = CodeProblem.from_dict(json.loads(json_str))

    solution_path = Path(wmc_dominating_set_imp.__file__)
    test_solution(
        candidate_solution_path=str(solution_path),
        code_problem=code_problem,
        use_sandbox=False,
    )
