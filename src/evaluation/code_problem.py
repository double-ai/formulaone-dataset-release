# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass


@dataclass(frozen=True, repr=True, eq=True)
class CodeProblemTest:
    input: str
    expected_output: str | None


@dataclass(frozen=True, repr=True, eq=True)
class CodeProblem:

    # Metadata.
    id: str
    problem_description: str
    time_limit_in_seconds: int
    configuration: dict[str, int | bool | str]
    tags: list[str]

    # Test suites.
    correctness_tests: tuple[CodeProblemTest, ...]
    consistency_tests: tuple[CodeProblemTest, ...]
    consistency_without_correctness_tests: tuple[CodeProblemTest, ...]
    efficiency_tests: tuple[CodeProblemTest, ...]

    @staticmethod
    def from_dict(d: dict) -> "CodeProblem":
        return CodeProblem(
            id=d["id"],
            problem_description=d["problem_description"],
            time_limit_in_seconds=d["time_limit_in_seconds"],
            configuration=d["configuration"],
            tags=d["tags"],
            correctness_tests=tuple(CodeProblemTest(**t) for t in d["correctness_tests"]),
            consistency_tests=tuple(CodeProblemTest(**t) for t in d["consistency_tests"]),
            consistency_without_correctness_tests=tuple(
                CodeProblemTest(**t) for t in d["consistency_without_correctness_tests"]
            ),
            efficiency_tests=tuple(CodeProblemTest(**t) for t in d["efficiency_tests"]),
        )

    @staticmethod
    def read_code_problem(path: str) -> "CodeProblem":
        with open(path, "r") as f:
            return CodeProblem.from_dict(json.loads(f.read()))
