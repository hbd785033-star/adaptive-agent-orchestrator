from __future__ import annotations

from contracts.task import RiskLevel, SubtaskSpec, TaskContract, TaskType
from orchestrator.task_intake import (
    derive_task_profile,
    derive_task_requirements,
    intake_task,
)


def test_reasoning_and_execution_complexity_are_independent() -> None:
    task = TaskContract(goal="structured-only", complexity=5)
    profile = derive_task_profile(task)

    assert profile.reasoning_complexity == "high"
    assert profile.execution_complexity == "low"


def test_context_and_goal_are_not_hidden_routing_channels() -> None:
    task = TaskContract(
        goal="please use kanban and browse the web and write files",
        context={"runtime": "hermes", "mode": "kanban", "web": True},
        complexity=1,
    )
    intake = intake_task(task)

    assert intake.profile.persistent_execution is False
    assert intake.profile.long_running is False
    assert intake.profile.parallelizable is False
    assert intake.requirements.web is False
    assert intake.requirements.filesystem_write is False


def test_two_independent_ready_nodes_are_parallelizable() -> None:
    task = TaskContract(
        goal="structured-only",
        subtasks=[
            SubtaskSpec(id="a", goal="A"),
            SubtaskSpec(id="b", goal="B"),
        ],
    )
    profile = derive_task_profile(task)

    assert profile.parallelizable is True
    assert profile.cross_role_dependencies is False


def test_two_nodes_in_a_chain_are_not_parallelizable_at_initial_frontier() -> None:
    task = TaskContract(
        goal="structured-only",
        subtasks=[
            SubtaskSpec(id="a", goal="A"),
            SubtaskSpec(id="b", goal="B", dependencies=["a"]),
        ],
    )
    profile = derive_task_profile(task)

    assert profile.parallelizable is False
    assert profile.cross_role_dependencies is True


def test_dependency_dag_with_two_initial_ready_nodes_is_parallelizable() -> None:
    task = TaskContract(
        goal="structured-only",
        subtasks=[
            SubtaskSpec(id="a", goal="A"),
            SubtaskSpec(id="b", goal="B"),
            SubtaskSpec(id="c", goal="C", dependencies=["a", "b"]),
        ],
    )
    profile = derive_task_profile(task)

    assert profile.parallelizable is True
    assert profile.cross_role_dependencies is True
    assert profile.execution_complexity == "high"


def test_code_fix_requires_vendor_neutral_code_capabilities() -> None:
    task = TaskContract(goal="structured-only", task_type=TaskType.CODE_FIX)
    requirements = derive_task_requirements(task)

    assert requirements.filesystem_read is True
    assert requirements.filesystem_write is True
    assert requirements.shell is True
    assert requirements.tests is True
    assert requirements.web is False


def test_readme_code_review_requires_only_filesystem_read() -> None:
    task = TaskContract(
        task_type=TaskType.CODE_REVIEW,
        goal=(
            "Read README.md from the permitted workspace and return exactly the first "
            "non-empty line after the Markdown H1 heading, verbatim, with no quotation "
            "marks or explanation."
        ),
        allowed_paths=["README.md"],
        output_schema=[],
        risk=RiskLevel.LOW,
        complexity=1,
        subtasks=[],
    )

    requirements = derive_task_requirements(task)

    assert requirements.model_dump() == {
        "filesystem_read": True,
        "filesystem_write": False,
        "shell": False,
        "tests": False,
        "web": False,
        "background_execution": False,
        "persistent_tasks": False,
        "human_in_loop": False,
    }


def test_parallel_research_requires_web_without_write() -> None:
    task = TaskContract(goal="structured-only", task_type=TaskType.PARALLEL_RESEARCH)
    requirements = derive_task_requirements(task)

    assert requirements.web is True
    assert requirements.filesystem_write is False
    assert requirements.shell is False
    assert requirements.tests is False
