"""Runtime capability requirements derived from a task."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TaskRequirements(BaseModel):
    """
    Runtime capabilities required to complete a task.

    Requirements are vendor-neutral and derived from TaskContract plus profiling;
    risk remains owned by TaskContract.
    """

    model_config = ConfigDict(extra="forbid")

    filesystem_read: bool = False
    filesystem_write: bool = False

    shell: bool = False
    tests: bool = False
    web: bool = False

    background_execution: bool = False

    persistent_tasks: bool = False
    human_in_loop: bool = False
