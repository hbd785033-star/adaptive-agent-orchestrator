# adaptive-agent-orchestrator

Control plane for Hermes-backed multi-agent workflows.

```
用户任务
   ↓
Task Contract (Pydantic)
   ↓
Task Profiler
   ↓
Rule Router  ──  policies/default.yaml  ──  routing_decisions (append-only)
   ↓
Budget Gate + Approval Gate
   ↓
Workspace Manager  (git worktree per code-writing child)
   ↓
Hermes Runtime Adapter  (TUI Gateway WebSocket)
   single │ delegation
   ↓
Event Stream  (AgentEvent — typed)
   ↓
Deterministic Eval Gate
   paths · budget · secrets · lint · tests
   ↓
Telemetry + SQLite
   operational: tasks / runs / subtasks
   audit:       telemetry_events / eval_results / routing_decisions / usage_records
   ↓
COMPLETED  /  FAILED→RETRY→…  /  ABANDONED
```

---

## Phase 0 first: validate Hermes Gateway

Before running any real tasks, run the spike to confirm your Hermes instance
exposes the TUI Gateway and all required operations work:

```bash
python spike/phase0_gateway.py --ws ws://localhost:PORT --task "say hello"
```

The spike checks: connect → auth → session → submit → event stream →
delegation events → usage → steer → interrupt → reconnect → cleanup.

It writes a YAML compatibility report.  **Do not start Phase 1 until all
`blockers` pass.**

---

## Quick start

```bash
# create venv and install
uv venv .venv --python 3.11
uv pip install -e ".[dev]"

# run all tests (no Hermes required — uses MockHermesAdapter)
pytest tests/ -v

# submit a task via CLI
aao run \
  --goal "refactor src/auth/login.py to use the new session helper" \
  --allowed-paths "src/auth/**" \
  --task-type general
```

---

## Project layout

```
adaptive-agent-orchestrator/
│
├─ cli/            typer CLI entry point
├─ contracts/      TaskContract, AgentEvent, EvalResult — Pydantic models
├─ adapters/
│  ├─ runtime.py   AgentRuntime Protocol (swap Hermes for Codex later)
│  ├─ hermes/      HermesAdapter — WebSocket TUI Gateway
│  └─ mock.py      MockHermesAdapter — deterministic, no Hermes needed
├─ orchestrator/
│  ├─ engine.py    Main pipeline: profile→route→budget→execute→eval
│  ├─ state_machine.py  TaskStatus + SQLite persistence
│  ├─ router.py    RuleRouter — single / delegation, records policy_version
│  ├─ profiler.py  Extracts routing signals from TaskContract
│  ├─ budget.py    BudgetGate (hard limits) + ApprovalGate (CLI yes/no)
│  └─ workspace.py WorkspaceManager — git worktree lifecycle
├─ evals/
│  └─ gate.py      DeterministicEvalGate: paths, budget, secrets, lint, tests
├─ storage/        aiosqlite database — two conceptual groups (see below)
├─ telemetry/      Append-only event log + routing metrics queries
├─ policies/       default.yaml — routing rules + budget + approval triggers
├─ spike/          phase0_gateway.py — Hermes Gateway compatibility test
└─ tests/          46 unit + integration tests, all via MockHermesAdapter
```

---

## Storage model

Two groups, separate access patterns:

**Operational** (mutable, tracks live state)
- `tasks` — current TaskStatus, retry count
- `runs` — per-execution record
- `subtasks` — child agent runs

**Audit** (append-only, never modified after write)
- `telemetry_events` — structured event log
- `eval_results` — per-run eval gate output
- `routing_decisions` — route + reasons + `policy_version` column (indexed)
- `usage_records` — token + cost per run

`policy_version` is a top-level SQL column so you can join and compare:

```sql
SELECT route, policy_version, COUNT(*) tasks,
       ROUND(AVG(CASE WHEN e.overall = 'pass' THEN 1.0 ELSE 0.0 END), 3) success_rate
FROM routing_decisions r
JOIN eval_results e USING (task_id)
GROUP BY policy_version, route
ORDER BY policy_version, route;
```

---

## Routing policy

Edit `policies/default.yaml` to tune routing rules.
Increment `policy_version` on every change so `routing_decisions` can diff
success rates across versions.

```yaml
policy_version: "routing-v1.0"

routing:
  delegation:
    min_independent_subtasks: 2
    min_estimated_input_tokens: 8000
    allowed_task_types:
      - multi_file_refactor
      - parallel_research
      - test_and_implement
  single:
    max_complexity: 2
    max_affected_modules: 1
  constraints:
    sequential_dependency_forces_single: true
```

---

## Budget & approval

`policies/default.yaml` also controls hard limits and approval triggers:

```yaml
budget:
  max_children: 2
  max_depth: 1
  max_retries: 1
  max_total_calls: 8
  require_approval_above_calls: 5

approval:
  require_for_risk_levels: [high, critical]
  require_for_actions: [delete, deploy, merge_main, send_to_multiple_providers]
```

Tasks that exceed `require_approval_above_calls` or touch a forbidden action
pause and ask `Proceed? [y/N]` on the CLI before continuing.

---

## Worktree isolation

For any `delegation` task that writes code, the engine allocates one git
worktree per child agent:

```
repo/
└─ .worktrees/
   └─ <task_id>/
      ├─ child-1/   branch: agent/<task_id>/child-1
      └─ child-2/   branch: agent/<task_id>/child-2
```

Read-only / research delegation shares the main tree.

Worktree states: `ALLOCATED → ACTIVE → MERGING → CLEANED`
                                     `↘ ABANDONED`  (eval fail; kept for inspection)

---

## Adding a runtime adapter

1. Implement `adapters/runtime.py::AgentRuntime` (all async)
2. Add `events(run_id, *, after=...)` — yields `AgentEvent`, supports cursor-based reconnect
3. Wire it in `cli/main.py`

The router, eval gate, budget, and telemetry layers are adapter-agnostic.

---

## V1 / V2 / V3 roadmap

| Feature | Status |
|---|---|
| Task Contract + Pydantic | ✅ v1 |
| Hermes Gateway Adapter | ✅ v1 |
| Mock Adapter + 46 tests | ✅ v1 |
| Rule Router (single/delegation) | ✅ v1 |
| Budget Gate + Approval Gate | ✅ v1 |
| Git Worktree isolation | ✅ v1 |
| Deterministic Eval Gate | ✅ v1 |
| SQLite state + audit tables | ✅ v1 |
| Telemetry + metrics queries | ✅ v1 |
| Phase 0 Gateway spike | ✅ v1 |
| Model Council integration | v2 |
| LLM Judge | v2 |
| Trajectory Eval | v2 |
| Per-role model routing | v2 |
| LangGraph (if needed) | v2 optional |
| Learning Router | v3 |
| Adaptive budget | v3 |
