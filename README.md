# adaptive-agent-orchestrator

Control plane for Hermes-backed multi-agent workflows.

```
用户任务
   ↓
Task Contract (Pydantic)     ← 目标 / 约束 / 允许路径 / 成功标准
   ↓
Task Profiler                ← token 估算 / 子任务数 / 模块数
   ↓
Rule Router ── policies/default.yaml
   ├─ single                 ← 单 Agent
   └─ delegation
      ↓
   Explicit Subtask Plan     ← 独立 goal / allowed_paths / dependencies / expected_output
      ├─ 并行 wave
      └─ DAG dependency wave
   ↓
Budget Gate + Approval Gate  ← 硬限制 / 高风险人工确认
   ↓
Prompt Guard                 ← 把约束编码进 Agent prompt
   ↓
Workspace Manager            ← Git Worktree 隔离（写代码任务）
   ↓
Hermes Runtime Adapter       ← WebSocket TUI Gateway（experimental，需 Phase 0 实机验证）
   ↓
Deterministic Eval Gate      ← paths / budget / tests / secrets / lint
   ↓
SQLite（状态 + Telemetry）   ← append-only 审计日志
```

---

## Quick Start

### 1. 安装

```bash
git clone <repo>
cd adaptive-agent-orchestrator
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
```

### 2. 用 Mock Adapter 跑第一个任务（无需 Hermes 进程）

```bash
aao run "Fix the null pointer in login handler" \
    --mock \
    --type code_fix \
    --complexity 2 \
    --allow "src/auth/**" \
    --allow "tests/auth/**"
```

输出示例：
```
✅  Outcome: completed   Route: single   Retries: 0
   Tokens: in=500 out=200 total=700
   Eval: pass
```

### 3. 查看统计

```bash
aao stats
```

### 显式子任务（真正拆分，而不是复制父 goal）

```python
from contracts.task import SubtaskSpec, TaskContract

task = TaskContract(
    goal="Audit auth and optimize database queries",
    subtasks=[
        SubtaskSpec(id="auth", goal="Audit authentication security", allowed_paths=["src/auth/**"]),
        SubtaskSpec(
            id="db",
            goal="Profile and optimize database queries",
            allowed_paths=["src/db/**"],
            dependencies=["auth"],
            expected_output="query profile plus verified patch",
        ),
    ],
)
```

没有显式 `subtasks` 时，旧关键词 profiler 仍可触发 delegation，但会明确记录
`decomposition_mode=replicated_goal`；它是兼容性降级，不再宣称为真实任务拆分。

### 4. 路由回测（CI 里也会自动跑）

```bash
aao eval-routing
# Results: 15/15 passed  Policy: routing-v1.0
```

### 5. 连接真实 Hermes（experimental / Phase 0 spike）

先在本地启动 Hermes TUI Gateway，然后：

```bash
aao spike --url ws://localhost:4999
# 输出兼容性报告到 spike/phase0_report.yaml
```

只有 Phase 0 在目标 Hermes 版本上通过后，才应把 `--mock` 去掉：

```bash
aao run "Refactor auth module" \
    --type multi_file_refactor \
    --complexity 4 \
    --allow "src/auth/**" \
    --hermes ws://localhost:4999
```

---

## 架构分层

| 层 | 职责 | 关键文件 |
|---|---|---|
| 控制平面 | 路由、预算、状态机 | `orchestrator/` |
| 合约层 | 任务结构验证 | `contracts/` |
| 执行层 | Hermes Adapter / Mock | `adapters/` |
| 评估层 | 确定性 Eval Gate | `evals/gate.py` |
| 存储层 | SQLite（运行状态 + 审计） | `storage/` |
| 遥测层 | append-only 事件日志 | `telemetry/` |

## 目录结构

```
adaptive-agent-orchestrator/
├── aao_cli/main.py          CLI: run / spike / history / eval-routing / stats
├── aao_entry.py             console_script 入口
├── adapters/
│   ├── hermes/gateway.py    HermesAdapter (WebSocket + event stream + reconnect)
│   ├── mock.py              MockHermesAdapter (无需 Hermes 进程，用于测试)
│   └── runtime.py           AgentRuntime Protocol
├── contracts/
│   ├── task.py              TaskContract / TaskType / RiskLevel / WorkspaceSpec
│   ├── result.py            RunHandle / RunStatus / AgentResult / Usage / AgentEvent
│   └── evaluation.py        EvalResult / EvalCheck / EvalStatus
├── orchestrator/
│   ├── engine.py            完整执行流（profile → route → budget → workspace → eval）
│   ├── state_machine.py     TaskStatus 状态机 + SQLite 持久化
│   ├── router.py            Rule Router（single / delegation，记录 policy_version）
│   ├── profiler.py          TaskProfiler（token 估算 / subtask 数 / 模块数）
│   ├── budget.py            BudgetGate + ApprovalGate（CLI yes/no）
│   ├── workspace.py         WorkspaceManager（git worktree 四状态生命周期）
│   └── prompt_guard.py      PromptGuard（约束注入 + delegation splitting）
├── evals/gate.py            DeterministicEvalGate（paths/budget/tests/secrets/lint）
├── storage/
│   ├── database.py          aiosqlite 异步层
│   └── migrations/v1.py     operational + append-only audit 两组表
├── telemetry/events.py      TelemetryRecorder（append-only）
├── policies/default.yaml    路由规则（含 policy_version，可证伪）
├── datasets/
│   ├── routing_cases.yaml   路由回测用例（15 cases，aao eval-routing 使用）
│   └── failure_cases.yaml   失败模式记录（6 条，含根因和修复建议）
└── spike/phase0_gateway.py  Phase 0：Hermes Gateway 全生命周期验证脚本
```

## CLI 命令

```bash
aao run GOAL [OPTIONS]        # 提交任务
aao spike                     # Phase 0 Hermes Gateway 兼容性检查
aao history                   # 查看路由决策历史
aao eval-routing              # 路由策略回测（CI gate）
aao stats                     # 任务统计 / token 用量 / eval 通过率
```

## 运行测试

```bash
pytest tests/                 # 120 tests（含真实 Git worktree 交付/回滚）
ruff check . --ignore E501,B008  # lint（应 0 errors）
```

---

## V1 / V2 / V3 路线图

| 功能 | V1 ✅ | V2 | V3 |
|---|---|---|---|
| Task Contract + Pydantic | ✅ | | |
| Hermes Gateway Adapter | experimental / Phase 0 | ✅（实机兼容门禁后） | |
| Single Agent | ✅ | | |
| Explicit Subtask + DAG Delegation | ✅ | | |
| Rule Router + policy_version | ✅ | | |
| Budget Gate | ✅ | | |
| SQLite 状态持久化 | ✅ | | |
| Telemetry（append-only） | ✅ | | |
| Deterministic Eval Gate | ✅ | | |
| Prompt Guard | ✅ | | |
| Git Worktree 隔离 | ✅ | | |
| 简单 CLI 人工确认 | ✅ | | |
| Model Council 集成 | | ✅ | |
| LLM Judge | | ✅ | |
| 轨迹 Eval | | ✅ | |
| 角色模型路由 | | ✅ | |
| LangGraph 状态机 | | 视需要 | |
| 学习型 Router | | | ✅ |
| 自适应预算 | | | ✅ |

---

## 核心设计原则

> **先让单 Agent / 多 Agent 路由变得可测量、可追踪、可恢复，再加入更聪明的判断。**

```
控制平面 = 这个项目
执行平面 = Hermes
决策模块 = Model Council（V2 接入）
验证模块 = Eval Gate
```
