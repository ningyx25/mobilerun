# CLAUDE.md — `mobilerun/agent/`

本文件描述 `mobilerun/agent/` 包的**设计理念与内部结构**，供人和编码 agent 快速建立心智模型。
具体实现以代码为准；行号随重构会漂移，引用时请以函数/类名为锚点。

---

## 1. 一句话定位

`mobilerun/agent/` 是**基于 llama-index `Workflow`（事件驱动状态机）编排的 Agent 运行时**，负责把这条闭环跑起来：

```
目标 → 设备状态(截图/a11y) → LLM → 工具调用 → 设备动作 → 证据
```

它**不是**一个 Agent，而是一套把"设备控制 Agent"拆成可替换构件的骨架。

**边界**：设备 I/O 不在这里 —— 来自 `mobilerun-core-local` 的 driver（`AndroidDriver` / `IOSDriver` / `AndroidWorldHttpDriver` / `MobileWorldHttpDriver` / `VisualRemoteDriver`）以及设备上的 Portal app。LLM 访问走 `llama-index`。本包拥有的是**编排、prompt、工具表、状态、证据留痕**。

---

## 2. 架构总览

```
                 ┌──────────────────────────────────────────────┐
   instruction ─▶│  MobileAgent (droid/droid_agent.py)           │
                 │  编排层：事件驱动状态机，只做调度               │
                 │  start → execute_task → handle_* → finalize  │
                 └───────┬──────────────────────┬───────────────┘
                         │ reasoning=false      │ reasoning=true
                         │ (:589/:606/:635)     │
                         ▼                      ▼
                   ┌───────────┐        ┌──────────┐   ┌──────────┐
                   │ FastAgent │        │ Manager  │──▶│ Executor │
                   │ (XML 工具) │        │ (规划)    │   │ (动作)    │
                   └─────┬─────┘        └────┬─────┘   └────┬─────┘
                         │                   │              │
                         └───────────┬───────┴──────────────┘
                                     ▼
                        ┌────────────────────────┐
                        │ ToolRegistry (工具真相)  │◀── deps 能力裁剪
                        │  → XML / text / types   │◀── MCP / 用户工具
                        └───────────┬────────────┘
                                    ▼
                    ActionContext(driver / ui / shared_state / creds)
                                    ▼
                          设备动作 → ActionResult
                                    │
        ┌───────────────────────────┴──────────────────────────┐
        ▼                        ▼                ▼            ▼
   trajectory 落盘         CLI 实时输出     tracing/telemetry  usage 计量
              （全部是事件总线上的旁路消费者，不写进业务逻辑）
```

---

## 3. 五条核心设计理念

### 3.1 编排与执行分离

`droid/droid_agent.py` 的 `MobileAgent`（约 1438 行，最重的文件）里**几乎全是 `@step`**：

| step | 作用 |
|---|---|
| `start_handler` `:531` | 初始化 deadline/轨迹，按模式分支，或走 external 早退 |
| `execute_task` `:1001` | 跑一个 `FastAgent` episode |
| `handle_fast_agent_result` `:1064` | 收尾 → `FinalizeEvent` |
| `run_manager` `:1085` / `handle_manager_plan` `:1138` | 规划阶段 |
| `run_executor` `:1159` / `handle_executor_result` `:1188` | 动作阶段 |
| `finalize` `:1217` | 汇总结果 |

它自己**不构造 prompt、不解析 LLM 输出**——那些全在子 workflow 里。好处：换认知模式只改 `start_handler` 的分支，编排逻辑一行不动。

### 3.2 三种认知模式，同一套外围

| 模式 | 组成 | 每步 LLM 调用 | 适用 |
|---|---|---|---|
| `reasoning: false` | 单个 `FastAgent`（`fast_agent/fast_agent.py:82`） | 1 次 | 快、便宜 |
| `reasoning: true` + `executor.kind: "llm"` | `Manager`（`manager/manager_agent.py`）+ `Executor`（`executor/executor_agent.py`） | 2 次 | 准，规划与执行分家 |
| `reasoning: true` + `executor.kind: "grounded"` | `Manager`（LLM 不变）+ `GroundedExecutorAgent`（`executor/grounded.py`）——执行由本地决策模型 + oneflow 完成 | Manager 1 次 + clef/gui-actor 各若干次 | 用本地 `clef-flash`/`gui-actor` 取代 LLM 工具循环 |

关键：**三者外围完全一致** —— 同样的 driver、`ToolRegistry`、`MobileAgentState`、事件流、轨迹留痕。所以这是个"热插拔开关"，不是几套系统。`executor.kind`（`ExecutorConfig`）在 `droid_agent.py` 的 `__init__` 里选执行器；**`grounded` 目前只支持 `vision_only: true`**（非该组合直接在 `__init__` 报错）。

`ManagerAgent` 有变体 `StatelessManagerAgent`（`manager/stateless_manager_agent.py:1`）—— 每轮重建上下文，用于抑制长任务记忆膨胀。

#### grounded 执行器（`executor/grounded.py`）

每步与 LLM Executor 同契约（`run(subgoal=…)` → `{action, outcome, error, summary, thought}`），但内部是：`prepare`（抓当前原图，**不读 a11y 树**）→ `decide`（`POST /v1/systemone` 打 `clef-flash` 出 `operation` + `app_target`）→ `ground`（TAP/LONG_PRESS 由 `gui-actor` grounding；SWIPE 由 `oneflows/swipe_agent` 出归一化起止点；TYPE_TEXT 由 `oneflows/text_input_agent` 出精确文本）→ `execute`（直接调 `utils/actions.py` 的动作函数，绕过 ToolRegistry）。请求构造在 `executor/jev_prompt.py`；传输在 `utils/systemone.py`。两个决策模型的请求都只带**一张当步原始截图**（未缩放、无标注、无历史帧），历史经 `state.recentActions` 文本传递。任一失败都降级为该步反馈，不炸 episode。

### 3.3 事件驱动 + 单一事件总线

所有子 workflow 的事件都由 `handle_stream_event`（`droid_agent.py:1423`）汇聚到一个出口：

```python
if not isinstance(ev, StopEvent):
    ctx.write_event_to_stream(ev)          # 对外广播
    if self.trajectory:                    # 顺手落盘
        ScreenshotEvent    → screenshot_queue
        RecordUIStateEvent → ui_states
        other              → events
```

`common/events.py` 只定义 **3 个跨层事件**（`ScreenshotEvent` / `RecordUIStateEvent` / `ToolExecutionEvent`）；每个子 agent 各自一套私有 `events.py`（`droid/` `executor/` `manager/` `fast_agent/` 各有一份）。

=> **局部事件类型 + 全局汇聚点**：子模块可自由定义私有事件，只有需要跨层的才升级到 `common/`。

CLI 实时输出、trajectory 落盘、Langfuse/Phoenix tracing、telemetry 全是这条总线上的**旁路消费者**。

### 3.4 工具即数据 —— 单一事实源，多种渲染

`ToolRegistry`（`tool_registry.py:33`）是全框架唯一的工具真相来源。它**存结构、不存字符串**（`params` / `description` / `deps`），按消费者渲染：

| 方法 | 输出 | 给谁 |
|---|---|---|
| `get_tool_descriptions_xml` `:195` | `<functions><function>{json}</function>` | FastAgent 的 XML 协议 |
| `get_tool_descriptions_text` `:206` | `- name(a: int, b: str): desc` | Executor 的文本 prompt |
| `get_param_types` `:217` | `{param: type}` | XML 解析时的类型强转 |

工具还带 **`deps` 能力声明**（`utils/signatures.py:148` 起，如 `deps={"tap","element_index"}`），启动时用 `disable_unsupported(capabilities)`（`tool_registry.py:72`）按后端能力自动裁剪。这就是"一套工具表，Android / iOS / http 后端自动适配"的机制。

### 3.5 依赖注入 + 统一返回契约

- **`ActionContext`**（`action_context.py:21`）替代了早先的 `tools=tools_instance`。动作函数只声明 `ctx=`，拿到 driver / ui / shared_state / credential_manager，不依赖全局单例。
- **`ActionResult(success, summary)`**（`action_result.py`）是统一返回。
- **`ToolRegistry.execute`**（`tool_registry.py:105`）负责归一化各种返回值（`ActionResult` / `tuple` / `str` / 其它），并把 `Unknown tool`、`TypeError`（参数错）、任意 `Exception` 一律转成 `success=False` 的**结果**而非抛出。

=> 原则：**一次工具调用失败是"给模型看的一条反馈"，不应该炸掉整个 episode。**

---

## 4. 目录地图

```
agent/
├── droid/          编排层：MobileAgent workflow + 共享状态 MobileAgentState(state.py:18)
├── fast_agent/     直连模式：XML 工具调用协议（fast_agent.py + xml_parser.py）
├── manager/        分层模式上半：规划、子目标、终止判断（prompts.py: 响应解析与校验）
├── executor/       分层模式下半：把子目标翻译成具体动作
├── oneflows/       单用途小 workflow：AppStarter（描述→包名开 app）、StructuredOutputAgent、
│                   SwipeAgent / TextInputAgent（grounded 执行器的 SWIPE/TYPE_TEXT 参数）、common(共享 LLM/JSON 辅助)
├── external/       逃生口：自带完整实现的第三方 agent，零 import mobilerun
├── providers/      各 LLM 后端适配（Anthropic/Grok/MiniMax + OAuth + registry）
├── trajectory/     落盘：轨迹与证据包写出
├── common/         跨层共享：3 个事件 + LLM_HISTORY_LIMIT(=100)
├── utils/          横切关注点：actions(工具实现) / signatures(工具表) /
│                   llm_loader·llm_picker / vision_sizing / prompt_resolver /
│                   inference(重试) / tracing_setup / errors / oauth/
├── action_context.py   依赖注入容器
├── action_result.py    工具统一返回
├── tool_registry.py    工具单一事实源
├── usage.py            token 计量 + provider 别名表
└── droid/state.py      跨子 workflow 共享状态（pydantic）
```

---

## 5. 运行时主链路

### 5.1 `reasoning: false`（FastAgent 直连）

```
StartEvent → start_handler(:531) → FastAgentExecuteEvent
           → execute_task(:1001) → FastAgent.run()
               prepare_chat(:200) → handle_llm_input(:231) → handle_llm_output(:436)
               → execute_code(:523) → handle_execution_result(:607) → finalize(:640)
           → handle_fast_agent_result(:1064) → FinalizeEvent → finalize(:1217)
```

FastAgent 协议：LLM 输出 `<function_calls>` 块 → `xml_parser` 解析 → `ToolRegistry.execute` → 结果以 `<function_results>` 作为 user message 回灌。

### 5.2 `reasoning: true`（Manager + Executor）

```
start_handler → ManagerInputEvent
  → run_manager(:1085)  ── 出 plan/current_subgoal/answer
  → handle_manager_plan(:1138)
        ├─ 有 answer → FinalizeEvent
        └─ 否则 → ExecutorInputEvent → run_executor(:1159) → handle_executor_result(:1188)
                                            └─ 回到 ManagerInputEvent（循环）
```

每轮 `step_number += 1`，到 `config.agent.max_steps` 由 `run_manager` 顶部截断（`droid_agent.py:1089`）。

### 5.3 external agent 早退（`droid_agent.py:541-586`）

**在 driver 创建之前**就 return，时序上早于一切内部机制。判定只有一个布尔量：

```python
_BUILTIN_AGENT_NAMES = {"mobilerun", "droidrun"}      # :388
self._using_external_agent = config.agent.name not in _BUILTIN_AGENT_NAMES   # :389
```

被跳过的：LLM 加载（`:406`）、Manager/Executor 实例化（`:455-457`）、ToolRegistry / StateProvider / MCP / 事件流 / trajectory / usage / tracing。

=> **名称即路由**。

---

## 6. 三个扩展点及分工

| | **MCP** (`mobilerun/mcp/`) | **external agent** (`external/`) | **oneflows** (`oneflows/`) |
|---|---|---|---|
| 加什么 | 加**能力**（工具） | 换**大脑**（整个循环） | 加一个**小 workflow** |
| 依赖方向 | 插件依赖宿主工具契约 | 插件**零依赖**宿主 | 框架内新增 |
| 侵入位置 | 注册进 `ToolRegistry`，被 prompt 渲染 | 绕过 `ToolRegistry`，整段接管 | 独立 `Workflow`，被 step 调用 |
| 可观测性 | 完整（事件 / trajectory / usage） | **基本为零** | 取决于是否走事件总线 |
| 适用后端 | 全部（含 HTTP driver） | **仅 ADB** | 全部 |

一句话：**MCP 是"往框架里加手"，external 是"换一颗脑袋，把手也一起带走"。**

### external agent 契约（`external/__init__.py:11-22`）

```python
async def run(device: AdbDevice, instruction: str, config: dict, max_steps: int) -> dict
#                                                        └── {"success": bool, "reason": str, "steps": int}
```

- 发现机制：**文件即注册**。`list_agents()`（`:45`）扫目录，`load_agent()`（`:62`）用 `importlib` 动态导入。支持单文件或包（README:85-99）。无 `entry_points`、无 pip install。
- 配置：模块级 `DEFAULT_CONFIG` 垫底，用户的 `external_agents.<name>` 覆盖 → `{**defaults, **user}`（`droid_agent.py:564/571`）。
- 校验前置：缺 `run` / `run` 非协程 / `ImportError` 都在加载期返回 `None` 并给出具体报错（`external/__init__.py:74-91`）。
- 约束（README:125-131）：**zero imports from `mobilerun`**、自带 LLM 客户端、只用 `device`、不要自己 `connect()`。

---

## 7. 已知张力与陷阱（改动前必读）

这些是设计理念在落地时的**易错点**，也是最容易踩的地方。

### 7.1 异常边界必须一致 —— 别让一次读取失败炸掉整个 episode

**这是本包最重要的一条隐含原则**，也是唯一一个曾经被违反过的：

> 一次工具调用 / 一次状态读取失败，应该是"给模型看的一条反馈"或"这一步跳过"，**不应该终止整个 episode**。

原则的落点与历史教训：

| 位置 | 现状 | 备注 |
|---|---|---|
| `tool_registry.py:105` `execute` | ✅ 兜底 `Unknown tool` / `TypeError` / 任意 `Exception` → `success=False` | 设计如此 |
| `fast_agent.py:265-284` 截图抓取 | ✅ `except DeviceDisconnectedError: raise` + `except Exception: warn` | |
| `fast_agent.py:288-312` 取设备状态（内含 `screenshot_provider.py:50` 的 `image_dimensions`） | ✅ 同上 | |
| `fast_agent.py:353-367` 缩放截图 + 挂到消息上 | ✅ `except Exception: warn("Dropping undecodable screenshot for this step")` | **曾经缺失** |

第 4 行曾经**在 try 之外**：一帧损坏/截断的截图会让异常冒泡到 `droid_agent.py:1054-1061`，生成 `reason=f"Error: {error}"`，把一个"少看一帧图"升级成"整个任务判 0"。

已在 commit `f869a43`（`fix/screenshot-integrity`）修复，代码注释写明了设计意图：

```python
except Exception as e:  # noqa: BLE001
    # An undecodable frame is not worth failing the episode
    # over: skip it and run this step without vision. Do NOT
    # attach the raw bytes -- the provider would reject them.
    logger.warning(f"⚠️ Dropping undecodable screenshot for this step: {e}")
```

> **例外**：`DeviceDisconnectedError` 应当 `raise` —— 设备掉了就没必要再继续，这是唯一的合法中止理由。

**修改本条区域时的 checklist**：新增任何"取设备数据 → 解码/解析 → 塞进 prompt"的步骤，都要问一句"这里失败会不会杀掉整个 episode"。

> 事故记录：仓库根目录 `BUGREPORT-screenshot-decode-aborts-episode.md`（描述的是**修复前**的状态）。

### 7.2 XML 协议双轨 —— `add_memory` 不在 ToolRegistry 里

FastAgent 的 `<add_memory>…</add_memory>` 是**按标签正则解析**的（`xml_parser.py:182-188` `extract_add_memory`），**不注册进 `ToolRegistry`**；其它工具走 `<invoke name="...">`。提示词同时教了两种形式，模型偶尔把 `add_memory` 写成 `<invoke>` → `Unknown tool: add_memory`，白费 1-2 步。

=> "单一事实源"在协议层有个已知例外。

### 7.3 事件类型分散

每个子 agent 一套 `events.py`。好处是解耦，代价是同一件事（如"第 N 步的思考"）在不同层有不同字段名，跨层调试需在 `droid/` 与 `fast_agent/` 之间对照。

### 7.4 超时是 **workflow 级**，不是任务级

`MobileAgent(timeout=task_timeout)` → llama-index `Workflow` run timeout。卡满即整段取消：

```
WorkflowTimeoutError: Operation timed out after N seconds. Currently active steps: execute_task
```

- CLI 开关：`--task-timeout`（`cli/main.py`，默认 `1000.0`）
- 长流程 GUI 任务（16 分钟级）会因此被误杀，异常记进 `error` 字段且 `finished=False`

### 7.5 external agent 只吃 ADB，benchmark 显式拒绝

早退路径走 `adb.list()` / `adb.device(serial=...)`，**在 driver 工厂分支之前** → 拿不到任何 HTTP driver。因此 benchmark 入口主动拦截：

- `benchmark/android_world.py:868-874`
- `benchmark/mobile_world.py:406-409`

```python
# "External agents take MobileAgent's early-exit path, which drives adb
#  directly and would bypass the injected AndroidWorld driver."
```

=> external 目前的定位是**本地真机实验**，不是评测基线。

### 7.6 其他常量

- `common/constants.py`: `LLM_HISTORY_LIMIT = 100`（prompt 里保留的最近对话步数）
- `usage.py`: `SUPPORTED_PROVIDERS` / `PROVIDER_ALIASES` —— 新增 provider 时两处都要登记

---

## 8. 修改指引：想加 X 该动哪

| 目标 | 改动位置 |
|---|---|
| 加一个设备动作（tap/输入/自定义） | `utils/actions.py` 写函数（签名 `(..., ctx: ActionContext)`）→ `utils/signatures.py` 注册（含 `params`/`description`/`deps`）→ 可选补 `config/prompts/` 说明 |
| 加一个 LLM 后端 | `providers/` + `usage.py` 的 `SUPPORTED_PROVIDERS`；OAuth 类放 `utils/oauth/` |
| 改直连模式行为 | `fast_agent/fast_agent.py` + `config/prompts/fast_agent/*.jinja2` + `xml_parser.py` |
| 改分层模式行为 | `manager/prompts.py`（解析/校验）、`executor/prompts.py` |
| 改 grounded 执行器行为 | `executor/grounded.py` + `executor/jev_prompt.py`（请求构造）+ `utils/systemone.py`（传输）；SWIPE/TYPE_TEXT 的参数在 `oneflows/swipe_agent.py` / `oneflows/text_input_agent.py` |
| 加一个单用途 workflow | `oneflows/` 新建 `Workflow`，在 `start_handler` 或工具里调用 |
| 接入第三方 agent 做对比 | `external/<name>.py` 实现 `async def run(...)`，配 `external_agents.<name>` |
| 改证据/轨迹格式 | `trajectory/writer.py` + `droid_agent.py:handle_stream_event` + 上层 `mobilerun/benchmark/evidence.py` |
| 改状态字段 | `droid/state.py` 的 `MobileAgentState`（pydantic，跨子 workflow 共享）|
| 加跨层事件 | `common/events.py`；**子模块私有事件留在各自的 `events.py`** |
| 改步数/超时策略 | `run_manager` 顶部截断逻辑（`:1089`）+ CLI `--steps` / `--task-timeout` |

---

## 9. 读代码的建议顺序

1. `droid/droid_agent.py` 顶部 docstring（`:1-6`）+ `start_handler`（`:531`）—— 建立全局拓扑
2. `common/events.py` —— 理解跨层消息契约（只有 3 个）
3. `tool_registry.py` —— 理解"工具即数据"
4. `droid/state.py` `MobileAgentState` —— 理解跨子 workflow 共享了什么
5. 按需下钻 `fast_agent/` 或 `manager/` + `executor/`
6. 想理解扩展点：`external/README.md`（最全的扩展说明，132 行）

**注意**：`droid_agent.py:1438` 有 `DroidAgent = MobileAgent` 的兼容别名（deprecated，v0.8.0 移除）。看到 `DroidAgent` 一律按 `MobileAgent` 理解。

---

**设计理念一句话总结**：用「事件驱动的编排层 + 可插拔认知模式 + 注册表式工具 + 依赖注入 + 旁路可观测」把设备控制 Agent 做成**可组合、可替换、可观测**的运行时 —— FastAgent / Manager+Executor 、external agent、MCP、多后端，都只是挂在同一套骨架上的不同实现。
