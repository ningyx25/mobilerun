#!/usr/bin/env bash
# =============================================================================
# eval_mobilegym.sh —— MobileGym 一键测评（eval.sh 的 MobileGym 专用版）
#
# 拓扑（进程 i = 模型 i，下标从 1 开始）：
#
#     环境服务      http://<host>:9001 + (i-1)        进程内托管 K 个 MobileGym 浏览器后端
#     智能体服务    http://<host>:9301 + (i-1)        agent-harness worker
#     模型服务      http://<model-host>:29100 + (i-1) sglang / OpenAI 兼容端点
#                   （默认 <model-host> = --host；可用 --model-host / --model-base-url 分离）
#
# 与 eval.sh 的区别：
#   * MobileGym 没有独立的环境 HTTP 服务；模拟器是 :4180 上的 React SPA。
#     `mobilerun serve env` 在每个 backend 里用 Playwright 启动一个 Chromium page。
#   * 一个 backend = 一个浏览器。进程 i 默认 K=1 个后端；--backends-per-env 可调。
#     总浏览器数 = processes × backends-per-env，注意内存和 Chromium 的 inotify 配额。
#   * 模拟器没有 a11y tree；后端会报告 has_ui_tree=false，runner 自动 vision_only=true。
#   * 不适用 --seed / --n-task-combinations（MobileGym 是单实例任务）。
#   * 所有后端默认指向同一个模拟器 URL；--sim-urls 可给每个 backend 指定不同 URL。
#
# 用法：
#     ./eval_mobilegym.sh                 # 一键：起环境/智能体服务 → 跑测评
#     ./eval_mobilegym.sh run             # 显式 run：等价于无参数（起服务 → 跑测评）
#     ./eval_mobilegym.sh start           # 只起服务
#     ./eval_mobilegym.sh eval            # 只跑测评（复用 start 起的服务，可反复 resume）
#     ./eval_mobilegym.sh status          # 查看进程 / 端口 / 健康状态
#     ./eval_mobilegym.sh stop            # 停止本次启动的全部进程
#     ./eval_mobilegym.sh logs env 1      # 跟踪日志（env|agent|eval + 序号）
#     ./eval_mobilegym.sh --help
#
# 常用选项：
#     --sim-url URL             模拟器地址（默认 https://localhost:4180）
#     --sim-urls a,b,c          每个 backend 用不同模拟器（不足时复用最后一个）
#     --split NAME              MobileGym split（默认 test；支持 test+payment / 文件路径）
#     --ui-tree                 启用 DOM 派生的 accessibility tree（默认关闭；
#                               仅对 start 生效；后端 URL 追加 ui_tree=dom）
#     --no-ui-tree              关闭 accessibility tree（默认）
#     --headless / --no-headless
#     --browser-path PATH       Playwright 浏览器可执行文件（可选；用于系统不兼容时）
#     --startup-timeout SEC     单个模拟器启动等待（默认 120）
#     --processes N             进程/模型数（默认 4）
#     --backends-per-env N      每个进程的浏览器后端数（默认 1）
#     --parallel N              每个测评进程的并发（默认 = backends-per-env）
#     --models a,b,c,d          各进程模型名（默认从 /v1/models 自动读取）
#     --policy NAME             智能体策略（默认 fast-agent；vision-only 配置）
#     --policy-config PATH      策略配置底稿，按进程覆盖 model / base_url
#     --input-charset NAME      jev-mobile 的输入字符集：ascii|unicode
#                               （默认：jev-mobile 为 unicode，其它策略不改）
#     --output PATH             结果文件底名（自动加 _<模型名> 后缀）
#     --task-timeout SEC        单任务超时（默认 600）
#     --max-retry N             单任务出现 error 时的同进程重试次数（默认 10，0 关闭）
#     --task-index-range A-B    只跑任务列表的 A-B 段；所有进程共用同一段
#     --tasks id1,id2           只跑指定任务
#     --difficulty LEVEL        只跑指定难度（可重复：easy / medium / hard）
#     --no-resume               不跳过已记录的任务
#     --resume                  跳过已记录的任务（默认）
#     --rerun-fail              重跑各模型上一轮 score=0 的题（error 为空）
#                               源文件自动取 <run-dir>/results/<模型>.json
#                               结果写到 <模型>_rerun.json，原文件不动
#                               超时/取消等带 error 的行请用 eval --resume 重跑
#     --rerun-fail-dir DIR      指定失败源所在的 results 目录（默认当前 run-dir）
#     --no-evidence             不写证据包
#     --no-debug                不打印 agent 实时事件
#     --mobilegym-dir DIR       bench_env 所在的 mobilegym 仓库（默认 ../mobilegym）
#     --env-port-base N         环境服务端口基准（默认 9001）
#     --agent-port-base N       智能体服务端口基准（默认 9301）
#     --model-port-base N       模型服务端口基准（默认 29100；配合 --model-host）
#     --model-host HOST         模型服务主机（默认 = --host；裸主机自动拼 :port/v1）
#     --model-base-url URL      模型服务完整 base URL（如 https://api.openai.com/v1）
#     --model-urls a,b,c        每个进程的模型 base URL（不足时复用最后一个）
#     --health-timeout SEC      服务启动等待（默认 600；K 个浏览器是串行启动的）
#     --no-model-check          跳过模型服务健康检查
#     --host ADDR               环境/智能体服务的监听与连接地址（默认 127.0.0.1）
#     --run-dir DIR             指定运行目录（默认 runs/mobilegym_launch_<时间戳>）
#     --dry-run                 只打印将要执行的命令
#
# 说明：
#   * MobileGym 的 env 服务需要 `bench_env` 在 PYTHONPATH 上；脚本默认找
#     <repo>/../mobilegym，也可用 --mobilegym-dir 或 $MOBILEGYM_DIR 指定。
#   * 每个 backend 启动一个 Chromium。进程 i 的 env 服务会串行启动 K 个浏览器，
#     所以 --health-timeout 默认 600；K 较大时请相应调大。
#   * 模拟器 URL 必须可访问（脚本用 curl -k 预检）；同一个 URL 可以被多个独立
#     浏览器 page 共享，每个 page 的状态互不干扰。
#   * 默认 vision-only：没有 a11y tree，runner 会给 has_ui_tree=false 的后端
#     自动加 vision_only=true；agent policy 必须支持截图决策。
#     --ui-tree 会在后端 URL 追加 ui_tree=dom，输出 DOM 合成的 portal a11y tree
#     （has_ui_tree=true），可配合 --policy jev-mobile 使用。该树与 AndroidWorld/
#     MobileWorld 的字段契约一致，但不等于原生 uiautomator 树；iframe 内容可观察、
#     当前合成输入不可操作；结果与官方 vision-only 数字不可直接比较。
#   * MobileGym 的任务大量需要输入中文；--policy jev-mobile 时脚本默认把
#     input_charset 设为 unicode（可用 --input-charset ascii 关闭）。
#   * jev-mobile 默认 grounding_mode="box"（全链路截图决策，不需要 a11y tree）：
#     需要官方 TypeSafe（jev_base_url=https://api.typesafe.ai）和 TYPESAFE_API_KEY；
#     跨网络时 export HTTPS_PROXY=http://127.0.0.1:7890 或在 --policy-config
#     里设 "jev_proxy"。api.yuai.dev 不支持 Box，想要旧路径请显式设
#     "grounding_mode": "a11y"（可配合 --ui-tree）。
#   * --host 只用于环境/智能体服务；模型服务可用 --model-host / --model-base-url(s)
#     指向远程 sglang 或第三方 OpenAI 兼容端点。默认 fast-agent 配置通过
#     OPENAI_API_KEY 传 key；第三方端点如果没有 /models，加 --no-model-check。
#   * 服务在测评结束后保持运行，方便直接改参数重跑（./eval_mobilegym.sh eval）。
#   * 结果文件：<run-dir>/results/<模型>.json；汇总：<run-dir>/results/summary.json。
# =============================================================================
set -uo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")"
ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
cd "$ROOT" || { echo "无法进入脚本目录: $ROOT" >&2; exit 1; }

# ------------------------------- 默认可调项 --------------------------------
CMD="run"
PROCS=4
BACKENDS_PER_ENV=1
ENV_PORT_BASE=9001
AGENT_PORT_BASE=9301
MODEL_PORT_BASE=29100
HOST="127.0.0.1"
MODEL_HOST=""
MODEL_BASE_URLS=""
SIM_URL="https://localhost:4180"
SIM_URLS=""
SPLIT="test"
UI_TREE=0
HEADLESS=1
BROWSER_PATH=""
STARTUP_TIMEOUT=120
POLICY="fast-agent"
POLICY_CONFIG="$ROOT/agent-harness/agent_harness/agents/fast_agent/configs/config.json"
INPUT_CHARSET=""
MOBILEGYM_DIR="${MOBILEGYM_DIR:-}"
MODELS=""
OUTPUT=""
PARALLEL=""
TASK_TIMEOUT=600
MAX_RETRY=10
TASK_INDEX_RANGE=""
TASKS=""
DIFFICULTY=()
RESUME=1
RERUN_FAIL=0
RERUN_FAIL_DIR=""
EVIDENCE=1
DEBUG=1
HEALTH_TIMEOUT=600
SKIP_MODEL_CHECK=0
DRY_RUN=0
RUN_DIR=""

MOBILERUN_BIN="$ROOT/.venv/bin/mobilerun"
MOBILERUN_PY="$ROOT/.venv/bin/python"
AGENT_HARNESS_BIN="$ROOT/agent-harness/.venv/bin/agent-harness"
RUNS_ROOT="$ROOT/runs"

# -------------------------------- 小工具 -----------------------------------
info() { printf '[eval_mobilegym] %s\n' "$*" >&2; }
warn() { printf '[eval_mobilegym] 警告: %s\n' "$*" >&2; }
err()  { printf '[eval_mobilegym] 错误: %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

usage() { awk 'NR > 2 && /^#/ { sub(/^# ?/, ""); print; next } NR > 2 { exit }' "$SCRIPT_PATH"; }

port_in_use() {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk -v p="$p" '
      NR > 1 { n = split($4, a, ":"); if (a[n] == p) found = 1 }
      END { exit(found ? 0 : 1) }'
  else
    timeout 1 bash -c "</dev/tcp/127.0.0.1/$p" 2>/dev/null
  fi
}

pid_alive() {
  local pid="$1"
  [ -n "$pid" ] && [ -d "/proc/$pid" ] && kill -0 "$pid" 2>/dev/null
}

kill_tree() {
  local sig="$1" pid="$2"
  pid_alive "$pid" || return 0
  kill "-$sig" -- "-$pid" 2>/dev/null || kill "-$sig" "$pid" 2>/dev/null || true
}

http_ok() { curl -fsS -m "${2:-5}" -o /dev/null "$1" 2>/dev/null; }

# 占用某端口的 PID（找不到时输出空）
port_owner() {
  local p="$1"
  ss -ltnp 2>/dev/null | awk -v p="$p" '
    NR > 1 {
      n = split($4, a, ":"); if (a[n] != p) next
      if (match($0, /pid=[0-9]+/)) { print substr($0, RSTART + 4, RLENGTH - 4); exit }
    }'
}

# 某个 PID 属于哪个 run 目录（按 procs.tsv 反查）；找不到输出空。
run_dir_of_pid() {
  local pid="$1" f d
  if [ -n "${RUN_DIR:-}" ]; then
    for f in "$RUN_DIR/procs.tsv" "$RUN_DIR/procs.tsv.prev"; do
      [ -f "$f" ] || continue
      if awk -F'\t' -v p="$pid" '$4 == p { found = 1 } END { exit !found }' "$f"; then
        printf '%s' "$RUN_DIR"; return 0
      fi
    done
  fi
  for f in "$RUNS_ROOT"/mobilegym_launch_*/procs.tsv "$RUNS_ROOT"/mobilegym_launch_*/procs.tsv.prev; do
    [ -f "$f" ] || continue
    d="$(dirname "$f")"
    [ -d "$d" ] || continue
    if awk -F'\t' -v p="$pid" '$4 == p { found = 1 } END { exit !found }' "$f"; then
      printf '%s' "$d"; return 0
    fi
  done
  return 1
}

require_tools() {
  command -v curl >/dev/null 2>&1 || die "需要 curl"
  command -v jq   >/dev/null 2>&1 || die "需要 jq（配置生成与结果汇总都要用）"
  [ -x "$MOBILERUN_BIN" ] || MOBILERUN_BIN="$(command -v mobilerun || true)"
  [ -n "$MOBILERUN_BIN" ] || die "找不到 mobilerun（.venv/bin/mobilerun 不存在，也不在 PATH）"
  [ -x "$MOBILERUN_PY" ] || MOBILERUN_PY="$(dirname "$MOBILERUN_BIN")/python"
  [ -x "$MOBILERUN_PY" ] || MOBILERUN_PY="$(command -v python3 || true)"
  [ -n "$MOBILERUN_PY" ] || die "找不到 mobilerun 的 Python 解释器"
  [ -x "$AGENT_HARNESS_BIN" ] || AGENT_HARNESS_BIN="$(command -v agent-harness || true)"
  [ -n "$AGENT_HARNESS_BIN" ] || die "找不到 agent-harness（agent-harness/.venv/bin/agent-harness 不存在，也不在 PATH）"
}

# 进程 i 的服务端口
env_port()   { echo $((ENV_PORT_BASE   + $1 - 1)); }
agent_port() { echo $((AGENT_PORT_BASE + $1 - 1)); }
model_port() { echo $((MODEL_PORT_BASE + $1 - 1)); }

pidfile_of() { printf '%s/pids/%s_%s.pid' "$RUN_DIR" "$1" "$2"; }
logfile_of() { printf '%s/logs/%s_%s.log' "$RUN_DIR" "$1" "$2"; }

latest_run_dir() {
  local link d
  for link in "$RUNS_ROOT/latest_mobilegym.$(hostname)" "$RUNS_ROOT/latest_mobilegym"; do
    if [ -L "$link" ]; then
      d="$(readlink -f -- "$link" 2>/dev/null)"
      [ -n "$d" ] && [ -d "$d" ] && { printf '%s' "$d"; return 0; }
    elif [ -f "$link" ]; then
      d="$(cat "$link")"
      [ -n "$d" ] && [ -d "$d" ] && { printf '%s' "$d"; return 0; }
    fi
  done
  return 1
}

lookup_port() {
  local role="$1" idx="$2"
  [ -f "$RUN_DIR/procs.tsv" ] || return 1
  local p
  p="$(awk -F'\t' -v r="$role" -v i="$idx" '$1 == r && $2 == i { print $3; exit }' "$RUN_DIR/procs.tsv")"
  [ -n "$p" ] || return 1
  printf '%s' "$p"
}

recorded_indices() {
  local role="$1"
  [ -f "$RUN_DIR/procs.tsv" ] || return 0
  awk -F'\t' -v r="$role" '$1 == r { print $2 }' "$RUN_DIR/procs.tsv" | sort -n -u
}

record_proc() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$RUN_DIR/procs.tsv"; }

# 后台拉起一个进程；stdout 返回其 PID（setsid 的 wrapper，可直接 wait）
spawn() {
  local log="$1" pidfile="$2"; shift 2
  local wrapper pid w
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    (dry-run) %s\n' "$*" >&2
    printf '0' > "$pidfile"
    printf '0'
    return 0
  fi
  mkdir -p "$(dirname "$log")" "$(dirname "$pidfile")"
  rm -f "$pidfile"
  printf '===== %s  log start =====\n' "$(date '+%F %T')" >> "$log"
  setsid /bin/bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$pidfile" "$@" \
    >> "$log" 2>&1 < /dev/null &
  wrapper=$!
  for w in $(seq 1 50); do
    [ -s "$pidfile" ] && break
    sleep 0.2
  done
  pid="$(cat "$pidfile" 2>/dev/null)"
  [ -n "$pid" ] && [ "$pid" != "0" ] || { pid="$wrapper"; printf '%s' "$pid" > "$pidfile"; }
  printf '%s' "$pid"
}

wait_http() {
  local name="$1" url="$2" pid="$3" timeout="$4" log="$5"
  local start=$SECONDS elapsed
  while :; do
    if http_ok "$url"; then
      elapsed=$((SECONDS - start))
      info "$name 就绪 (${url}，${elapsed}s)"
      return 0
    fi
    if ! pid_alive "$pid"; then
      err "$name 启动失败：进程 $pid 已退出；日志尾部："
      tail -n 30 "$log" 2>/dev/null | sed 's/^/    /' >&2
      return 1
    fi
    elapsed=$((SECONDS - start))
    if [ "$elapsed" -ge "$timeout" ]; then
      err "$name 等待超时（${timeout}s, ${url}）；日志尾部："
      tail -n 30 "$log" 2>/dev/null | sed 's/^/    /' >&2
      return 1
    fi
    sleep 2
  done
}

uri() { jq -rn --arg v "$1" '$v|@uri'; }
headless_bool() { [ "$HEADLESS" -eq 1 ] && printf 'true' || printf 'false'; }

# ------------------------------ 参数解析 ----------------------------------
case "${1:-}" in
  run|start|eval|stop|status|restart) CMD="$1"; shift ;;
  logs)
    CMD="logs"; shift
    LOG_ROLE="${1:-}"; LOG_IDX="${2:-}"
    [ -n "${1:-}" ] && shift
    [ -n "${1:-}" ] && shift
    ;;
  -h|--help|help) usage; exit 0 ;;
  -*) ;;
  "") ;;
  *) die "未知命令: $1（--help 查看用法）" ;;
esac

_args=()
for _a in "$@"; do
  case "$_a" in
    --*=*) _args+=( "${_a%%=*}" "${_a#*=}" ) ;;
    *)     _args+=( "$_a" ) ;;
  esac
done
set -- "${_args[@]+"${_args[@]}"}"

while [ $# -gt 0 ]; do
  case "$1" in
    --processes)           PROCS="${2:?缺数值}"; shift 2 ;;
    --backends-per-env)    BACKENDS_PER_ENV="${2:?缺数值}"; shift 2 ;;
    --env-port-base)       ENV_PORT_BASE="${2:?缺数值}"; shift 2 ;;
    --agent-port-base)     AGENT_PORT_BASE="${2:?缺数值}"; shift 2 ;;
    --model-port-base)     MODEL_PORT_BASE="${2:?缺数值}"; shift 2 ;;
    --model-host)          MODEL_HOST="${2:?缺数值}"; shift 2 ;;
    --model-base-url|--model-urls) MODEL_BASE_URLS="${2:?缺数值}"; shift 2 ;;
    --host)                HOST="${2:?缺数值}"; shift 2 ;;
    --sim-url)             SIM_URL="${2:?缺数值}"; shift 2 ;;
    --sim-urls)            SIM_URLS="${2:?缺数值}"; shift 2 ;;
    --split)               SPLIT="${2:?缺数值}"; shift 2 ;;
    --ui-tree)             UI_TREE=1; shift ;;
    --no-ui-tree)          UI_TREE=0; shift ;;
    --browser-path)        BROWSER_PATH="${2:?缺路径}"; shift 2 ;;
    --startup-timeout)     STARTUP_TIMEOUT="${2:?缺数值}"; shift 2 ;;
    --headless)            HEADLESS=1; shift ;;
    --no-headless)         HEADLESS=0; shift ;;
    --policy)              POLICY="${2:?缺数值}"; shift 2 ;;
    --policy-config)       POLICY_CONFIG="${2:?缺路径}"; shift 2 ;;
    --input-charset)       INPUT_CHARSET="${2:?缺数值}"; shift 2 ;;
    --mobilegym-dir)       MOBILEGYM_DIR="${2:?缺路径}"; shift 2 ;;
    --models)              MODELS="${2:?缺数值}"; shift 2 ;;
    --output)              OUTPUT="${2:?缺数值}"; shift 2 ;;
    --parallel)            PARALLEL="${2:?缺数值}"; shift 2 ;;
    --task-timeout)        TASK_TIMEOUT="${2:?缺数值}"; shift 2 ;;
    --max-retry)           MAX_RETRY="${2:?缺数值}"; shift 2 ;;
    --task-index-range)    TASK_INDEX_RANGE="${2:?缺数值}"; shift 2 ;;
    --tasks)               TASKS="${2:?缺数值}"; shift 2 ;;
    --difficulty)          DIFFICULTY+=( "${2:?缺数值}" ); shift 2 ;;
    --health-timeout)      HEALTH_TIMEOUT="${2:?缺数值}"; shift 2 ;;
    --run-dir)             RUN_DIR="${2:?缺数值}"; shift 2 ;;
    --no-resume)           RESUME=0; shift ;;
    --resume)              RESUME=1; shift ;;
    --rerun-fail)          RERUN_FAIL=1; shift ;;
    --rerun-fail-dir)      RERUN_FAIL=1; RERUN_FAIL_DIR="${2:?缺目录}"; shift 2 ;;
    --no-evidence)         EVIDENCE=0; shift ;;
    --evidence)            EVIDENCE=1; shift ;;
    --no-debug)            DEBUG=0; shift ;;
    --debug)               DEBUG=1; shift ;;
    --no-model-check)      SKIP_MODEL_CHECK=1; shift ;;
    --dry-run)             DRY_RUN=1; shift ;;
    -h|--help)             usage; exit 0 ;;
    *)                     die "未知选项: $1（--help 查看用法）" ;;
  esac
done

for _n in PROCS BACKENDS_PER_ENV ENV_PORT_BASE AGENT_PORT_BASE MODEL_PORT_BASE \
          STARTUP_TIMEOUT TASK_TIMEOUT HEALTH_TIMEOUT; do
  case "${!_n}" in
    ''|*[!0-9]*) die "$_n 必须是正整数（当前: ${!_n}）" ;;
  esac
done
case "$MAX_RETRY" in
  ''|*[!0-9]*) die "--max-retry 必须是非负整数（当前: $MAX_RETRY）" ;;
esac
[ "$PROCS" -ge 1 ] || die "--processes 至少为 1"
[ "$BACKENDS_PER_ENV" -ge 1 ] || die "--backends-per-env 至少为 1"
case "$INPUT_CHARSET" in
  ""|ascii|unicode) ;;
  *) die "--input-charset 必须是 ascii 或 unicode（当前: $INPUT_CHARSET）" ;;
esac
[ -n "$SPLIT" ] || die "--split 不能为空"
[ -n "$PARALLEL" ] && { case "$PARALLEL" in ''|*[!0-9]*) die "--parallel 必须是正整数";; esac; }
[ "$PARALLEL" -ge 1 ] 2>/dev/null || [ -z "$PARALLEL" ] || die "--parallel 至少为 1"
if [ -n "$TASK_INDEX_RANGE" ]; then
  case "$TASK_INDEX_RANGE" in
    *[!0-9-]*|*-*-*) die "--task-index-range 应写成 A-B（如 1-116），当前: $TASK_INDEX_RANGE" ;;
    *-*) ;;
    *) die "--task-index-range 应写成 A-B（如 1-116），当前: $TASK_INDEX_RANGE" ;;
  esac
fi

# ------------------------------ 预检 ---------------------------------------
SIM_URL_LIST=()
MODEL_URL_LIST=()

resolve_mobilegym_dir() {
  if [ -z "$MOBILEGYM_DIR" ]; then
    local d
    for d in "$ROOT/../mobilegym" "$ROOT/../mobilegym-repo" "$ROOT/mobilegym"; do
      if [ -f "$d/bench_env/__init__.py" ]; then MOBILEGYM_DIR="$(cd "$d" && pwd -P)"; break; fi
    done
  fi
  [ -n "$MOBILEGYM_DIR" ] || die "找不到 mobilegym 仓库；用 --mobilegym-dir 或 \$MOBILEGYM_DIR 指定（需要 bench_env/）"
  [ -f "$MOBILEGYM_DIR/bench_env/__init__.py" ] || die "$MOBILEGYM_DIR 下没有 bench_env/__init__.py"
  export PYTHONPATH="$MOBILEGYM_DIR${PYTHONPATH:+:$PYTHONPATH}"
  info "bench_env: $MOBILEGYM_DIR"
}

parse_sim_urls() {
  SIM_URL_LIST=()
  if [ -n "$SIM_URLS" ]; then
    local u
    IFS=',' read -r -a _urls <<< "$SIM_URLS"
    for u in "${_urls[@]}"; do
      u="$(printf '%s' "$u" | tr -d '[:space:]')"
      [ -n "$u" ] && SIM_URL_LIST+=( "$u" )
    done
    [ "${#SIM_URL_LIST[@]}" -ge 1 ] || die "--sim-urls 为空"
  else
    SIM_URL_LIST=( "$SIM_URL" )
  fi
}

sim_url_for() {
  local j="$1" idx
  idx=$((j - 1))
  [ "$idx" -ge "${#SIM_URL_LIST[@]}" ] && idx=$(( ${#SIM_URL_LIST[@]} - 1 ))
  printf '%s' "${SIM_URL_LIST[$idx]}"
}

parse_model_urls() {
  MODEL_URL_LIST=()
  [ -n "$MODEL_BASE_URLS" ] || return 0
  local u
  IFS=',' read -r -a _murls <<< "$MODEL_BASE_URLS"
  for u in "${_murls[@]}"; do
    u="$(printf '%s' "$u" | tr -d '[:space:]')"
    u="${u%/}"
    [ -n "$u" ] && MODEL_URL_LIST+=( "$u" )
  done
  [ "${#MODEL_URL_LIST[@]}" -ge 1 ] || die "--model-base-url/--model-urls 为空"
}

# 进程 i 的模型 base URL：--model-urls / --model-base-url 优先，
# 其次 --model-host（裸主机自动拼 :port/v1），最后回落到 --host。
model_base_url_for() {
  local i="$1" idx host url
  if [ "${#MODEL_URL_LIST[@]}" -gt 0 ]; then
    idx=$((i - 1))
    [ "$idx" -ge "${#MODEL_URL_LIST[@]}" ] && idx=$(( ${#MODEL_URL_LIST[@]} - 1 ))
    printf '%s' "${MODEL_URL_LIST[$idx]}"
    return 0
  fi
  host="${MODEL_HOST:-$HOST}"
  case "$host" in
    *://*)
      url="${host%/}"
      case "$url" in
        */v1) ;;
        *) url="$url/v1" ;;
      esac
      printf '%s' "$url"
      ;;
    *)
      printf 'http://%s:%s/v1' "$host" "$(model_port "$i")"
      ;;
  esac
}

# 模型端点的 /models 检查：默认 fast-agent 用 OPENAI_API_KEY，第三方端点也复用它。
model_http_ok() {
  local url="$1" timeout="${2:-5}"
  if [ -n "${OPENAI_API_KEY:-}" ] && [ "$OPENAI_API_KEY" != "EMPTY" ]; then
    curl -fsS -m "$timeout" -H "Authorization: Bearer $OPENAI_API_KEY" -o /dev/null "$url" 2>/dev/null
  else
    curl -fsS -m "$timeout" -o /dev/null "$url" 2>/dev/null
  fi
}

preflight_mobilegym() {
  resolve_mobilegym_dir
  parse_sim_urls
  if [ -n "$BROWSER_PATH" ]; then
    [ -x "$BROWSER_PATH" ] || die "--browser-path 不可执行: $BROWSER_PATH"
  fi
  local browsers=$((PROCS * BACKENDS_PER_ENV))
  if [ "$browsers" -gt 4 ]; then
    warn "将启动 $browsers 个 Chromium 浏览器（processes=$PROCS × backends-per-env=$BACKENDS_PER_ENV）；注意内存与 inotify 配额"
  else
    info "将启动 $browsers 个 Chromium 浏览器"
  fi
  [ "$DRY_RUN" -eq 1 ] && { info "dry-run：跳过模拟器/依赖健康检查"; return 0; }

  local u
  for u in "${SIM_URL_LIST[@]}"; do
    curl -k -fsS -m 8 -o /dev/null "$u" \
      || die "模拟器不可达: $u（先确认 MobileGym 已启动；HTTPS 自签证书用 curl -k 探测）"
    info "模拟器: $u"
  done
  "$MOBILERUN_PY" -c "import bench_env" 2>/dev/null \
    || die "mobilerun 的 Python 无法 import bench_env；检查 PYTHONPATH=$MOBILEGYM_DIR 和 mobilegym 依赖（playwright/Pillow/tqdm/opencc）"
}

declare -A PROCESS_MODEL=()

preflight_models() {
  local i url model k
  local -a wanted=()
  [ -n "$MODELS" ] && IFS=',' read -r -a wanted <<< "$MODELS"

  for i in $(seq 1 "$PROCS"); do
    url="$(model_base_url_for "$i")"
    if [ "$SKIP_MODEL_CHECK" -eq 0 ] && [ "$DRY_RUN" -eq 0 ] && ! model_http_ok "$url/models" 5; then
      die "模型服务 $url 不可用（进程 $i）。
    每个进程需要一个 OpenAI 兼容端点（默认 http://$HOST:$((MODEL_PORT_BASE + i - 1))/v1）。
    用 --models / --model-host / --model-base-url / --model-port-base 调整，
    或加 --no-model-check 跳过检查。"
    fi
    if [ "${#wanted[@]}" -gt 0 ]; then
      k=$((i - 1))
      [ "$k" -ge "${#wanted[@]}" ] && k=$(( ${#wanted[@]} - 1 ))
      model="$(printf '%s' "${wanted[k]}" | tr -d '[:space:]')"
    elif [ "$DRY_RUN" -eq 1 ]; then
      model="$POLICY"
    else
      local -a auth=()
      if [ -n "${OPENAI_API_KEY:-}" ] && [ "$OPENAI_API_KEY" != "EMPTY" ]; then
        auth=( -H "Authorization: Bearer $OPENAI_API_KEY" )
      fi
      model="$(curl -fsS -m 5 "${auth[@]+"${auth[@]}"}" "$url/models" 2>/dev/null \
        | jq -r '.data[0].id // empty' 2>/dev/null)"
    fi
    if [ -z "$model" ]; then
      warn "读不到进程 $i 的模型名（$url/models）；用 --models 指定"
      model="$POLICY"
    fi
    PROCESS_MODEL[$i]="$model"
    info "进程 $i 模型: ${model}  ($url)"
  done
}

preflight_ports() {
  [ "$DRY_RUN" -eq 1 ] && return 0
  local i port
  for i in $(seq 1 "$PROCS"); do
    for port in "$(env_port "$i")" "$(agent_port "$i")"; do
      if port_in_use "$port"; then
        local holder; holder="$(port_owner "$port")"
        die "端口 $port 已被占用（进程 $i）${holder:+，pid=$holder $(ps -o args= -p "$holder" 2>/dev/null | cut -c1-90)}
    先 ./eval_mobilegym.sh stop，或换一套端口：--env-port-base 9101 --agent-port-base 9401"
      fi
    done
  done
}

# ------------------------------ 目录准备 -----------------------------------
prepare_run_dir() {
  [ -n "$RUN_DIR" ] || RUN_DIR="$RUNS_ROOT/mobilegym_launch_$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$RUN_DIR"/{logs,pids,configs,results,evidence} \
    || die "无法创建运行目录: $RUN_DIR"
  if [ -s "$RUN_DIR/procs.tsv" ]; then
    cp -f "$RUN_DIR/procs.tsv" "$RUN_DIR/procs.tsv.prev" 2>/dev/null || true
  fi
  : > "$RUN_DIR/procs.tsv"
  : > "$RUN_DIR/models.tsv"
}

publish_run_dir() {
  [ "$DRY_RUN" -eq 1 ] && return 0
  local abs; abs="$(readlink -f -- "$RUN_DIR" 2>/dev/null || printf '%s' "$RUN_DIR")"
  case "$abs" in
    "$RUNS_ROOT"/*)
      ln -sfn "$abs" "$RUNS_ROOT/latest_mobilegym.$(hostname)" 2>/dev/null || true
      ;;
    *)
      warn "运行目录不在 $RUNS_ROOT 下（$abs），不更新 latest_mobilegym 指针"
      ;;
  esac
}

# ------------------------------ 阶段 1/2 -----------------------------------
build_backend_specs() {
  local j url q specs=""
  for ((j = 1; j <= BACKENDS_PER_ENV; j++)); do
    url="$(sim_url_for "$j")"
    q="split=$(uri "$SPLIT")&headless=$(headless_bool)&startup_timeout=$STARTUP_TIMEOUT"
    [ "$UI_TREE" -eq 1 ] && q+="&ui_tree=dom"
    [ -n "$BROWSER_PATH" ] && q+="&browser_path=$(uri "$BROWSER_PATH")"
    specs+="${specs:+,}mg$((j - 1))=mobile-gym@${url}?${q}"
  done
  printf '%s' "$specs"
}

start_env_services() {
  local i port log specs pid failed=0
  for i in $(seq 1 "$PROCS"); do
    port="$(env_port "$i")"
    log="$(logfile_of env "$i")"
    specs="$(build_backend_specs)"
    pid="$(spawn "$log" "$(pidfile_of env "$i")" \
      "$MOBILERUN_BIN" serve env --host "$HOST" --port "$port" --backends "$specs")"
    [ "$DRY_RUN" -eq 1 ] || record_proc env "$i" "$port" "$pid" "$log"
  done
  [ "$DRY_RUN" -eq 1 ] && return 0

  for i in $(seq 1 "$PROCS"); do
    wait_http "环境服务 $i" "http://$HOST:$(env_port "$i")/v1/health" \
      "$(cat "$(pidfile_of env "$i")")" "$HEALTH_TIMEOUT" "$(logfile_of env "$i")" \
      || failed=$((failed + 1))
  done
  [ "$failed" -eq 0 ] || die "$failed 个环境服务未就绪（日志: $RUN_DIR/logs）"
}

write_policy_config() {
  local i="$1" out="$2" model="$3" base charset
  base="$(model_base_url_for "$i")"
  charset="$INPUT_CHARSET"
  if [ -z "$charset" ] && [ "$POLICY" = "jev-mobile" ]; then
    charset="unicode"
  fi
  if [ -f "$POLICY_CONFIG" ]; then
    if [ -n "$charset" ]; then
      jq --arg m "$model" --arg b "$base" --arg cs "$charset" \
        '.model = $m | .base_url = $b | .input_charset = $cs' "$POLICY_CONFIG" > "$out" \
        || die "无法生成策略配置 $out（底稿: $POLICY_CONFIG）"
    else
      jq --arg m "$model" --arg b "$base" '.model = $m | .base_url = $b' "$POLICY_CONFIG" > "$out" \
        || die "无法生成策略配置 $out（底稿: $POLICY_CONFIG）"
    fi
  else
    if [ -n "$charset" ]; then
      jq -n --arg m "$model" --arg b "$base" --arg cs "$charset" \
        '{model: $m, base_url: $b, input_charset: $cs}' > "$out" \
        || die "无法生成策略配置 $out"
    else
      jq -n --arg m "$model" --arg b "$base" '{model: $m, base_url: $b}' > "$out" \
        || die "无法生成策略配置 $out"
    fi
  fi
}

start_agent_services() {
  local i port log cfg cap pid failed=0
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  cap="${PARALLEL:-$BACKENDS_PER_ENV}"
  [ "$cap" -ge 1 ] 2>/dev/null || cap=1

  for i in $(seq 1 "$PROCS"); do
    port="$(agent_port "$i")"
    log="$(logfile_of agent "$i")"
    cfg="$RUN_DIR/configs/p${i}.json"
    write_policy_config "$i" "$cfg" "${PROCESS_MODEL[$i]:-$POLICY}"
    printf '%s\t%s\n' "$i" "${PROCESS_MODEL[$i]:-$POLICY}" >> "$RUN_DIR/models.tsv"

    pid="$(spawn "$log" "$(pidfile_of agent "$i")" \
      "$AGENT_HARNESS_BIN" serve \
        --policy "$POLICY" --policy-config "$cfg" --name "$POLICY" \
        --host "$HOST" --port "$port" \
        --max-concurrent "$cap" \
        --evidence-dir "$RUN_DIR/evidence/p${i}" \
        --log-level info)"
    [ "$DRY_RUN" -eq 1 ] || record_proc agent "$i" "$port" "$pid" "$log"
  done
  [ "$DRY_RUN" -eq 1 ] && return 0

  for i in $(seq 1 "$PROCS"); do
    wait_http "智能体服务 $i" "http://$HOST:$(agent_port "$i")/v1/health" \
      "$(cat "$(pidfile_of agent "$i")")" "$HEALTH_TIMEOUT" "$(logfile_of agent "$i")" \
      || failed=$((failed + 1))
  done
  [ "$failed" -eq 0 ] || die "$failed 个智能体服务未就绪（日志: $RUN_DIR/logs）"
}

start_all() {
  require_tools
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  preflight_mobilegym
  parse_model_urls
  preflight_models
  preflight_ports
  prepare_run_dir
  info "运行目录: $RUN_DIR"
  start_env_services
  start_agent_services
  publish_run_dir
  info "就绪：${PROCS} 个环境服务（每个 ${BACKENDS_PER_ENV} 个 MobileGym 浏览器）+ ${PROCS} 个智能体服务"
}

# ------------------------------ 阶段 3 -------------------------------------
model_of_process() {
  local i="$1" m
  if [ -f "$RUN_DIR/models.tsv" ]; then
    m="$(awk -F'\t' -v i="$i" '$1 == i { print $2; exit }' "$RUN_DIR/models.tsv")"
    [ -n "$m" ] && { printf '%s' "$m"; return 0; }
  fi
  if [ -f "$RUN_DIR/configs/p${i}.json" ]; then
    m="$(jq -r '.model // empty' "$RUN_DIR/configs/p${i}.json" 2>/dev/null)"
    [ -n "$m" ] && { printf '%s' "$m"; return 0; }
  fi
  m="${PROCESS_MODEL[$i]:-}"
  [ -n "$m" ] && { printf '%s' "$m"; return 0; }
  local -a auth=()
  if [ -n "${OPENAI_API_KEY:-}" ] && [ "$OPENAI_API_KEY" != "EMPTY" ]; then
    auth=( -H "Authorization: Bearer $OPENAI_API_KEY" )
  fi
  curl -fsS -m 5 "${auth[@]+"${auth[@]}"}" "$(model_base_url_for "$i")/models" 2>/dev/null \
    | jq -r '.data[0].id // empty' 2>/dev/null
}

aggregate_results() {
  local -a files=()
  local -a rerun_files=()
  local f
  for f in "$RUN_DIR"/results/*.json; do
    [ -f "$f" ] || continue
    case "$f" in */summary.json|*/.model-map.json) continue ;; esac
    case "$(basename "$f")" in
      *_rerun.json) rerun_files+=( "$f" ) ;;
      *)            files+=( "$f" ) ;;
    esac
  done
  if [ "${#files[@]}" -eq 0 ]; then
    [ "${#rerun_files[@]}" -gt 0 ] && show_rerun_summary
    warn "没有结果文件可汇总"
    return 0
  fi

  local mapfile="$RUN_DIR/results/.model-map.json"
  local -a mapitems=()
  local base m
  for f in "${files[@]}"; do
    base="$(basename "$f" .json)"
    m=""
    if [ -f "$RUN_DIR/models.tsv" ]; then
      m="$(awk -F'\t' -v b="$base" \
        '{ s = $2; gsub(/[^A-Za-z0-9._-]/, "_", s); if (s == b) { print $2; exit } }' \
        "$RUN_DIR/models.tsv")"
    fi
    [ -n "$m" ] || m="$base"
    mapitems+=( "$(jq -cn --arg p "$f" --arg m "$m" '{path: $p, model: $m}')" )
  done
  printf '%s\n' "${mapitems[@]+"${mapitems[@]}"}" | jq -s '.' > "$mapfile"

  local merged="$RUN_DIR/results/summary.json"
  jq -s --slurpfile map "$mapfile" '
    ($map[0] | map({(.path): .model}) | add // {}) as $labels
    | [ .[]
      | { output: (.results_path // "?"),
          model:  ($labels[.results_path] // .model // .agent // "?"),
          tasks:  (.summary.tasks  // 0),
          scored: (.summary.scored // 0),
          errors: (.summary.errors // 0),
          mean_score:       (.summary.mean_score       // 0),
          mean_gated_score: (.summary.mean_gated_score // 0),
          completion_rate:  (.summary.completion_rate  // 0),
          total_runtime_s:  (.summary.total_runtime_s  // 0),
          total_tokens:     (.summary.total_tokens     // 0),
          total_requests:   (.summary.total_requests   // 0) }
    ]
    | sort_by(-.mean_gated_score, -.mean_score)
    | { models: ., count: length }' "${files[@]}" > "$merged" \
    || { warn "汇总失败"; return 0; }

  printf '\n汇总: %s\n\n' "$merged"
  printf '%-24s %6s %7s %8s %9s %7s %9s\n' \
    MODEL TASKS SCORED MEAN GATED DONE ERRORS
  printf '%s\n' '------------------------------------------------------------------------------'
  jq -r '.models[] | [
      (.model | if length > 24 then .[0:23] + "…" else . end),
      (.tasks | tostring),
      (.scored | tostring),
      (.mean_score | tostring),
      (.mean_gated_score | tostring),
      (.completion_rate | tostring),
      (.errors | tostring)
    ] | @tsv' "$merged" | awk -F'\t' '{printf "%-24s %6s %7s %8.4f %9.4f %7.4f %9s\n", $1,$2,$3,$4,$5,$6,$7}'

  if [ "$(jq -r '.count' "$merged")" -gt 1 ]; then
    printf '\n相对最佳模型（按门控平均分）:\n'
    jq -r '.models as $m | ($m[0].mean_gated_score) as $best
      | $m[] | "\(.model)\t\(.mean_gated_score - $best)"' "$merged" \
      | awk -F'\t' '{printf "  %-24s %+.4f\n", $1, $2}'
  fi

  show_rerun_summary
}

show_rerun_summary() {
  local -a rerun_files=()
  local f
  for f in "$RUN_DIR"/results/*_rerun.json; do
    [ -f "$f" ] || continue
    rerun_files+=( "$f" )
  done
  [ "${#rerun_files[@]}" -gt 0 ] || return 0

  printf '\n重跑增量（失败子集，不计入上表）:\n'
  printf '  %-24s %5s %7s %8s %s\n' MODEL TASKS RUN_OK MEAN_RERUN "合并后(估计)"
  local rf stem base merged_est
  for rf in "${rerun_files[@]}"; do
    stem="$(basename "$rf" _rerun.json)"
    base="$RUN_DIR/results/${stem}.json"
    merged_est=""
    if [ -f "$base" ]; then
      merged_est="$(jq -rs '
        .[0] as $old | .[1] as $new
        | (($old.tasks // []) | map({key: "\(.task_type)#\(.task_idx)", value: .}) | from_entries) as $m
        | (($new.tasks // []) | map({key: "\(.task_type)#\(.task_idx)", value: .}) | from_entries) as $n
        | ($m + $n) | [.[] | .score] | map(select(. != null))
        | if length > 0 then ((add / length) * 10000 | floor) / 10000 else null end
      ' "$base" "$rf" 2>/dev/null)"
      [ -n "$merged_est" ] && [ "$merged_est" != "null" ] || merged_est=""
    fi
    jq -r --arg est "${merged_est:-—}" '
      "\(.summary.tasks // 0)\t\([.tasks[]? | select((.error == null) and ((.score // 0) > 0))] | length)\t\((.summary.mean_score // 0) * 10000 | floor | . / 10000)"' \
      "$rf" | awk -F'\t' -v m="$stem" -v e="$merged_est" \
      '{printf "  %-24s %5s %7s %8.4f %s\n", m, $1, $2, $3, (e == "" ? "—" : sprintf("%.4f", e))}'
  done
  printf '  （合并后(估计) = 用重跑结果覆盖基线同题后的全量平均分，仅供参考）\n'
}

cmd_eval() {
  if [ "$DRY_RUN" -eq 1 ]; then
    [ -n "$RUN_DIR" ] || RUN_DIR="$(latest_run_dir 2>/dev/null || true)"
    [ -n "$RUN_DIR" ] || RUN_DIR="$RUNS_ROOT/mobilegym_launch_dryrun"
  else
    [ -n "$RUN_DIR" ] || RUN_DIR="$(latest_run_dir)" \
      || die "找不到上一次 start 的运行目录；先 ./eval_mobilegym.sh start，或用 --run-dir 指定"
    [ -f "$RUN_DIR/procs.tsv" ] || die "$RUN_DIR/procs.tsv 不存在；先 ./eval_mobilegym.sh start"
  fi

  local -a shards=()
  local i port aport
  if [ "$DRY_RUN" -eq 1 ]; then
    shards=( $(recorded_indices agent) )
    [ "${#shards[@]}" -gt 0 ] || shards=( $(seq 1 "$PROCS") )
  else
    for i in $(recorded_indices agent); do
      aport="$(lookup_port agent "$i" 2>/dev/null)" || continue
      port="$(lookup_port env "$i" 2>/dev/null)" || env_port "$i"
      http_ok "http://$HOST:$port/v1/health" 5  || { warn "环境服务 (:$port) 不健康，跳过该模型"; continue; }
      http_ok "http://$HOST:$aport/v1/health" 5 || { warn "智能体服务 $i (:$aport) 不健康，跳过该模型"; continue; }
      shards+=( "$i" )
    done
  fi
  if [ "${#shards[@]}" -eq 0 ]; then
    die "没有可用的 env/agent 服务对；先 ./eval_mobilegym.sh start"
  fi

  if [ "$DRY_RUN" -eq 0 ]; then
    awk -F'\t' '$1 != "eval"' "$RUN_DIR/procs.tsv" > "$RUN_DIR/procs.tsv.tmp" \
      && mv -f "$RUN_DIR/procs.tsv.tmp" "$RUN_DIR/procs.tsv"
    mkdir -p "$RUN_DIR/results"
  fi

  local total=0
  if [ -z "$TASK_INDEX_RANGE" ] && [ -z "$TASKS" ] && [ "$DRY_RUN" -eq 0 ]; then
    local eport0; eport0="$(lookup_port env "${shards[0]}")"
    total="$(curl -fsS -m 30 "http://$HOST:$eport0/v1/tasks" 2>/dev/null \
      | jq -r '[.[].id] | reduce .[] as $x ([]; if any(.[]; . == $x) then . else . + [$x] end) | length' \
      2>/dev/null)"
    case "$total" in ''|*[!0-9]*)
      die "无法读取任务清单（http://$HOST:$eport0/v1/tasks）" ;;
    esac
    info "任务总数: $total；${#shards[@]} 个进程各跑完整任务集"
  fi

  local idx out parallel log pid eport aport model safe_model src evdir
  local -a launched=()
  local -a skipped_rerun=()
  local cap="${PARALLEL:-$BACKENDS_PER_ENV}"
  [ "$cap" -ge 1 ] 2>/dev/null || cap=1

  for idx in "${shards[@]}"; do
    model="$(model_of_process "$idx")"
    safe_model="$(printf '%s' "${model:-p$idx}" | tr -c 'A-Za-z0-9._-' '_')"
    out="$RUN_DIR/results/${safe_model}.json"
    if [ -n "$OUTPUT" ]; then
      case "$OUTPUT" in
        *.json) out="${OUTPUT%.json}_${safe_model}.json" ;;
        *)      out="${OUTPUT}_${safe_model}.json" ;;
      esac
    fi

    src=""
    if [ "$RERUN_FAIL" -eq 1 ]; then
      local src_dir; src_dir="${RERUN_FAIL_DIR:-$RUN_DIR/results}"
      src="$src_dir/${safe_model}.json"
      if [ ! -f "$src" ]; then
        warn "模型 ${model:-?} 没有源结果文件（$src），跳过"
        skipped_rerun+=( "$idx" )
        continue
      fi
      out="$RUN_DIR/results/${safe_model}_rerun.json"
    fi

    eport="$(lookup_port env "$idx" 2>/dev/null)" || eport="$(env_port "$idx")"
    aport="$(lookup_port agent "$idx" 2>/dev/null)" || aport="$(agent_port "$idx")"
    evdir="$RUN_DIR/evidence/p${idx}"
    [ "$RERUN_FAIL" -eq 1 ] && evdir="$RUN_DIR/evidence/p${idx}_rerun"

    local -a cmd=( "$MOBILERUN_BIN" eval
      --env "http://$HOST:$eport"
      --agent "http://$HOST:$aport"
      --parallel "$cap"
      --task-timeout "$TASK_TIMEOUT"
      --max-retry "$MAX_RETRY"
      --evidence-dir "$evdir"
      --output "$out" )
    [ -n "$TASK_INDEX_RANGE" ] && cmd+=( --task-index-range "$TASK_INDEX_RANGE" )
    [ -n "$TASKS" ] && cmd+=( --tasks "$TASKS" )
    local d
    for d in "${DIFFICULTY[@]+"${DIFFICULTY[@]}"}"; do cmd+=( --difficulty "$d" ); done
    [ "$RERUN_FAIL" -eq 1 ] && cmd+=( --rerun-fail "$src" )
    [ "$RESUME" -eq 1 ] && cmd+=( --resume )
    [ "$EVIDENCE" -eq 0 ] && cmd+=( --no-evidence )
    [ "$DEBUG" -eq 1 ] && cmd+=( --debug )

    if [ "$RERUN_FAIL" -eq 1 ]; then
      info "重跑 $idx：模型 ${model:-?}，失败源 $src，并发 $cap，结果 $out"
    else
      info "测评 $idx：模型 ${model:-?}，并发 $cap，结果 $out"
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
      printf '    (dry-run) %s\n' "${cmd[*]}" >&2
      continue
    fi

    log="$(logfile_of eval "$idx")"
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"
    printf '===== %s  eval start (p%s, model=%s) =====\n' \
      "$(date '+%F %T')" "$idx" "${model:-?}" >> "$log"
    setsid /bin/bash -c 'echo $$ > "$1"; shift; exec "$@"' _ \
      "$(pidfile_of eval "$idx")" "${cmd[@]}" >> "$log" 2>&1 < /dev/null &
    pid=$!
    launched+=( "$idx:$pid" )
    record_proc eval "$idx" "0" "$pid" "$log"
  done

  if [ "${#skipped_rerun[@]}" -gt 0 ]; then
    info "跳过 ${#skipped_rerun[@]} 个没有源结果文件的模型: ${skipped_rerun[*]}"
  fi
  [ "$DRY_RUN" -eq 1 ] && { info "dry-run：未执行测评"; return 0; }
  if [ "${#launched[@]}" -eq 0 ]; then
    if [ "$RERUN_FAIL" -eq 1 ]; then
      warn "没有任何模型有可重跑的源结果文件（源目录: ${RERUN_FAIL_DIR:-$RUN_DIR/results}）"
      return 0
    fi
    die "没有启动任何测评进程"
  fi

  info "已启动 ${#launched[@]} 个测评进程（每进程一个模型、各自跑完整任务集），等待全部结束"
  local failed=0 entry
  for entry in "${launched[@]}"; do
    idx="${entry%%:*}"; pid="${entry##*:}"
    if wait "$pid"; then
      info "测评 $idx（$(model_of_process "$idx")）完成 ✅"
    else
      warn "测评 $idx（$(model_of_process "$idx")）退出码非零（部分任务失败或服务中断）；见 $(logfile_of eval "$idx")"
      failed=$((failed + 1))
    fi
  done

  aggregate_results
  [ "$failed" -eq 0 ] || return 1
  return 0
}

# --------------------------------- 状态 -------------------------------------
cmd_status() {
  [ -n "$RUN_DIR" ] || RUN_DIR="$(latest_run_dir)" || { info "还没有运行记录"; return 0; }
  [ -f "$RUN_DIR/procs.tsv" ] || { info "没有进程记录: $RUN_DIR/procs.tsv"; return 0; }

  printf '运行目录: %s\n\n' "$RUN_DIR"
  printf '%-6s %-4s %-6s %-8s %-9s %s\n' ROLE IDX PORT PID HEALTH URL
  local role idx port pid log health url rc=0
  while IFS=$'\t' read -r role idx port pid log; do
    [ -z "${role:-}" ] && continue
    if ! pid_alive "$pid"; then
      health="down"; rc=1; url=""
    elif [ "$role" = "eval" ]; then
      health="running"; url="$(basename "${log:-}")"
    else
      url="http://$HOST:$port/v1/health"
      if http_ok "$url" 3; then health="ready"; else health="starting"; rc=1; fi
    fi
    printf '%-6s %-4s %-6s %-8s %-9s %s\n' "$role" "$idx" "$port" "$pid" "$health" "$url"
  done < "$RUN_DIR/procs.tsv"
  return "$rc"
}

# --------------------------------- 日志 -------------------------------------
cmd_logs() {
  [ -n "$RUN_DIR" ] || RUN_DIR="$(latest_run_dir)" || die "还没有运行记录"
  [ -n "${LOG_ROLE:-}" ] && [ -n "${LOG_IDX:-}" ] || die "用法: ./eval_mobilegym.sh logs <env|agent|eval> <idx>"
  local f; f="$(logfile_of "$LOG_ROLE" "$LOG_IDX")"
  [ -f "$f" ] || die "日志不存在: $f"
  exec tail -n 60 -F "$f"
}

# --------------------------------- 停止 -------------------------------------
orphan_pids_on_ports() {
  local port pid args f
  for f in "$RUN_DIR/procs.tsv" "$RUN_DIR/procs.tsv.prev"; do
    [ -f "$f" ] || continue
    while IFS=$'\t' read -r role idx port pid log; do
      [ -z "${port:-}" ] && continue
      [ "$port" = "0" ] && continue
      pid="$(port_owner "$port")"
      [ -n "$pid" ] || continue
      args="$(ps -o args= -p "$pid" 2>/dev/null)"
      case "$args" in
        *mobilerun*serve\ env*|*agent-harness*serve*) printf '%s\n' "$pid" ;;
      esac
    done < "$f"
  done | sort -u
}

cmd_stop() {
  [ -n "$RUN_DIR" ] || RUN_DIR="$(latest_run_dir)" || { info "没有可停止的运行记录"; return 0; }
  [ -f "$RUN_DIR/procs.tsv" ] || { info "没有进程记录: $RUN_DIR/procs.tsv"; return 0; }

  local pass role idx port pid log
  local -a targets=()

  for pass in eval agent env; do
    local f
    for f in "$RUN_DIR/procs.tsv" "$RUN_DIR/procs.tsv.prev"; do
      [ -f "$f" ] || continue
      while IFS=$'\t' read -r role idx port pid log; do
        [ "$role" = "$pass" ] || continue
        if pid_alive "$pid"; then
          info "停止 $role $idx (pid $pid)"
          kill_tree TERM "$pid"
          targets+=( "$pid" )
        fi
      done < "$f"
    done
  done

  local orphan seen
  while read -r orphan; do
    [ -n "$orphan" ] || continue
    seen=0
    for pid in "${targets[@]}"; do
      [ "$pid" = "$orphan" ] && { seen=1; break; }
    done
    [ "$seen" -eq 1 ] && continue
    warn "停止未记录的遗留服务 (pid $orphan，占着记录中的端口)"
    kill_tree TERM "$orphan"
    targets+=( "$orphan" )
  done < <(orphan_pids_on_ports)

  if [ "${#targets[@]}" -eq 0 ]; then
    info "没有需要停止的进程（记录中的 PID 均不存活，端口也没有本次启动的遗留服务）"
    return 0
  fi

  local deadline=$((SECONDS + 30)) alive
  while [ "$SECONDS" -lt "$deadline" ]; do
    alive=0
    for pid in "${targets[@]}"; do
      pid_alive "$pid" && alive=$((alive + 1))
    done
    [ "$alive" -eq 0 ] && break
    sleep 1
  done
  for pid in "${targets[@]}"; do
    if pid_alive "$pid"; then
      warn "pid $pid 未在 30s 内退出，强制结束"
      kill_tree KILL "$pid"
    fi
  done

  info "已停止 ${#targets[@]} 个进程（记录保留在 $RUN_DIR/procs.tsv）"
}

# -------------------------------- 分发 -------------------------------------
case "$CMD" in
  run)
    start_all
    info "开始测评 —— ${PROCS} 个模型并行，每个进程各跑完整任务集"
    cmd_eval
    ;;
  start)
    start_all
    cmd_status
    ;;
  restart)
    cmd_stop
    start_all
    cmd_status
    ;;
  eval)
    require_tools
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
    parse_model_urls
    cmd_eval
    ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  *)      die "未知命令: $CMD" ;;
esac
