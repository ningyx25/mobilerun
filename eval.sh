#!/usr/bin/env bash
# =============================================================================
# eval.sh —— 一键测评：环境服务 → 智能体服务 → 测评
#
# 默认 4 个进程并行，进程 i 独占一套服务（下标从 1 开始）：
#
#     环境服务      http://<host>:9001 + (i-1)     每个服务带 4 个环境后端
#     环境后端      http://<host>:3801 + 4*(i-1) .. 3804 + 4*(i-1)
#     智能体服务    http://<host>:9301 + (i-1)     agent-harness worker
#     模型服务      http://<host>:29100 + (i-1)    sglang / OpenAI 兼容端点
#
# 每个进程跑【一整个任务集】，进程之间比的是模型、不是任务：
# 第 i 个进程用第 i 个模型端口在全部任务上跑一遍，四个进程并行 →
# 一份结果文件 = 一个模型的全量分数，最后汇总成「模型 × 分数」对比表。
#
# 即：第 1 个进程 = env 9001 / 后端 3801-3804 / agent 9301 / 模型 29100，
#     第 2 个进程 = env 9002 / 后端 3805-3808 / agent 9302 / 模型 29101，
#     第 3 个进程 = env 9003 / 后端 3809-3812 / agent 9303 / 模型 29102，
#     第 4 个进程 = env 9004 / 后端 3813-3816 / agent 9304 / 模型 29103。
#
# 用法：
#     ./eval.sh                   # 一键：起环境服务 → 起智能体服务 → 跑测评
#     ./eval.sh start             # 只起环境服务 + 智能体服务
#     ./eval.sh eval              # 只跑测评（复用 start 起的服务，可反复 resume）
#     ./eval.sh status            # 查看各进程 / 端口 / 健康状态
#     ./eval.sh stop              # 停止本次启动的全部进程
#     ./eval.sh logs env 1        # 跟踪日志（env|agent|eval + 序号）
#     ./eval.sh --help
#
# 常用选项：
#     --processes N             进程数（默认 4）
#     --env-port-base N         环境服务端口基准（默认 9001）
#     --agent-port-base N       智能体服务端口基准（默认 9301）
#     --backend-kind KIND       环境后端类型：android-world（默认）| mobile-world
#     --backend-port-base N     环境后端端口基准（默认 android-world 3801 / mobile-world 6801）
#     --backends-per-env N      每个环境服务的后端数（默认 4）
#     --model-port-base N       模型服务端口基准（默认 29100）
#     --models a,b,c,d          各进程的模型名（默认从 /v1/models 自动读取）
#     --policy NAME             智能体策略（默认 qwen35）
#     --policy-config PATH      策略配置底稿，按进程覆盖 model / base_url
#     --output PATH             结果文件底名（自动加 _<模型名> 后缀，一模型一个文件）
#     --parallel N              每个测评进程的并发（默认 = 该环境服务的后端数）
#     --task-timeout SEC        单任务超时（默认 3600）
#     --task-index-range A-B    只跑任务列表的 A-B 段；所有进程共用同一段
#     --seed N                  套件随机种子（AndroidWorld）
#     --n-task-combinations N   AndroidWorld 每任务实例数
#     --no-resume               不跳过已记录的任务
#     --rerun-fail              重跑各模型上一轮失败的题（error 为空且 score==0）
#                               源文件自动取 <run-dir>/results/<模型>.json
#                               结果写到 <模型>_rerun.json，原文件不动
#                               默认带 --resume：中断后再跑一次即续跑未完成的题；
#                               要强制全量重跑用 --no-resume
#     --rerun-fail-dir DIR      指定失败源所在的 results 目录（默认当前 run-dir）
#     --no-evidence             不写证据包
#     --no-debug                不打印 agent 实时事件
#     --allow-degraded          后端不可用时只用健康的后端（默认直接中止）
#     --pool                    池模式：只起一个环境服务统管全部后端，所有进程
#                               动态租用它（空闲后端可被其他进程接管）
#     --no-model-check          跳过模型服务健康检查
#     --host ADDR               监听地址（默认 127.0.0.1）
#     --run-dir DIR             指定运行目录（默认 runs/launch_<时间戳>）
#     --dry-run                 只打印将要执行的命令
#
# 说明：
#   * 环境服务启动时会对每个后端做 /health 连接，缺一个就起不来，所以 start
#     前先做后端预检；不可用的后端会列出来。
#   * mobile-world 后端用 680* 端口（一个端口 = 一台设备），后端规格写成
#     mwN=mobile-world@http://host:port；--backend-kind mobile-world 会同时把
#     端口基准切到 6801，并改用该后端需要的健康检查。
#   * 每个进程写自己的结果文件，全部结束后汇总 runs/.../results/summary.json。
#   * stop 只结束本次记录的 PID（先 eval，再 agent，最后 env）。
#   * 服务在测评结束后保持运行，方便直接改参数重跑（./eval.sh eval）。
#   * --pool：非池模式下进程 i 独占自己的 4 个后端；池模式下只起 env:9001
#     统管全部后端，所有测评进程共用一个租约池 —— 某个进程跑完放出后端，
#     其他还在跑的进程立刻能抢到。代价是 env 成为单点，且设备 I/O 汇聚到
#     同一个事件循环，所以默认关闭。
# =============================================================================
set -uo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")"
ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
cd "$ROOT" || { echo "无法进入脚本目录: $ROOT" >&2; exit 1; }

# ------------------------------- 默认可调项 --------------------------------
CMD="run"
PROCS=4
ENV_PORT_BASE=9001
AGENT_PORT_BASE=9301
BACKEND_KIND="android-world"
BACKEND_PORT_BASE=""          # 留空 = 按 BACKEND_KIND 取默认值
BACKENDS_PER_ENV=4
MODEL_PORT_BASE=29100
HOST="127.0.0.1"
POLICY="qwen35"
POLICY_CONFIG="$ROOT/agent-harness/agent_harness/agents/qwen35/configs/qwen35_aug.json"
MODELS=""
OUTPUT=""
PARALLEL=""
RESUME=1
RERUN_FAIL=0
RERUN_FAIL_DIR=""
TASK_TIMEOUT=3600
TASK_INDEX_RANGE=""
SEED=""
NCOMB=""
HEALTH_TIMEOUT=300
EVIDENCE=1
DEBUG=1
ALLOW_DEGRADED=0
SKIP_MODEL_CHECK=0
POOL=0
DRY_RUN=0
RUN_DIR=""

MOBILERUN_BIN="$ROOT/.venv/bin/mobilerun"
AGENT_HARNESS_BIN="$ROOT/agent-harness/.venv/bin/agent-harness"
RUNS_ROOT="$ROOT/runs"

# -------------------------------- 小工具 -----------------------------------
info() { printf '[eval] %s\n' "$*" >&2; }
warn() { printf '[eval] 警告: %s\n' "$*" >&2; }
err()  { printf '[eval] 错误: %s\n' "$*" >&2; }
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
# 共享 runs/ 下可能有别的机器的记录，所以还要确认目录在本机存在。
run_dir_of_pid() {
  local pid="$1" f d
  # 先看本次要用的 RUN_DIR（可能是 --run-dir 指定的任意路径）；
  # prepare_run_dir 已把上一版记录备份成 procs.tsv.prev，两个都查。
  if [ -n "${RUN_DIR:-}" ]; then
    for f in "$RUN_DIR/procs.tsv" "$RUN_DIR/procs.tsv.prev"; do
      [ -f "$f" ] || continue
      if awk -F'\t' -v p="$pid" '$4 == p { found = 1 } END { exit !found }' "$f"; then
        printf '%s' "$RUN_DIR"; return 0
      fi
    done
  fi
  for f in "$RUNS_ROOT"/launch_*/procs.tsv "$RUNS_ROOT"/launch_*/procs.tsv.prev; do
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
  [ -x "$AGENT_HARNESS_BIN" ] || AGENT_HARNESS_BIN="$(command -v agent-harness || true)"
  [ -n "$AGENT_HARNESS_BIN" ] || die "找不到 agent-harness（agent-harness/.venv/bin/agent-harness 不存在，也不在 PATH）"
}

# 进程 i 的服务端口
env_port()     { echo $((ENV_PORT_BASE   + $1 - 1)); }
agent_port()   { echo $((AGENT_PORT_BASE + $1 - 1)); }
model_port()   { echo $((MODEL_PORT_BASE + $1 - 1)); }
backend_port() { echo $((BACKEND_PORT_BASE + ($1 - 1) * BACKENDS_PER_ENV + $2 - 1)); }

pidfile_of() { printf '%s/pids/%s_%s.pid' "$RUN_DIR" "$1" "$2"; }
logfile_of() { printf '%s/logs/%s_%s.log' "$RUN_DIR" "$1" "$2"; }

# runs/ 可能被多台机器共享（同一块 /fs1），所以：
#   * 指针按主机名隔离，避免 node02 / node03 互相覆盖；
#   * 再回落到不带主机名的旧指针（兼容历史目录）；
#   * 指向的目标必须在本机真实存在（排除别的机器的 /tmp 路径）。
latest_run_dir() {
  local link d
  for link in "$RUNS_ROOT/latest_launch.$(hostname)" "$RUNS_ROOT/latest_launch"; do
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

# start 时记录下来的进程下标（升序）
recorded_indices() {
  local role="$1"
  [ -f "$RUN_DIR/procs.tsv" ] || return 0
  awk -F'\t' -v r="$role" '$1 == r { print $2 }' "$RUN_DIR/procs.tsv" | sort -n -u
}

# 进程 i 的 env / agent 端口：优先用 start 时记录的，缺失则按基准推算
shard_env_port()   { lookup_port env "$1"   2>/dev/null || env_port   "$1"; }
shard_agent_port() { lookup_port agent "$1" 2>/dev/null || agent_port "$1"; }

# 测评进程 idx 该连哪个环境服务：池模式下所有进程都连同一个 env:9001
eval_env_port() {
  if [ "$POOL" -eq 1 ]; then
    lookup_port env 1 2>/dev/null || env_port 1
  else
    shard_env_port "$1"
  fi
}

# 进程 i 被测的模型名：优先读 start 时记下的 models.tsv，其次 configs/p<i>.json，
# 再退回本次预检结果，最后退回模型端点自报的名字。
model_of_shard() {
  local i="$1" cfg m
  if [ -f "$RUN_DIR/models.tsv" ]; then
    m="$(awk -F'\t' -v i="$i" '$1 == i { print $2; exit }' "$RUN_DIR/models.tsv")"
    [ -n "$m" ] && { printf '%s' "$m"; return 0; }
  fi
  cfg="$RUN_DIR/configs/p${i}.json"
  if [ -f "$cfg" ]; then
    m="$(jq -r '.model // empty' "$cfg" 2>/dev/null)"
    [ -n "$m" ] && { printf '%s' "$m"; return 0; }
  fi
  m="${PROCESS_MODEL[$i]:-}"
  [ -n "$m" ] && { printf '%s' "$m"; return 0; }
  curl -fsS -m 5 "http://$HOST:$(model_port "$i")/v1/models" 2>/dev/null \
    | jq -r '.data[0].id // empty' 2>/dev/null
}

record_proc() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$RUN_DIR/procs.tsv"; }

# 后台拉起一个进程；stdout 返回其 PID（setsid 的 wrapper，可直接 wait）
#   用法: pid=$(spawn <log> <pidfile> <cmd...>)
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
  # 必须先清掉旧 pidfile：复用 --run-dir 时它带着上一轮的 PID，而下面的
  # 等待条件是「文件非空」—— 不清就会立刻读到那个已经死掉的旧 PID，
  # wait_http 随即判定「启动失败」，而新进程其实正在正常启动。
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
      tail -n 25 "$log" 2>/dev/null | sed 's/^/    /' >&2
      return 1
    fi
    elapsed=$((SECONDS - start))
    if [ "$elapsed" -ge "$timeout" ]; then
      err "$name 等待超时（${timeout}s, ${url}）；日志尾部："
      tail -n 25 "$log" 2>/dev/null | sed 's/^/    /' >&2
      return 1
    fi
    sleep 2
  done
}

# 该环境服务的健康后端数（读 /v1/health）
env_capacity() {
  local port="$1" n
  n="$(curl -fsS -m 5 "http://$HOST:$port/v1/health" 2>/dev/null \
        | jq -r '[.backends[]? | select(.state != "down")] | length' 2>/dev/null)"
  printf '%s' "${n:-0}"
}

# -------------------------------- 参数解析 ----------------------------------
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
    --env-port-base)       ENV_PORT_BASE="${2:?缺数值}"; shift 2 ;;
    --agent-port-base)     AGENT_PORT_BASE="${2:?缺数值}"; shift 2 ;;
    --backend-kind)        BACKEND_KIND="${2:?缺数值}"; shift 2 ;;
    --backend-port-base)   BACKEND_PORT_BASE="${2:?缺数值}"; shift 2 ;;
    --backends-per-env)    BACKENDS_PER_ENV="${2:?缺数值}"; shift 2 ;;
    --model-port-base)     MODEL_PORT_BASE="${2:?缺数值}"; shift 2 ;;
    --models)              MODELS="${2:?缺数值}"; shift 2 ;;
    --policy)              POLICY="${2:?缺数值}"; shift 2 ;;
    --policy-config)       POLICY_CONFIG="${2:?缺数值}"; shift 2 ;;
    --output)              OUTPUT="${2:?缺数值}"; shift 2 ;;
    --parallel)            PARALLEL="${2:?缺数值}"; shift 2 ;;
    --task-timeout)        TASK_TIMEOUT="${2:?缺数值}"; shift 2 ;;
    --task-index-range)    TASK_INDEX_RANGE="${2:?缺数值}"; shift 2 ;;
    --seed)                SEED="${2:?缺数值}"; shift 2 ;;
    --n-task-combinations) NCOMB="${2:?缺数值}"; shift 2 ;;
    --host)                HOST="${2:?缺数值}"; shift 2 ;;
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
    --allow-degraded)      ALLOW_DEGRADED=1; shift ;;
    --pool)                POOL=1; shift ;;
    --no-pool)             POOL=0; shift ;;
    --no-model-check)      SKIP_MODEL_CHECK=1; shift ;;
    --dry-run)             DRY_RUN=1; shift ;;
    -h|--help)             usage; exit 0 ;;
    *)                     die "未知选项: $1（--help 查看用法）" ;;
  esac
done

# 后端类型：统一成 harness 认的写法，并按类型给端口基准补默认值
case "$BACKEND_KIND" in
  android-world|androidworld|aw)   BACKEND_KIND="android-world";  _kind_prefix="aw" ;;
  mobile-world|mobileworld|mw)     BACKEND_KIND="mobile-world";   _kind_prefix="mw" ;;
  *) die "--backend-kind 只支持 android-world / mobile-world（当前: $BACKEND_KIND）" ;;
esac
if [ -z "$BACKEND_PORT_BASE" ]; then
  case "$BACKEND_KIND" in
    mobile-world) BACKEND_PORT_BASE=6801 ;;
    *)            BACKEND_PORT_BASE=3801 ;;
  esac
fi

for _n in PROCS BACKENDS_PER_ENV ENV_PORT_BASE AGENT_PORT_BASE BACKEND_PORT_BASE MODEL_PORT_BASE TASK_TIMEOUT; do
  case "${!_n}" in
    ''|*[!0-9]*) die "$_n 必须是正整数（当前: ${!_n}）" ;;
  esac
done
[ "$PROCS" -ge 1 ] || die "--processes 至少为 1"
[ "$BACKENDS_PER_ENV" -ge 1 ] || die "--backends-per-env 至少为 1"
if [ -n "$TASK_INDEX_RANGE" ]; then
  case "$TASK_INDEX_RANGE" in
    *[!0-9-]*|*-*-*) die "--task-index-range 应写成 A-B（如 1-116），当前: $TASK_INDEX_RANGE" ;;
    *-*) ;;
    *) die "--task-index-range 应写成 A-B（如 1-116），当前: $TASK_INDEX_RANGE" ;;
  esac
fi
[ -n "$PARALLEL" ] && { case "$PARALLEL" in ''|*[!0-9]*) die "--parallel 必须是正整数";; esac; }

# ------------------------------ 预检：后端 ----------------------------------
# ACTIVE          —— 有可用后端的槽位；非池模式下即「进程号」，池模式下恒为 (1)
# ACTIVE_BACKENDS —— 槽位 → 该槽位环境服务要挂的后端端口列表
# ENV_SLOTS       —— 要起几个环境服务
# AGENT_SLOTS     —— 要起几个智能体服务（每个测一个模型）
ACTIVE=()
declare -A ACTIVE_BACKENDS=()
ENV_SLOTS=()
AGENT_SLOTS=()
POOL_BACKEND_COUNT=0
POOL_ENV_PORT=""
POOL_PARALLEL=""

# 后端是否真的能用。两类后端共用 /health，但 mobile-world 还要求返回
# {"ok": true}（驱动会在 connect 时校验这个字段，只回 200 会在启动时炸）。
backend_healthy() {
  local port="$1"
  case "$BACKEND_KIND" in
    mobile-world)
      local body
      body="$(curl -fsS -m 8 "http://$HOST:$port/health" 2>/dev/null)" || return 1
      printf '%s' "$body" | jq -e '.ok == true' >/dev/null 2>&1
      ;;
    *)
      http_ok "http://$HOST:$port/health" 8
      ;;
  esac
}

preflight_backends() {
  local i j port ports down=()
  local -a all_ports=()

  for ((i = 1; i <= PROCS; i++)); do
    ports=""
    for ((j = 1; j <= BACKENDS_PER_ENV; j++)); do
      port="$(backend_port "$i" "$j")"
      if backend_healthy "$port"; then
        ports+="${ports:+,}${port}"
        all_ports+=( "$port" )
      else
        down+=( "$port" )
        warn "后端 :$port（进程 $i 的第 $j 个 $BACKEND_KIND）不可用"
      fi
    done
    if [ -n "$ports" ]; then
      ACTIVE+=( "$i" )
      ACTIVE_BACKENDS[$i]="$ports"
    elif [ "$POOL" -eq 0 ]; then
      warn "进程 $i 没有可用后端，跳过"
    fi
  done

  if [ "${#down[@]}" -gt 0 ] && [ "$ALLOW_DEGRADED" -eq 0 ]; then
    die "以下 $BACKEND_KIND 后端不可用: ${down[*]}
    环境服务启动时会对每个后端做 /health 连接，缺一个就起不来。
    请先恢复这些后端（ssh 隧道 / 容器），或加 --allow-degraded。"
  fi

  if [ "$POOL" -eq 1 ]; then
    # 池模式：把所有健康后端挂到唯一的环境服务上，槽位恒为 1
    local all
    all="$(IFS=,; printf '%s' "${all_ports[*]}")"
    [ "${#all_ports[@]}" -gt 0 ] || die "池模式下没有任何可用的 $BACKEND_KIND 后端"
    ACTIVE=( 1 )
    ACTIVE_BACKENDS[1]="$all"
    POOL_BACKEND_COUNT="${#all_ports[@]}"
    POOL_ENV_PORT="$(env_port 1)"
  else
    [ "${#ACTIVE[@]}" -gt 0 ] || die "没有任何可用的 $BACKEND_KIND 后端（端口 ${BACKEND_PORT_BASE}..$((BACKEND_PORT_BASE + PROCS * BACKENDS_PER_ENV - 1))）"
  fi

  # 槽位分离：池模式一个 env 服务 + PROCS 个 agent；非池模式二者合并
  if [ "$POOL" -eq 1 ]; then
    ENV_SLOTS=( 1 )
    AGENT_SLOTS=( $(seq 1 "$PROCS") )
  else
    ENV_SLOTS=( "${ACTIVE[@]}" )
    AGENT_SLOTS=( "${ACTIVE[@]}" )
  fi
}

# ------------------------------ 预检：模型 ----------------------------------
declare -A PROCESS_MODEL=()

preflight_models() {
  local i port url model k
  local -a wanted=()
  [ -n "$MODELS" ] && IFS=',' read -r -a wanted <<< "$MODELS"

  # 模型是按「被测 agent」绑定的，池模式下与 env 槽位无关
  for i in "${AGENT_SLOTS[@]}"; do
    port="$(model_port "$i")"
    url="http://$HOST:$port/v1"

    if [ "$SKIP_MODEL_CHECK" -eq 0 ] && ! http_ok "$url/models" 5; then
      die "模型服务 $url 不可用（进程 $i）。
    每个进程需要一个 OpenAI 兼容端点（默认端口 $((MODEL_PORT_BASE + i - 1))）。
    用 --models / --model-port-base 调整，或加 --no-model-check 跳过检查。"
    fi

    if [ "${#wanted[@]}" -gt 0 ]; then
      k=$((i - 1))
      [ "$k" -ge "${#wanted[@]}" ] && k=$(( ${#wanted[@]} - 1 ))
      model="$(printf '%s' "${wanted[k]}" | tr -d '[:space:]')"
    else
      model="$(curl -fsS -m 5 "$url/models" 2>/dev/null | jq -r '.data[0].id // empty' 2>/dev/null)"
    fi
    if [ -z "$model" ]; then
      warn "读不到进程 $i 的模型名（$url/models）；用 --models 指定"
      model="$POLICY"
    fi
    PROCESS_MODEL[$i]="$model"
    info "进程 $i 模型: ${model}  ($url)"
  done

  # 池模式并发：公平份额 + 余量，让早跑完的进程能吸收别人放出的后端
  if [ "$POOL" -eq 1 ]; then
    local n_proc="${#AGENT_SLOTS[@]}"
    [ "$n_proc" -ge 1 ] || n_proc=1
    # ceil(total/n) + 2
    POOL_PARALLEL=$(( (POOL_BACKEND_COUNT + n_proc - 1) / n_proc + 2 ))
    info "池模式: ${POOL_BACKEND_COUNT} 个后端共享，每个进程默认并发 ${POOL_PARALLEL}"
  fi
}

# ------------------------------- 目录准备 -----------------------------------
prepare_run_dir() {
  [ -n "$RUN_DIR" ] || RUN_DIR="$RUNS_ROOT/launch_$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$RUN_DIR"/{logs,pids,configs,results,evidence} \
    || die "无法创建运行目录: $RUN_DIR"
  # 复用 --run-dir 时，旧的 procs.tsv 仍是「端口被谁占着」的唯一线索：
  # 启动中途 die 的进程不会留下任何记录。先备份再去清空，端口冲突时
  # run_dir_of_pid 才能反查到它，给出可执行的 stop/kill 建议。
  if [ -s "$RUN_DIR/procs.tsv" ]; then
    cp -f "$RUN_DIR/procs.tsv" "$RUN_DIR/procs.tsv.prev" 2>/dev/null || true
  fi
  : > "$RUN_DIR/procs.tsv"
  : > "$RUN_DIR/models.tsv"
}

# 只在服务真正就绪（或用户显式指定 --run-dir）后，才把 latest_launch 指过来，
# 免得一次预检失败的 start 把指针推到一个空目录上。
publish_run_dir() {
  # 指向共享区之外（比如本机 /tmp）的路径不要发布：别的机器根本访问不到，
  # 会污染共享指针并给出误导性的 stop 建议。
  local abs; abs="$(readlink -f -- "$RUN_DIR" 2>/dev/null || printf '%s' "$RUN_DIR")"
  case "$abs" in
    "$RUNS_ROOT"/*)
      ln -sfn "$abs" "$RUNS_ROOT/latest_launch.$(hostname)" 2>/dev/null || true
      ;;
    *)
      warn "运行目录不在 $RUNS_ROOT 下（$abs），不更新 latest_launch 指针"
      ;;
  esac
}

# 端口被占时给用户的处置建议：先按 PID 反查所属 run 目录，查不到就只给 kill。
port_conflict_hint() {
  local pid="$1" d
  printf '    最常见的原因：上一轮测评跑完后 env/agent 服务会保留（方便重跑 eval）。\n'
  if [ -n "$pid" ]; then
    if d="$(run_dir_of_pid "$pid" 2>/dev/null)" && [ -n "$d" ]; then
      printf '    停止它:  ./eval.sh stop --run-dir %s\n' "$d"
    else
      printf '    停止它:  kill %s    #（记录里找不到这个 PID，直接按 PID 停）\n' "$pid"
    fi
  fi
  printf '    或换一套端口: --env-port-base 9101 --agent-port-base 9401'
}

# ------------------------------- 阶段 1 -------------------------------------
start_env_services() {
  local i port log specs j pid failed=0
  for i in "${ENV_SLOTS[@]}"; do
    port="$(env_port "$i")"
    log="$(logfile_of env "$i")"
    specs=""
    for j in ${ACTIVE_BACKENDS[$i]//,/ }; do
      specs+="${specs:+,}${_kind_prefix}${j}=${BACKEND_KIND}@http://${HOST}:${j}"
    done
    if [ "$DRY_RUN" -eq 0 ] && port_in_use "$port"; then
      local holder hint
      holder="$(port_owner "$port")"
      hint="$(port_conflict_hint "$holder")"
      die "环境服务端口 $port 已被占用（进程 $i）${holder:+，占用者 pid=$holder $(ps -o args= -p "$holder" 2>/dev/null | cut -c1-90)}
$hint"
    fi
    pid="$(spawn "$log" "$(pidfile_of env "$i")" \
      "$MOBILERUN_BIN" serve env --host "$HOST" --port "$port" --backends "$specs")"
    [ "$DRY_RUN" -eq 1 ] || record_proc env "$i" "$port" "$pid" "$log"
  done
  [ "$DRY_RUN" -eq 1 ] && return 0

  for i in "${ENV_SLOTS[@]}"; do
    wait_http "环境服务 $i" "http://$HOST:$(env_port "$i")/v1/health" \
      "$(cat "$(pidfile_of env "$i")")" "$HEALTH_TIMEOUT" "$(logfile_of env "$i")" \
      || failed=$((failed + 1))
  done
  [ "$failed" -eq 0 ] || die "$failed 个环境服务未就绪（日志: $RUN_DIR/logs）"
}

# ------------------------------- 阶段 2 -------------------------------------
write_policy_config() {
  local i="$1" out="$2" model="$3" base="http://$HOST:$(model_port "$i")/v1"
  if [ -f "$POLICY_CONFIG" ]; then
    jq --arg m "$model" --arg b "$base" '.model = $m | .base_url = $b' "$POLICY_CONFIG" > "$out" \
      || die "无法生成策略配置 $out（底稿: $POLICY_CONFIG）"
  else
    jq -n --arg m "$model" --arg b "$base" '{model: $m, base_url: $b}' > "$out" \
      || die "无法生成策略配置 $out"
  fi
}

start_agent_services() {
  local i port log cfg cap pid failed=0
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

  for i in "${AGENT_SLOTS[@]}"; do
    port="$(agent_port "$i")"
    log="$(logfile_of agent "$i")"
    cfg="$RUN_DIR/configs/p${i}.json"
    write_policy_config "$i" "$cfg" "${PROCESS_MODEL[$i]}"
    # 记下「进程 → 被测模型」，汇总时用它给结果文件贴标签
    printf '%s\t%s\n' "$i" "${PROCESS_MODEL[$i]}" >> "$RUN_DIR/models.tsv"

    if [ -n "$PARALLEL" ]; then
      cap="$PARALLEL"
    elif [ "$POOL" -eq 1 ]; then
      # 池模式：和 eval 的默认并发保持一致，别让 agent 侧无限堆积 episode
      cap="$POOL_PARALLEL"
    else
      cap="$(env_capacity "$(env_port "$i")")"
    fi
    [ "$cap" -ge 1 ] 2>/dev/null || cap="$BACKENDS_PER_ENV"

    if [ "$DRY_RUN" -eq 0 ] && port_in_use "$port"; then
      local holder hint
      holder="$(port_owner "$port")"
      hint="$(port_conflict_hint "$holder")"
      die "智能体服务端口 $port 已被占用（进程 $i）${holder:+，占用者 pid=$holder $(ps -o args= -p "$holder" 2>/dev/null | cut -c1-90)}
$hint"
    fi
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

  for i in "${AGENT_SLOTS[@]}"; do
    wait_http "智能体服务 $i" "http://$HOST:$(agent_port "$i")/v1/health" \
      "$(cat "$(pidfile_of agent "$i")")" "$HEALTH_TIMEOUT" "$(logfile_of agent "$i")" \
      || failed=$((failed + 1))
  done
  [ "$failed" -eq 0 ] || die "$failed 个智能体服务未就绪（日志: $RUN_DIR/logs）"
}

# 池模式：对共享 env 服务上的每个后端各应用一次套件种子/实例数。
# 之后各 runner 不再传 --seed，避免它们并发无锁地互相重置套件。
apply_pool_seed() {
  local sidx="$1" eport id
  eport="$(eval_env_port "$sidx")"
  local -a ids=()
  while read -r id; do
    [ -n "$id" ] && ids+=( "$id" )
  done < <(curl -fsS -m 10 "http://$HOST:$eport/v1/health" 2>/dev/null \
            | jq -r '.backends[]?.id' 2>/dev/null)
  [ "${#ids[@]}" -gt 0 ] || { warn "池模式：读不到后端列表，跳过统一套件初始化"; return 0; }

  if [ "$DRY_RUN" -eq 1 ]; then
    info "池模式：将对 ${#ids[@]} 个后端统一应用 seed=${SEED:-–} n_task_combinations=${NCOMB:-–}"
    return 0
  fi

  local q="" t
  [ -n "$SEED" ] && q="seed=$SEED"
  [ -n "$NCOMB" ] && q="${q:+$q&}n_task_combinations=$NCOMB"
  info "池模式：统一应用套件设置到 ${#ids[@]} 个后端（${q}）"
  for id in "${ids[@]}"; do
    if ! curl -fsS -m 1800 -X POST \
        "http://$HOST:$eport/v1/backends/$id/reinitialize?$q" \
        -o /dev/null 2>/dev/null; then
      warn "后端 $id 套件初始化失败（继续，该后端可能使用旧实例集）"
    fi
  done
}

# ------------------------------- 阶段 3 -------------------------------------
cmd_eval() {
  [ -n "$RUN_DIR" ] || RUN_DIR="$(latest_run_dir)" \
    || die "找不到上一次 start 的运行目录；先 ./eval.sh start，或用 --run-dir 指定"
  [ -f "$RUN_DIR/procs.tsv" ] || die "$RUN_DIR/procs.tsv 不存在；先 ./eval.sh start"

  # 复用已启动的服务：要测的进程列表以 procs.tsv 为准（start 起了几个就测几个模型），
  # 所以 eval 子命令不需要、也不使用 --processes / --backends-per-env。
  if [ "${#ACTIVE[@]}" -eq 0 ] && { [ "$PROCS" -ne 4 ] || [ "$BACKENDS_PER_ENV" -ne 4 ]; }; then
    info "eval 复用 start 起好的服务，进程列表取自 $RUN_DIR/procs.tsv（--processes/--backends-per-env 在此不起作用）"
  fi
  # 分片列表以【agent】为准：池模式下 env 只有 1 个，但模型有 PROCS 个
  local -a shards=()
  local i port aport
  for i in $(recorded_indices agent); do
    aport="$(lookup_port agent "$i" 2>/dev/null)" || continue
    port="$(eval_env_port "$i")"
    http_ok "http://$HOST:$port/v1/health" 5  || { warn "环境服务 (:$port) 不健康，跳过该模型"; continue; }
    http_ok "http://$HOST:$aport/v1/health" 5 || { warn "智能体服务 $i (:$aport) 不健康，跳过该模型"; continue; }
    shards+=( "$i" )
  done
  if [ "${#shards[@]}" -eq 0 ]; then
    if [ "$DRY_RUN" -eq 1 ] && [ "${#ACTIVE[@]}" -gt 0 ]; then
      shards=( "${AGENT_SLOTS[@]}" )   # dry-run：还没真起服务，用计划中的 agent 槽位
    else
      die "没有可用的 env/agent 服务对；先 ./eval.sh start"
    fi
  fi

  # 清掉上一次的 eval 记录，避免重复行
  awk -F'\t' '$1 != "eval"' "$RUN_DIR/procs.tsv" > "$RUN_DIR/procs.tsv.tmp" \
    && mv -f "$RUN_DIR/procs.tsv.tmp" "$RUN_DIR/procs.tsv"
  mkdir -p "$RUN_DIR/results"

  # 每个进程跑【完整】任务集：进程之间比的是模型，不是任务。
  # --task-index-range 是给所有进程共同使用的筛选条件，不是切片。
  local total=0
  if [ -z "$TASK_INDEX_RANGE" ]; then
    local env_port_0; env_port_0="$(shard_env_port "${shards[0]}")"
    total="$(curl -fsS -m 30 "http://$HOST:$env_port_0/v1/tasks" 2>/dev/null \
      | jq -r '[.[].id] | reduce .[] as $x ([]; if any(.[]; . == $x) then . else . + [$x] end) | length' \
      2>/dev/null)"
    case "$total" in ''|*[!0-9]*)
      [ "$DRY_RUN" -eq 1 ] || die "无法读取任务清单（http://$HOST:$env_port_0/v1/tasks）"
      total=0 ;;
    esac
    if [ "$total" -gt 0 ]; then
      info "任务总数: $total；${#shards[@]} 个进程各跑完整任务集"
    else
      [ "$DRY_RUN" -eq 1 ] || die "任务清单为空"
      warn "dry-run：没有读到任务清单"
    fi
  else
    info "任务区间: $TASK_INDEX_RANGE；${#shards[@]} 个进程各跑这一整段"
  fi

  # 池模式下套件种子必须由这里统一应用一次：多个 runner 各自 reinitialize
  # 是无锁且互相破坏的（见 runner.py:526-544 与 app.py:123-131）。
  if [ "$POOL" -eq 1 ] && [ -n "$SEED$NCOMB" ] && [ "$BACKEND_KIND" = "android-world" ]; then
    apply_pool_seed "${shards[0]}"
  fi

  local idx out parallel log pid eport aport model safe_model src
  local -a launched=()
  local -a skipped_rerun=()
  for idx in "${shards[@]}"; do
    # 每个进程一个结果文件：与它的模型绑定，而不是与任务切片绑定
    model="$(model_of_shard "$idx")"
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
      # 重跑结果另存，基线文件保持不动
      out="$RUN_DIR/results/${safe_model}_rerun.json"
    fi

    eport="$(eval_env_port "$idx")"
    aport="$(shard_agent_port "$idx")"
    if [ -n "$PARALLEL" ]; then
      parallel="$PARALLEL"
    elif [ "$POOL" -eq 1 ]; then
      # 池模式：公平份额 + 余量，所有人共享同一个后端池
      parallel="$POOL_PARALLEL"
      [ "$parallel" -ge 1 ] 2>/dev/null || parallel="$BACKENDS_PER_ENV"
    else
      parallel="$(env_capacity "$eport")"
      [ "$parallel" -ge 1 ] 2>/dev/null || parallel="$BACKENDS_PER_ENV"
    fi

    log="$(logfile_of eval "$idx")"
    # 证据目录按任务名分桶（<task>_<idx>）写入且不清理已有内容，重跑会直接
    # 覆盖掉基线那一轮的截图与轨迹。重跑因此落到独立目录，基线证据保持可回溯。
    local evdir="$RUN_DIR/evidence/p${idx}"
    [ "$RERUN_FAIL" -eq 1 ] && evdir="$RUN_DIR/evidence/p${idx}_rerun"
    local -a cmd=( "$MOBILERUN_BIN" eval
      --env "http://$HOST:$eport"
      --agent "http://$HOST:$aport"
      --parallel "$parallel"
      --task-timeout "$TASK_TIMEOUT"
      --evidence-dir "$evdir"
      --output "$out" )
    [ -n "$TASK_INDEX_RANGE" ] && cmd+=( --task-index-range "$TASK_INDEX_RANGE" )
    [ "$RERUN_FAIL" -eq 1 ] && cmd+=( --rerun-fail "$src" )
    # 重跑也需要 --resume：中断后再跑一次就能从 <model>_rerun.json 接着来。
    # runner 只在 --resume 与 --rerun-fail 指向同一个文件时才拒绝
    # （runner.py:407），这里输出是 _rerun.json、源是基线文件，永不重合。
    [ "$RESUME" -eq 1 ] && cmd+=( --resume )
    [ "$EVIDENCE" -eq 0 ]   && cmd+=( --no-evidence )
    [ "$DEBUG" -eq 1 ]      && cmd+=( --debug )
    # 套件种子/实例数只对 AndroidWorld 有意义；mobile-world 的套件由服务端固定。
    # 池模式下套件已由 eval.sh 统一应用过一次，这里绝不能再传给 runner：
    # 每个 runner 都会遍历所有后端做 reinitialize，并发无锁会把别人正在跑的
    # episode 的实例集重置掉。
    if [ "$BACKEND_KIND" != "android-world" ]; then
      [ -n "$SEED$NCOMB" ] && warn "$BACKEND_KIND 不支持 --seed / --n-task-combinations，已忽略"
    elif [ "$POOL" -eq 0 ]; then
      [ -n "$SEED" ]  && cmd+=( --seed "$SEED" )
      [ -n "$NCOMB" ] && cmd+=( --n-task-combinations "$NCOMB" )
    fi

    if [ "$RERUN_FAIL" -eq 1 ]; then
      info "重跑 $idx：模型 ${model:-?}，失败源 $src，并发 $parallel，结果 $out"
    else
      info "测评 $idx：模型 ${model:-?}，并发 $parallel，结果 $out"
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
      printf '    (dry-run) %s\n' "${cmd[*]}" >&2
      continue
    fi
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"
    printf '===== %s  eval start (p%s, model=%s) =====\n' \
      "$(date '+%F %T')" "$idx" "${model:-?}" >> "$log"
    setsid /bin/bash -c 'echo $$ > "$1"; shift; exec "$@"' _ \
      "$(pidfile_of eval "$idx")" "${cmd[@]}" >> "$log" 2>&1 < /dev/null &
    pid=$!                       # wrapper 是本 shell 的直接子进程，可以 wait
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
      info "测评 $idx（$(model_of_shard "$idx")）完成 ✅"
    else
      warn "测评 $idx（$(model_of_shard "$idx")）退出码非零（部分任务失败或服务中断）；见 $(logfile_of eval "$idx")"
      failed=$((failed + 1))
    fi
  done

  aggregate_results
  [ "$failed" -eq 0 ] || return 1
  return 0
}

aggregate_results() {
  # 重跑文件（*_rerun.json）只含失败子集，混进主表会让那一行指标失真，
  # 所以单独收集，主表只统计完整的基线文件。
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
  [ "${#files[@]}" -gt 0 ] || {
    [ "${#rerun_files[@]}" -gt 0 ] && show_rerun_summary
    warn "没有结果文件可汇总"
    return 0
  }

  # 结果文件 → 模型名：结果文件名就是清洗后的模型名；清洗会丢字符，
  # 所以再拿 models.tsv 按「清洗后相等」回查原始名。
  local mapfile="$RUN_DIR/results/.model-map.json"
  local -a mapitems=()
  local f base m
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
  # 每个结果文件是【同一个任务集在一个模型上】的一轮完整评测，所以这里
  # 不做相加平均，而是摊平成「模型 × 指标」一张对比表。
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

  # 同一任务集在多个模型上的结果：给出每个模型相对最好成绩的差值
  if [ "$(jq -r '.count' "$merged")" -gt 1 ]; then
    printf '\n相对最佳模型（按门控平均分）:\n'
    jq -r '.models as $m | ($m[0].mean_gated_score) as $best
      | $m[] | "\(.model)\t\(.mean_gated_score - $best)"' "$merged" \
      | awk -F'\t' '{printf "  %-24s %+.4f\n", $1, $2}'
  fi

  show_rerun_summary
}

# 重跑增量：*_rerun.json 只含失败子集，不与主表混算。这里给出每个模型的
# 「重跑 N 题 / 成功 M 题」，并在有完整基线时给出合并后的估计分（仅展示）。
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
      # 按 task_type#task_idx 把重跑行覆盖到基线行上，再算均值
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
  [ -n "${LOG_ROLE:-}" ] && [ -n "${LOG_IDX:-}" ] || die "用法: ./eval.sh logs <env|agent|eval> <idx>"
  local f; f="$(logfile_of "$LOG_ROLE" "$LOG_IDX")"
  [ -f "$f" ] || die "日志不存在: $f"
  exec tail -n 60 -F "$f"
}

# --------------------------------- 停止 -------------------------------------
# 收集本机仍在监听该端口的、属于本次 eval 的进程 PID。
# 用于兜住「启动中途 die，PID 没来得及写进 procs.tsv」的遗留服务 —— 那种
# 进程还在占端口，但记录里查不到，旧版 stop 会静默报告「已停止」。
orphan_pids_on_ports() {
  local port pid args f
  # 当前记录 + 启动前备份的记录都要看：一次 start 会在清空 procs.tsv 之后
  # 才因端口冲突退出，被占端口只存在于 .prev 里。
  for f in "$RUN_DIR/procs.tsv" "$RUN_DIR/procs.tsv.prev"; do
    [ -f "$f" ] || continue
    while IFS=$'\t' read -r role idx port pid log; do
      [ -z "${port:-}" ] && continue
      [ "$port" = "0" ] && continue
      pid="$(port_owner "$port")"
      [ -n "$pid" ] || continue
      # 只认我们自己拉起的服务，别误杀同端口上的其他程序
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

  # 记录里没有、但确实占着本服务端口的遗留进程（跳过已在上一步停止的）
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

# --------------------------------- 启动 -------------------------------------
start_all() {
  require_tools
  # 预检先做：预检失败时不留下任何目录，也不动 latest_launch
  preflight_backends
  preflight_models
  prepare_run_dir
  info "运行目录: $RUN_DIR"
  start_env_services
  start_agent_services
  publish_run_dir
  if [ "$POOL" -eq 1 ]; then
    info "就绪：1 个环境服务（:$(env_port 1)，${POOL_BACKEND_COUNT} 个后端，所有进程共享）+ ${#AGENT_SLOTS[@]} 个智能体服务"
  else
    info "环境服务 + 智能体服务就绪（${#ACTIVE[@]} 个进程）"
  fi
}

# -------------------------------- 分发 -------------------------------------
case "$CMD" in
  run)
    start_all
    info "开始测评 —— ${#AGENT_SLOTS[@]} 个模型并行，每个进程各跑完整任务集"
    cmd_eval
    # 服务留着方便重跑（./eval.sh eval）；下次 start 会先要求 stop 或换端口
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
    cmd_eval
    ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  *)      die "未知命令: $CMD" ;;
esac
