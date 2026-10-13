#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_ROOT="${MASKWAM_CHECKPOINT_ROOT:-$ROOT/ckpt}"
DATA_ROOT="${MASKWAM_DATA_ROOT:-$ROOT/realdata}"
cd "$ROOT"

# 配置参数
CKPT="${CKPT:-$CHECKPOINT_ROOT/MaskWAM-libero-ckpt}"
SUITES=("libero_object" "libero_spatial" "libero_goal" "libero_10") # "libero_object" "libero_spatial" "libero_goal""libero_10"
NUM_TRIALS=50
NUM_TASKS=10
GPUS=(0 1 2 3 4 5 6 7) # 1 2 3 4 5 6 7

# T-shape specific parameters
USE_MASK=false
MEAN_STD_PATH="${MEAN_STD_PATH:-$DATA_ROOT/libero_vam/qpos_mean_std_online.pkl}"

T5_EMBEDS="${T5_EMBEDS:-}"

# 日志根目录
LOG_ROOT="./eval_logs_T_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

# ============================================================================
# GPU 管理核心逻辑 (带自动补位功能)
# ============================================================================
declare -A gpu_pid
for gpu_id in "${GPUS[@]}"; do gpu_pid[$gpu_id]=""; done

is_running() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }

get_free_gpu() {
    while true; do
        for gpu_id in "${GPUS[@]}"; do
            if ! is_running "${gpu_pid[$gpu_id]}"; then
                echo "$gpu_id"
                return 0
            fi
        done
        sleep 2
    done
}

# ============================================================================
# 嵌套循环启动任务
# ============================================================================
echo "开始并行评估 LIBERO T-shape 任务集: ${SUITES[*]}"

for suite in "${SUITES[@]}"; do
    echo "------------------------------------------------"
    echo "正在启动 Suite: $suite"

    # 为每个 suite 创建独立日志子目录
    mkdir -p "${LOG_ROOT}/${suite}"

    for i in $(seq 0 $((NUM_TASKS-1))); do
        # 获取空闲显卡
        gpu_id=$(get_free_gpu)
        log_file="${LOG_ROOT}/${suite}/task_${i}.log"

        echo "[GPU $gpu_id] Suite: $suite | Task ID: $i"

        (
            # 隔离 GPU 资源
            if [ "$gpu_id" = "0" ]; then
                export CUDA_VISIBLE_DEVICES=0
            else
                export CUDA_VISIBLE_DEVICES=${gpu_id},0
            fi
            export MUJOCO_GL=egl
            export MUJOCO_EGL_DEVICE_ID=0

            # Build optional arguments
            EXTRA_ARGS=""
            if [ "$USE_MASK" = false ]; then
                EXTRA_ARGS="$EXTRA_ARGS --no_mask"
            fi
            if [ -n "$T5_EMBEDS" ]; then
                EXTRA_ARGS="$EXTRA_ARGS --t5_embeds $T5_EMBEDS"
            fi

            # 执行 T-shape 评估脚本
            echo "N" | python -u platforms/libero/eval_libero_T.py \
                --ckpt_path "$CKPT" \
                --task_suite "$suite" \
                --num_trials $NUM_TRIALS \
                --task_id $i \
                --mean_std_path "$MEAN_STD_PATH" \
                $EXTRA_ARGS > "$log_file" 2>&1
        ) &

        # 记录该显卡上运行的进程 PID
        gpu_pid[$gpu_id]=$!

        # 稍微延迟启动，防止初始化冲突
        sleep 2
    done
done

# 等待所有后台任务完成
wait

# ============================================================================
# 汇总评估结果 (Eval Summary)
# ============================================================================
SUMMARY_FILE="${LOG_ROOT}/summary.txt"

{
    echo ""
    echo "================================================"
    echo "           T-shape Evaluation Summary"
    echo "================================================"
    echo ""

    for suite in "${SUITES[@]}"; do
        echo "Suite: $suite"
        echo "------------------------------------------------"

        total_successes=0
        total_episodes=0
        task_count=0

        for i in $(seq 0 $((NUM_TASKS-1))); do
            log_file="${LOG_ROOT}/${suite}/task_${i}.log"

            if [ ! -f "$log_file" ]; then
                echo "  Task $i: [LOG NOT FOUND]"
                continue
            fi

            # Parse success/episodes from log: "Success: X/Y (Z%)"
            success_line=$(grep -oP 'Success: \K\d+/\d+' "$log_file" | tail -1)
            rate_line=$(grep -oP 'Success: \d+/\d+ \(\K[0-9.]+%' "$log_file" | tail -1)
            # Parse task description
            task_desc=$(grep -oP 'Task \d+: \K.*' "$log_file" | head -1)

            if [ -n "$success_line" ]; then
                successes=$(echo "$success_line" | cut -d'/' -f1)
                episodes=$(echo "$success_line" | cut -d'/' -f2)
                total_successes=$((total_successes + successes))
                total_episodes=$((total_episodes + episodes))
                task_count=$((task_count + 1))
                echo "  Task $i: ${successes}/${episodes} (${rate_line}) - ${task_desc}"
            else
                echo "  Task $i: [PARSE FAILED] - check $log_file"
            fi
        done

        echo "------------------------------------------------"
        if [ "$total_episodes" -gt 0 ]; then
            overall_rate=$(awk "BEGIN {printf \"%.2f\", $total_successes * 100 / $total_episodes}")
            echo "  $suite Overall: ${total_successes}/${total_episodes} (${overall_rate}%)"
        else
            echo "  $suite Overall: No results"
        fi
        echo ""
    done

    echo "================================================"
    echo "所有 T-shape 任务集测试完成！日志存储在: $LOG_ROOT"
} | tee "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"
