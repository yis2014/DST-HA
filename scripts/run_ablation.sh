#!/bin/bash

# DST_HA 消融实验运行脚本
# 用法: ./run_ablation.sh [experiment_name]
# 如果不指定实验名称，将运行所有实验

set -e  # 遇到错误立即退出

# 项目根目录
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# 配置
CONFIG_DIR="$BASE_DIR/config/ablation"
RESULTS_DIR="$BASE_DIR/results/ablation"
PYTHON_CMD="python"

# 确保结果目录存在
mkdir -p "$RESULTS_DIR"

# 实验列表
declare -A EXPERIMENTS=(
    ["full"]="hz_full.yaml"
    ["wo_se_te_ht"]="hz_wo_se_te_ht.yaml"
    ["wo_se_te"]="hz_wo_se_te.yaml"
    ["wo_se_ht"]="hz_wo_se_ht.yaml"
    ["wo_te_ht"]="hz_wo_te_ht.yaml"
    ["wo_se"]="hz_wo_se.yaml"
    ["wo_te"]="hz_wo_te.yaml"
    ["wo_ht"]="hz_wo_ht.yaml"
)

# 运行单个实验的函数
run_experiment() {
    local exp_name="$1"
    local config_file="$2"
    
    echo "=========================================="
    echo "运行消融实验: $exp_name"
    echo "配置文件: $config_file"
    echo "=========================================="
    
    # 构建日志目录
    local log_dir="$RESULTS_DIR/ablation_$exp_name"
    mkdir -p "$log_dir"
    
    # 运行实验
    $PYTHON_CMD main.py \
        --config "$CONFIG_DIR/$config_file" \
        --log_dir "$log_dir" \
        --seed 42 \
        2>&1 | tee "$log_dir/run.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ 实验 $exp_name 完成"
    else
        echo "✗ 实验 $exp_name 失败"
        return 1
    fi
}

# 主逻辑
if [ $# -eq 0 ]; then
    # 运行所有实验
    echo "开始运行所有 DST_HA 消融实验..."
    echo "共 ${#EXPERIMENTS[@]} 组实验"
    
    start_time=$(date +%s)
    
    for exp_name in "${!EXPERIMENTS[@]}"; do
        config_file="${EXPERIMENTS[$exp_name]}"
        run_experiment "$exp_name" "$config_file"
    done
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "所有实验完成! 总耗时: $((duration / 3600))小时 $(((duration % 3600) / 60))分钟"
    
else
    # 运行指定实验
    exp_name="$1"
    if [[ -n "${EXPERIMENTS[$exp_name]}" ]]; then
        config_file="${EXPERIMENTS[$exp_name]}"
        run_experiment "$exp_name" "$config_file"
    else
        echo "错误: 未找到实验 '$exp_name'"
        echo "可用实验: ${!EXPERIMENTS[*]}"
        exit 1
    fi
fi

echo "结果保存在: $RESULTS_DIR" 