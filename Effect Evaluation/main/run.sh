#!/bin/bash

# ==========================================
# PPFL-TEE 攻防实验矩阵 (安全并发版)
# ==========================================

# ⚠️ 核心设置：最大并发任务数
MAX_JOBS=4

# 1. 定义核心变量矩阵 (命名严格对应 config.yaml)
MODELS=("lenet5" "resnet20" "resnet18")
POISON_RATIOS=(0.1 0.2 0.3 0.4)
DEFENSES=("none" "layers_proj_detect" "krum" "clustering" "median")

# 攻击类型划分
TARGETED_ATTACKS=("backdoor" "label_flip")
UNTARGETED_ATTACKS=("random_poison" "gradient_amplify")

ROUNDS=50

# 存放所有待执行命令的临时文件
CMD_FILE="tasks_queue.txt"
> $CMD_FILE # 清空旧文件

echo "🚀 正在生成 PPFL-TEE 攻防实验任务队列..."

# ---------------------------------------------------------
# 阶段一：纯净联邦训练 (上限基线: none Attack)
# ---------------------------------------------------------
for model in "${MODELS[@]}"; do
    echo "python main.py --model $model --attack none --defense none --rounds $ROUNDS > main/results/log_${model}_noneAttack.txt 2>&1" >> $CMD_FILE
done

# ---------------------------------------------------------
# 阶段二：有目标攻击测试 (Targeted Attacks: backdoor, label_flip)
# ---------------------------------------------------------
for attack in "${TARGETED_ATTACKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ratio in "${POISON_RATIOS[@]}"; do
            for defense in "${DEFENSES[@]}"; do
                echo "python main.py --model $model --attack $attack --poison_ratio $ratio --defense $defense --rounds $ROUNDS > main/results/log_${model}_${attack}_p${ratio}_${defense}.txt 2>&1" >> $CMD_FILE
            done
        done
    done
done

# ---------------------------------------------------------
# 阶段三：无目标攻击测试 (Untargeted Attacks: random_poison, gradient_amplify)
# ---------------------------------------------------------
for attack in "${UNTARGETED_ATTACKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ratio in "${POISON_RATIOS[@]}"; do
            for defense in "${DEFENSES[@]}"; do
                echo "python main.py --model $model --attack $attack --poison_ratio $ratio --defense $defense --rounds $ROUNDS > main/results/log_${model}_${attack}_p${ratio}_${defense}.txt 2>&1" >> $CMD_FILE
            done
        done
    done
done

TOTAL_TASKS=$(wc -l < $CMD_FILE)
echo "✅ 成功生成 $TOTAL_TASKS 个实验任务！"
echo "🔥 开始并行执行，最大并发进程数限制为: $MAX_JOBS"
echo "=========================================="

# 核心并行执行逻辑：使用 xargs 维持最大并发数为 MAX_JOBS
cat $CMD_FILE | xargs -P $MAX_JOBS -I {} bash -c '{}'

echo "🎉 所有并行实验执行完毕！"