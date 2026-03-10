# Open-AgentRL 项目深度解析：Agentic RL 的实现架构

> 项目地址：https://github.com/Gen-Verse/Open-AgentRL

该项目包含两个核心工作：**DemyAgent**（Agentic Reasoning 的 RL 训练）和 **RLAnything**（Policy/Reward/Environment 三者闭环协同优化）。以下从源码层面逐一拆解。

---

## 一、项目整体架构

|  | DemyAgent | RLAnything |
|---|---|---|
| **核心思想** | 真实端到端 trajectory + 探索友好技术 + 审慎推理 | Policy/Reward/Environment 三者闭环动态优化 |
| **训练框架** | 基于 veRL (Volcengine RL)，使用 GRPO-TCR | 自建训练循环，Accelerate + DeepSpeed |
| **工具调用** | code_interpreter（SandboxFusion 沙箱） | 多种环境：AlfWorld 文本游戏、OSWorld GUI 操控、代码执行 |
| **数据** | 3K SFT + 30K RL（开源 HuggingFace） | CodeContests + AlfWorld + OSWorld |

---

## 二、DemyAgent 的 Agentic RL 实现详解

### 2.1 数据格式

DemyAgent 使用 **Parquet 格式**存储数据，核心字段定义在 `recipe/demystify/reward.py` 的 `CustomRLHFDataset` 中：

```python
# 训练数据格式（Parquet 行）
{
    "data_source": "AIME2024" | "LiveCodeBench_v6" | "math_dapo" | ...,
    "prompt": [{"role": "user", "content": "..."}],  # 单轮 chat 格式
    "ability": "MATH" | "code" | "physics" | ...,
    "reward_model": {"ground_truth": "答案字符串"},
    "agent_name": "tool_agent",  # 标记为 agent 模式
    "extra_info": {...}  # 代码题的额外评测信息
}
```

**关键设计**：prompt 中自带 agent 指令模板，不同数据源使用不同模板：

- **数学题**："Analyze and solve the following math problem step by step..." + 工具提示 + `\boxed{}` 答案格式要求
- **代码题**："Read the inputs from stdin..." + 工具调用提示 + ` ```python``` ` 代码格式要求
- **GPQA（科学题）**：按学科领域定制 prompt + 工具提示 + 选项格式要求

### 2.2 工具定义

DemyAgent 只使用**一个核心工具**：`code_interpreter`，定义在 `sandbox_fusion_tool_config.yaml`：

```yaml
tools:
  - class_name: "recipe.demystify.reward.CustomSandboxFusionTool"
    config:
      sandbox_fusion_url: "https://<your-sandbox-api>/run_code"
      num_workers: 128        # 并发执行器数量
      rate_limit: 128         # 速率限制
      default_timeout: 30     # 执行超时（秒）
      default_language: "python"
      memory_limit_mb: 2048
    tool_schema:
      type: "function"
      function:
        name: "code_interpreter"
        description: "A tool for executing code."
        parameters:
          type: "object"
          properties:
            code:
              type: "string"
              description: "The code to execute."
          required: ["code"]
```

**CustomSandboxFusionTool 的特殊处理**（源码 `reward.py`）：

- 自动提取 ` ```python ... ``` ` 代码块
- 如果最后一行不是 `print()`，自动包裹为 `print(last_line)`
- 通过 SandboxFusion API 远程执行代码，返回 stdout 结果

### 2.3 多步调用的实现机制

多步工具调用的核心在 veRL 框架的 **multi_turn rollout** 机制中：

```bash
# 训练脚本 grpo_tcr_qwen3_4b.sh 中的关键配置：
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.max_user_turns=16       # 最多 16 轮用户交互
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16   # 最多 16 轮助手回复
actor_rollout_ref.rollout.multi_turn.tool_config_path=...     # 工具配置
actor_rollout_ref.rollout.multi_turn.format=hermes            # 工具调用格式
```

**Rollout 流程**（由 veRL 内部实现）：

1. 模型收到 user prompt
2. 模型生成 response，可能包含工具调用（Hermes function calling 格式）
3. veRL 检测到工具调用 → 执行 SandboxFusion → 将结果拼接为新的 tool message
4. 模型收到工具结果，继续生成（可能再次调用工具或给出最终答案）
5. 重复直到：模型不再调用工具 / 达到 max_turns=16 / 超出 max_response_length=20480 tokens

### 2.4 Reward 设计（GRPO-TCR）

Reward 函数定义在 `recipe/demystify/reward.py` 的 `compute_score`：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    # 根据数据类型选择评判方式
    if 'code' in data_source:
        result = code_math.compute_score(solution_str, ground_truth)  # 代码执行验证
    else:
        result = math_dapo.compute_score(...)  # 数学答案匹配（strict_box_verify）

    # TCR: Tool Call Reward（工具调用奖励/惩罚）
    num_turns = int(extra_info.get("num_turns", 0))
    if result["score"] < 0:  # 答案错误时
        tool_call_reward = (num_turns - 2) / 2 * 0.1  # 调用越多工具，惩罚越轻
        result["score"] = float(min(-0.6, result["score"] + tool_call_reward))
    return result
```

**TCR（Tool Call Reward）的核心思想**：

- 答案**正确**：reward = 1.0（不管调用了多少次工具）
- 答案**错误**：基础 reward = -1.0，但**如果模型尝试了工具调用**（num_turns > 2），惩罚会减轻
- 公式：`score = min(-0.6, -1.0 + (num_turns - 2) / 2 * 0.1)`
- **设计意图**：鼓励模型在不确定时尝试使用工具，而不是直接猜答案

### 2.5 RL 算法配置（GRPO + DAPO 增强）

```bash
# 来自 grpo_tcr_qwen3_4b.sh
adv_estimator=grpo                  # Group Relative Policy Optimization
use_kl_in_reward=False              # 移除 KL 散度约束 ✓
kl_coef=0.0
clip_ratio_low=0.2                  # 不对称裁剪 ✓
clip_ratio_high=0.28                # 高端裁剪更宽松（鼓励探索）
loss_agg_mode="token-mean"          # token 级别平均 ✓
reward_manager=dapo                 # DAPO 风格 reward 管理
enable_overlong_buffer=True         # 超长 buffer ✓
overlong_buffer_len=4096            # 4K token buffer
overlong_penalty_factor=1.0         # 超长惩罚因子
n_resp_per_prompt=16                # 每个 prompt 采样 16 条 trajectory
train_batch_size=64                 # batch size
actor_lr=1e-6                       # 学习率
```

**关键训练技巧**：

- **移除 KL 约束**：让模型更自由地探索工具调用策略
- **不对称 clip**：`clip_ratio_low=0.2, clip_ratio_high=0.28`，上限更宽松，鼓励探索高 reward 的策略
- **Overlong Buffer**：给超出 max_response_length 的 response 额外 4K buffer，避免截断导致的格式错误被严惩
- **Group Size = 16**：每个 prompt 采样 16 条不同 trajectory，组内归一化计算 advantage

---

## 三、RLAnything 的闭环优化实现

### 3.1 三模型协同架构

RLAnything 的核心创新是同时优化三个模型：

```yaml
# 配置文件 configs/coding_rl.yaml
model:
  policy_model: ...       # 策略模型（写代码/解题）
  reward_model: ...       # 奖励模型（生成单元测试来评估代码）
  environment_model: ...  # 环境模型（动态调整题目难度）
```

### 3.2 训练循环（coding_rl.py 主循环）

```python
while epoch <= total_step:
    # Step 1: Policy Rollout - 策略模型生成代码
    policy_sample(worker_hosts, epoch, cfg, "train")

    # Step 2: Reward Rollout - 奖励模型生成合成单元测试
    reward_sample(worker_hosts, epoch, cfg, "train")

    # Step 3: Code Execution - 在沙箱中执行代码，得到 correctness 矩阵
    execute(worker_hosts, epoch, cfg, "train")

    # Step 4: Environment Rollout - 环境模型根据准确率动态调整题目
    env_sample(worker_hosts, epoch, cfg, "train")

    # Step 5: Aggregate - 汇总所有节点数据
    aggregate(epoch, cfg, "train")

    # Step 6: Compute Rewards - 计算最终 reward
    reward(epoch, cfg, "train")

    # Step 7: Train Policy Model
    train(worker_hosts, epoch, cfg, target="policy")

    # Step 8: Train Reward Model
    train(worker_hosts, epoch, cfg, target="reward")
```

### 3.3 各步骤详解

#### Step 1: Policy Rollout（llm_policy_rollout.py）

- 使用 vLLM 做大批量推理，TP=4
- 每个题目采样 `k_sample=32` 条代码
- prompt 格式：`"You need to think first then write python script..."`
- 自动提取 ` ```python ... ``` ` 代码块
- **环境协同进化**：从 state 中加载当前活跃题目集（可能已被 environment model 修改过）+ pending 中的临时题目

输出数据结构：

```python
{
    "source": "dataset" | "temp",    # dataset=稳定题目, temp=环境模型生成的待验证题目
    "slot_id": 42,                   # 题目槽位 ID
    "question": "题目描述...",
    "test_input": ["1 2\n", ...],    # 标准输入
    "test_output": ["3\n", ...],     # 期望输出
    "prompt": "完整 prompt...",
    "full_output": ["模型输出1", ...],       # k_sample 条
    "extracted_output": ["提取的代码1", ...], # k_sample 条
    "response_length": [512, ...]            # token 长度
}
```

#### Step 2: Reward Rollout（llm_reward_rollout.py）

- Reward 模型的角色是 **"单元测试生成器"**
- 对每个题目，生成 `reward_k_sample=16` 组合成单元测试
- 使用 vLLM 推理，TP=8

Reward 模型的 prompt 模板：

```
System: "You are a rigorous unit-test designer for coding problems."
User: "You need to provide a new test example... A good test example should be
completely accurate and possessing enough discriminative power to distinguish
correct code from incorrect code."
Finally output: **Test Input:** ```...``` **Test Output:** ```...```
```

输出会解析出 `syn_input` 和 `syn_output`（合成单元测试）。

#### Step 3: Code Execution（execute.py）

- 将 policy 生成的所有代码分别在 GT 单元测试和合成单元测试上执行
- 生成 `correctness` 矩阵：`[m_code × m_gt_test]`
- 和 `syn_correctness`：`[m_code × m_syn_test]`
- 每个位置是 True/False，表示该代码是否通过该测试

#### Step 4: Environment Rollout（llm_env_rollout.py）

**RLAnything 的核心创新**——动态环境调整：

- 如果 policy 在某题上准确率 > 80%（太简单）→ 生成更 harder 的变体
- 如果准确率 < 20%（太难）→ 生成更 easier 的变体
- 使用 vLLM 驱动环境模型，TP=8

环境模型生成新题时，会参考：

- 原始根题目（ANCHOR，防止 topic 漂移）
- 当前活跃题目
- Policy 在每个单元测试上的通过率
- 合成单元测试的通过率

生成结果经过**验证-接受**流程：

```python
# 接受条件（llm_env_rollout.py）
if goal == "harder":
    ok = (now_acc < prev_acc) and (now_acc >= 0.2)  # 确实变难了，但不至于完全做不出
elif goal == "easier":
    ok = (now_acc > prev_acc) and (now_acc <= 0.8)  # 确实变简单了，但不至于太容易
```

#### Step 6: Reward Computation（llm_rl_reward.py）

同时为 policy 和 reward model 计算训练信号：

**Policy Reward**：

```python
# 对每条生成的代码
raw_reward = (通过的 GT 测试数) / (GT 测试总数)  # pass ratio [0, 1]
rewards = z_score_normalize(raw_rewards)          # 组内归一化
# 过滤条件：只保留准确率在 [0.2, 0.8] 的题目（太简单/太难的不训练）
```

**Reward Model Reward**（更复杂——衡量合成测试的"区分力"）：

```python
# GT 代码 = 通过所有 GT 测试的代码
# GT 单元测试 = 通过所有 GT 代码的合成测试
对每个合成单元测试 k：
    if 是 GT 单元测试:
        reward = 能区分出多少条非 GT 代码（fail 的非 GT 代码数）
    else:
        reward = -(错误通过的非 GT 代码数)
# 如果没有非 GT 代码：reward = +1（GT 测试）或 -1（非 GT 测试）
rewards = z_score_normalize(raw_rewards)
```

#### Step 7-8: Training（alfworld_train.py）

- 使用 **Accelerate + DeepSpeed Zero3** 进行分布式训练
- GRPO 算法：group 内归一化 advantage，无需 value model
- **Policy 和 Reward Model 分别训练**，共享同一训练循环
- 支持 KL 惩罚（`beta=0.01`）和梯度裁剪（`max_grad_norm=1.0`）

---

## 四、AlfWorld（文本游戏）的多步交互实现

AlfWorld 场景展示了更典型的**多步工具调用**模式：

### 交互格式

```
# 模型在每一步生成：
<thought>我需要找到苹果...</thought>
<answer>go to countertop 1</answer>

# 环境返回观察：
"On the countertop 1, you see a apple 1."

# 模型继续：
<thought>找到了苹果，需要拿起来...</thought>
<answer>take apple 1 from countertop 1</answer>
```

### 奖励设计（alfworld_rl_reward.py）

```python
# 结合 outcome reward + step-wise reward
for each rollout j:
    outcome_reward = 1.0 if success else 0.0
    for each step k:
        step_scores = reward_model 对该步的评分（多次采样取平均）
        policy_reward[j][k] = outcome_reward + mean(step_scores)

# 跨 rollout 按步骤归一化
policy_rewards = normalize_process_list(policy_r_list)  # 每步独立 z-score
```

**Step-wise Reward 的关键**：reward model 对每一步的"行动质量"打分，与最终成功/失败结合，提供更细粒度的学习信号。

---

## 五、工具间的逻辑关系总结

### DemyAgent 的工具链

```
用户问题 → 模型思考 → [可选] 调用 code_interpreter → 获得执行结果 → 继续思考 → [可选] 再次调用 → ... → 最终答案
                              ↑
                    SandboxFusion 沙箱（支持 Python，超时 30s，内存限制 2GB）
```

**工具关系**：**单工具多次调用**。`code_interpreter` 是唯一工具，模型学会在不同阶段（验证计算、测试代码、调试）多次调用它。

### RLAnything 的三模型协同

```
┌─────────────┐    生成代码    ┌──────────────┐
│ Policy Model │ ───────────→ │   沙箱执行    │ → correctness 矩阵
└──────┬──────┘              └──────────────┘
       │      ↑                      │
       │  训练信号 (GRPO)       执行合成测试
       │      │                      │
┌──────┴──────┐   生成合成测试  ┌──────┴──────┐
│ Reward Model │ ───────────→ │   沙箱执行    │ → syn_correctness
└─────────────┘              └──────────────┘
       ↑                            │
       │       训练信号 (区分力)      │
       │                            │
┌──────┴──────────┐  调整题目难度
│ Environment Model│ ───────────→ 更新题目集（harder / easier）
└─────────────────┘
       ↑
       │  根据 policy 准确率反馈
```

---

## 六、如何训练多步工具调用能力（完整流程）

### Phase 1: SFT 冷启动

1. 下载 3K Agentic SFT 数据
2. 基座模型：Qwen2.5-7B-Instruct 或 Qwen3-4B-Instruct-2507
3. 运行 `bash recipe/demystify/qwen3_4b_sft.sh`
4. 合并模型：`python3 -m verl.model_merger merge ...`

### Phase 2: 配置工具环境

1. 部署 **SandboxFusion**（本地 Docker 或火山引擎云服务）
2. 获取 API endpoint，配置到 `sandbox_fusion_tool_config.yaml`
3. 同时配置 `verl/utils/reward_score/livecodebench/code_math.py` 中的 `check_correctness`

### Phase 3: Agentic RL 训练

1. 下载 30K Agentic RL 数据
2. 配置 `grpo_tcr_qwen3_4b.sh`（模型路径、数据路径、checkpoint 目录）
3. 运行 `bash recipe/demystify/grpo_tcr_qwen3_4b.sh`
4. 在 wandb 中观察训练动态和评估结果

### 关键超参数建议

| 参数 | DemyAgent 值 | 说明 |
|---|---|---|
| max_turns | 16 | 最大工具调用轮数 |
| n_resp_per_prompt | 16 | GRPO group size |
| max_response_length | 20480 | 最大 response token 数 |
| max_prompt_length | 2560 | 最大 prompt token 数 |
| actor_lr | 1e-6 | 学习率 |
| clip_ratio_low/high | 0.2 / 0.28 | 不对称 clip |
| overlong_buffer_len | 4096 | 超长 buffer |
| train_batch_size | 64 | 批大小 |
| 硬件 | 8×A100 | 单节点 |

---

## 七、关键 Insights 和最佳实践

1. **真实 trajectory 胜过合成数据**：端到端在环境中采集的多步交互数据，效果远好于 GPT-4 合成的 trajectory
2. **审慎推理（Deliberative Reasoning）优于频繁调用**：模型应该在思考后选择性地调用工具，而不是每步都调用
3. **TCR 鼓励工具探索**：对答案错误但尝试了工具的 trajectory 给予更轻的惩罚，引导模型学会使用工具
4. **移除 KL 约束 + 不对称 clip**：给模型更大的探索空间，这是 DAPO 风格的探索增强
5. **4B 模型可以打败 32B**：通过正确的 SFT + RL 配方，DemyAgent-4B 在 AIME2025 上超越了 ReTool-32B 和 DeepSeek-R1-Zero
6. **环境协同进化**（RLAnything）：动态调整训练题目难度，保持 policy 始终处于"最近发展区"（准确率 20%-80%）

---

## 八、参考论文

- [DemyAgent: Demystifying Reinforcement Learning in Agentic Reasoning](https://github.com/Gen-Verse/Open-AgentRL)
- [RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System](https://github.com/Gen-Verse/Open-AgentRL)
- 框架：[veRL (Volcengine RL)](https://github.com/volcengine/verl)
- 工具沙箱：[SandboxFusion](https://github.com/volcengine/SandboxFusion)

---

> 以上分析基于 Open-AgentRL 项目源码的逐文件阅读。如需更深入的某个模块细节，可以进一步探讨。
