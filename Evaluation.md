# BookWorld 评估指南 (Evaluation Guide)

## 1. 评估原理
评估过程利用大语言模型（LLM）作为裁判，从多个维度（如角色一致性、情节逻辑、语言表达等）对 BookWorld 生成的内容进行评分，并与基准模型（Naive 生成）进行对比。

## 2. 如何启动评估
你可以通过发送 HTTP POST 请求到 `server.py` 提供的 API 接口来触发评估。

### API 接口
- **URL**: `http://localhost:8000/api/evaluate`
- **Method**: `POST`
- **Payload (JSON)**:
  ```json
  {
    "eval_llm": "gpt-4o"  // 可选，指定用于评分的 LLM 模型名。默认为 config.json 中的 world_llm_name
  }
  ```

### 示例代码 (Python)
```python
import requests

url = "http://localhost:8000/api/evaluate"
data = {"eval_llm": "gpt-4o"}
response = requests.post(url, json=data)
print(response.json())
```

## 3. 评估最佳实践 (Best Practices)
为了获得最准确和有意义的评估结果，建议遵循以下最佳实践：

- **积累足够的上下文**：建议在模拟运行至少 **10 轮 (Rounds)** 以上后再启动评估。评估逻辑会跳过初期的角色介绍，如果历史记录太短，评估结果可能缺乏参考价值。
- **选用高性能评估模型**：评估（尤其是 `naive_winner` 对比）需要极强的逻辑推理能力。强烈建议使用 **`gpt-4o`** 或 **`claude-3.5-sonnet`** 作为 `eval_llm`，避免使用轻量级模型（如 mini 版）进行评分。
- **保持配置一致性**：评估过程中会重新加载 `preset` 和角色配置，请确保在模拟运行结束到评估启动期间，没有手动修改相关的 JSON 配置文件。
- **单次运行原则**：评估任务是资源密集型的。在当前版本中，建议**不要同时并发**触发多个评估任务，以免造成 LLM API 速率限制（Rate Limit）或服务器资源争抢。
- **语言匹配**：评估器会自动根据 `config.json` 中的 `language` 字段调整字数统计逻辑（中文按字符，英文按单词），请确保配置准确。

## 4. 评估结果与临时文件
评估过程中产生的所有数据和临时文件都将存储在 `./eval_saves/` 目录下。

### 存储路径结构
评估结果按以下层级组织：
`./eval_saves/{mode}/{role_model}/{subexperiment_name}/{source}/{experiment_name}/{timestamp}/`

- **mode**: 模拟模式 (`free` 或 `script`)。
- **role_model**: 角色所使用的模型。
- **subexperiment_name**: 子实验名称（默认为 `full`）。
- **source**: 剧本/世界观来源。
- **timestamp**: 评估开始的时间戳。

### 关键文件说明
- **`eval_agent.json`**: **最重要的文件**。包含了评估得分（`scoring`）、胜出者对比（`winner`）以及用于对比的基准文本。
- **`meta_info.json`**: 记录了评估时的模拟轮数等元信息。
- **`server_info.json`**: 模拟运行时的服务器内部状态快照。

## 4. 临时文件存储位置总结
在 Evaluation 阶段，临时文件和最终结果统一存储在项目根目录下的 **`./eval_saves/`** 目录中。请确保该目录具有写入权限。

---
*注：评估过程可能消耗较多 Token，建议使用性能较强的模型（如 GPT-4o 或 Claude-3.5-Sonnet）作为 eval_llm 以获得更准确的评分。*
