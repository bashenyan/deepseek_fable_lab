# DeepSeek-V4-Fable 二次训练与封装

这套工具把流程拆成四步，并额外提供两类安全任务入口：

1. 准备对话数据
2. 进行 LoRA/SFT 训练
3. 合并导出模型
4. 启动 OpenAI 兼容服务
5. 授权资产的防御审计
6. 自有 Telegram 消息分类

## 目录

- `prepare_data.py`：把原始 JSON/JSONL 清洗成统一格式
- `train_sft.py`：LoRA/SFT 训练入口
- `merge_lora.py`：合并适配器权重
- `serve.py`：启动本地推理服务
- `audit_api.py`：授权资产防御审计服务
- `telegram_classify.py`：Telegram 消息离线分类服务
- `examples/train_sample.jsonl`：示例训练数据
- `examples/audit_sample.txt`：防御审计样本
- `examples/telegram_sample.jsonl`：Telegram 分类样本

## 安装

```powershell
cd C:\Users\123\project\deepseek_fable_lab
pip install -r requirements.txt
pip install -e .
```

## 上线前准备

建议先配置访问密钥，避免把服务直接暴露给公网：

```powershell
$env:DEEPSEEK_FABLE_API_KEY="替换成你自己的密钥"
```

调用时统一带上：

```http
X-API-Key: 替换成你自己的密钥
```

如果要在 GPU 机器上运行，推理和训练都可以显式切换：

```powershell
# 推理
python -m deepseek_fable_lab.cli serve --model-path outputs\merged --device cuda --dtype fp16

# 训练
python -m deepseek_fable_lab.cli train --train-file data\train.jsonl --device cuda --dtype fp16
```

## 数据格式

推荐格式：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

也兼容：

```json
{"prompt":"...","completion":"..."}
```

## 两类安全任务的数据建议

### 防御审计

建议把样本组织成如下三段：

```json
{"messages":[
  {"role":"system","content":"你是一个授权环境中的防御安全审计助手。"},
  {"role":"user","content":"这里放 OpenAPI、响应头、网关配置或日志快照。"},
  {"role":"assistant","content":"这里输出发现项、风险等级、证据和修复建议。"}
]}
```

### Telegram 分类

建议把样本组织成如下三段：

```json
{"messages":[
  {"role":"system","content":"你是一个离线消息分类助手，只能基于自有数据进行分类。"},
  {"role":"user","content":"这里放 Telegram 消息正文。"},
  {"role":"assistant","content":"这里输出标签、置信度和简短理由。"}
]}
```

## 推荐训练流程

这是当前最稳的做法：

1. 先把原始样本整理成统一 JSONL
2. 做去重、过滤、自动切分验证集
3. 先单任务训练，再做混合训练
4. 每轮训练都跑一次评估
5. 把误判样本回灌到下一轮数据里

### 1. 整理数据

```powershell
python -m deepseek_fable_lab.prepare_data --input data\raw.jsonl --output data\train.jsonl --eval-output data\eval.jsonl --dedupe --eval-ratio 0.05
```

### 2. 先单任务训练

```powershell
# 防御审计
python -m deepseek_fable_lab.cli train --task audit --train-file data\train.jsonl --eval-file data\eval.jsonl --output-dir outputs\audit-lora --device auto --group-by-length

# Telegram 分类
python -m deepseek_fable_lab.cli train --task telegram --train-file data\train.jsonl --eval-file data\eval.jsonl --output-dir outputs\telegram-lora --device auto --group-by-length
```

### 3. 评估

```powershell
python -m deepseek_fable_lab.evaluate_outputs --input data\eval.jsonl --task telegram
python -m deepseek_fable_lab.evaluate_outputs --input data\eval.jsonl --task audit
```

### 4. 再混合训练

```powershell
python -m deepseek_fable_lab.cli train --task all --train-file data\train.jsonl --eval-file data\eval.jsonl --output-dir outputs\multi-task --device auto --group-by-length
```

## 准备数据

```powershell
python -m deepseek_fable_lab.cli prepare --input data\raw.jsonl --output data\train.jsonl
```

## 训练

```powershell
python -m deepseek_fable_lab.cli train `
  --train-file data\train.jsonl `
  --output-dir outputs\lora-sft `
  --base-model Chunjiang-Intelligence/DeepSeek-v4-Fable `
  --bf16
```

建议训练时优先使用：

- `--group-by-length`
- `--task audit` 或 `--task telegram` 先单训
- 有验证集时让脚本自动保留最佳模型
- GPU 上优先 `--dtype fp16` 或 `--dtype bf16`

## 合并导出

```powershell
python -m deepseek_fable_lab.cli merge `
  --base-model Chunjiang-Intelligence/DeepSeek-v4-Fable `
  --lora-dir outputs\lora-sft `
  --output-dir outputs\merged
```

## 启动服务

```powershell
python -m deepseek_fable_lab.cli serve `
  --model-path outputs\merged `
  --host 0.0.0.0 `
  --port 8000 `
  --device auto
```

调用示例：

```powershell
curl http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 替换成你自己的密钥" `
  -d '{"model":"local-model","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```

## Docker 上线

### 1. 配置密钥

```powershell
$env:DEEPSEEK_FABLE_API_KEY="替换成你自己的密钥"
```

### 2. 启动防御审计服务

```powershell
docker compose --profile audit up -d
```

### 3. 启动 Telegram 分类服务

```powershell
docker compose --profile telegram up -d
```

### 4. 启动模型推理服务

CPU 版：

```powershell
docker compose --profile model up -d
```

GPU 版：

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile model up -d
```

### 5. 常见环境变量

```powershell
$env:MODEL_PATH="outputs/merged"
$env:LORA_PATH=""
$env:DEVICE="auto"
$env:DTYPE="auto"
$env:TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
```

如果你已经有合并好的本地模型，把 `MODEL_PATH` 指向挂载目录即可。没有本地模型时，服务会使用 `BASE_MODEL` 直接加载基座。

## 防御审计服务

```powershell
python -m deepseek_fable_lab.cli audit --host 0.0.0.0 --port 8010
```

调用示例：

```powershell
curl http://127.0.0.1:8010/v1/audit `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 替换成你自己的密钥" `
  -d '{"asset_name":"gateway-a","content":"Access-Control-Allow-Origin: *"}'
```

直接给审计样本做一次本地规则初筛：

```powershell
python -m deepseek_fable_lab.audit_api --demo-file examples\audit_sample.txt
```

## Telegram 分类服务

```powershell
python -m deepseek_fable_lab.cli telegram --host 0.0.0.0 --port 8020
```

调用示例：

```powershell
curl http://127.0.0.1:8020/v1/classify `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 替换成你自己的密钥" `
  -d '{"text":"验证码 123456，5 分钟内有效"}'
```

直接给消息样本做一次本地规则分类：

```powershell
python -m deepseek_fable_lab.telegram_classify --demo-file examples\telegram_sample.jsonl
```

## 说明

- 训练脚本默认采用通用兜底模板；如果你的基座模型有专属 chat template，可以在 `serve.py` 和 `template.py` 里替换。
- 当前实现是 LoRA/SFT 基线，适合先跑通完整链路。
- 如果后续你补充高质量偏好数据，可以再扩展 DPO，但不建议一开始就上。
- `audit_api.py` 和 `telegram_classify.py` 目前提供的是可直接运行的安全基线，后续你可以把它们替换成你训练后的模型推理。
- 如果当前机器没有 NVIDIA GPU，服务会自动退回 CPU；想切到 GPU 需要驱动、CUDA 和对应的 PyTorch GPU 轮子。
- GPU 版 PyTorch 请按官方安装矩阵选择对应的 CUDA 轮子：<https://pytorch.org/get-started/locally/>
- 训练效果最关键的是样本质量、标签一致性和误判回灌，不是单纯延长训练时间。
