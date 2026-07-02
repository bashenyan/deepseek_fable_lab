"""统一配置常量。

这里保留最少的默认值，避免把训练超参写死在脚本里。
"""

DEFAULT_BASE_MODEL = "Chunjiang-Intelligence/DeepSeek-v4-Fable"
DEFAULT_OUTPUT_DIR = "outputs/lora-sft"
DEFAULT_MERGED_DIR = "outputs/merged"
DEFAULT_MAX_LENGTH = 8192
DEFAULT_SEED = 42

# 统一的角色标记，作为兜底模板。
ROLE_PREFIX = {
    "system": "### 系统\n",
    "user": "### 用户\n",
    "assistant": "### 助手\n",
}
ROLE_SUFFIX = "\n"
TURN_SEPARATOR = "\n"
