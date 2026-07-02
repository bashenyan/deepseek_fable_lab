"""统一命令行入口。"""

from __future__ import annotations

import argparse
import runpy
import sys


COMMANDS = {
    "prepare": "deepseek_fable_lab.prepare_data",
    "train": "deepseek_fable_lab.train_sft",
    "merge": "deepseek_fable_lab.merge_lora",
    "serve": "deepseek_fable_lab.serve",
    "audit": "deepseek_fable_lab.audit_api",
    "telegram": "deepseek_fable_lab.telegram_classify",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Fable 工具入口")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    sys.argv = [COMMANDS[parsed.command], *parsed.args]
    runpy.run_module(COMMANDS[parsed.command], run_name="__main__")


if __name__ == "__main__":
    main()
