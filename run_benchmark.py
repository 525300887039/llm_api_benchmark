#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速运行基准测试的入口脚本。

不需要安装包，可以直接运行此脚本进行测试。
"""

from src.llm_api_benchmark.cli import legacy_cli

if __name__ == "__main__":
    legacy_cli()