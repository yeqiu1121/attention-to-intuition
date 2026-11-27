# Intuition V0.1 实验材料

本文以一维距离和时间为例，检验经验能否留下一个可独立调用的数学对象。[返回论坛文章](../README.md)

## 直接复算

把仓库下载或克隆到本地，进入 `experiment/runs/run_01/runtime` 目录。使用 Python 3.10 或以上版本运行：

```sh
python -I -S runtime.py --registry registry.json --tasks tasks.json --out replay.json
```

`replay.json` 将记录三个距离 14、21、25，以及运行程序实际访问的文件。前两项是新预测，第三项已在验证阶段算过。把任务单另存为 `my-tasks.json`，将 v3 的时间改为 `"11"`，再把命令中的任务单和输出文件名换成自己的，就可以给它出一道新题。

## 复核依据

- [训练与验证数据](data/observations.json)
- [搜索器](intuition/discovery.py)和[精确代数检查器](intuition/proof.py)
- [候选表达式逐条记录](runs/run_01/audit/candidates.jsonl)和[并列结果](runs/run_01/audit/discovery.json)
- [验证阶段记录](runs/run_01/audit/validation.json)与[删除前后状态](runs/run_01/audit/deletion.json)
- [三次运行的一致性记录](runs/reproducibility.json)与[原实验的自动化测试记录](runs/test_results.txt)

当前公开版本包含独立执行示例、源码模块和已完成的运行记录。完整编排脚本与长篇实验报告尚未随本次发布。
