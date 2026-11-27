# Intuition V0.1 实验材料

本文以一维距离和时间为例，检验经验能否留下一个可独立调用的数学对象。[返回论坛文章](../README.md)

## 从头运行完整实验

下载并解压[原版实验包](../Intuition_V0.1.zip)，进入解压后的 `intuition_v01` 目录。如果已经克隆本仓库，直接进入 `experiment` 目录也可以。需要 Python 3.10 或更高版本，只用标准库。

```sh
python run_all.py
```

Windows 也可以双击 `run_experiment.cmd`。一次启动会运行三轮搜索、注册、验证、删除本轮的两份输入副本，再用新进程计算，最后比较三轮结果并运行测试。已有的 `runs` 目录会保留，新结果默认写入下一个编号目录。当前目录的 `REPORT.md` 会更新为本次报告。

原始三轮实验的三项输出是 14、21、25。前两项相对于训练和验证均为新预测，第三项已经在验证阶段执行过，是删除输入后的复算。原版报告中“未见任务”一项应按这个口径阅读。[原版实验报告](REPORT.md) · [完整编排源码](run_all.py)

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

归档包保留了原始目录结构，包括删除输入后留下的空目录。Git 的普通克隆不会还原空目录。若直接核查归档现场，请用原版实验包；若从克隆的代码重新运行，程序会自行创建本轮目录。
