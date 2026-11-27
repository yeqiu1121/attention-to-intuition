"""Observer/harness, intentionally outside the learner and runtime boundaries.

Fixtures and expected answers exist only in this harness/data and tests. The
learner receives a training file; the fresh runtime receives a registry/tasks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(arguments, cwd=ROOT):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Child process failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


def run_once(directory):
    inputs, audit, runtime = (directory / part for part in ("inputs", "audit", "runtime"))
    for path in (inputs, audit, runtime):
        path.mkdir(parents=True, exist_ok=True)
    events = []

    def event(stage, **details):
        events.append({"seq": len(events) + 1, "stage": stage, **details})
        with (audit / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(events[-1], ensure_ascii=False, sort_keys=True) + "\n")

    fixtures = read(ROOT / "data" / "observations.json")
    training_path, validation_path = inputs / "training.json", inputs / "validation.json"
    save(training_path, fixtures["training"])
    # No validation file exists when the learner runs.
    event("initial_capabilities", primitives=["Distance", "Duration"], operations=["+", "-", "*", "/"],
          variables=["d", "t"], max_ast_nodes=5, literal_leaves=False,
          equivalence="kernel of a total function", quotient="generic X/~",
          type_registry="generic, empty", validation_available=False, final_tasks_available=False)
    manifest = {"grammar": {"leaves": ["d", "t"], "binary_ops": ["+", "-", "*", "/"],
                             "max_nodes": 5, "constant_leaves": [], "deduplication": "none"},
                "decoder_grammar": {"leaves": ["z", "t"], "same_binary_ops_and_bound": True},
                "initial_types": ["Distance", "Duration"], "registered_types_before": [],
                "prediction_target": "reconstruct distance from coordinate and duration",
                "supplied_domain": "exact rationals d>=0,t>0; future duration>0",
                "source_hashes": {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted([
                    *(ROOT / "intuition").glob("*.py"), *(ROOT / "tests").glob("*.py"), ROOT / "run_all.py"])},
                "fixture_sha256": sha(ROOT / "data" / "observations.json")}
    save(audit / "initial_manifest.json", manifest)
    registry_path = runtime / "registry.json"
    invoke(["-S", "-m", "intuition.pipeline", "train", "--input", training_path,
            "--registry", registry_path, "--audit", audit])
    event("training_process_exited", registry_sha256=sha(registry_path))
    discovery = read(audit / "discovery.json")
    event("structure_selected_and_registered", selected=discovery["selected"],
          groups=discovery["groups"], candidate_count=discovery["candidate_count"])
    # This read/creation happens only after the registry is finalized.
    save(validation_path, fixtures["validation"])
    frozen_hash = sha(registry_path)
    invoke(["-S", "-m", "intuition.pipeline", "validate", "--input", validation_path,
            "--registry", registry_path, "--audit", audit])
    if sha(registry_path) != frozen_hash:
        raise AssertionError("Validation changed the registry")
    event("validation_process_exited", passed=read(audit / "validation.json")["passed"], registry_unchanged=True)
    before = [{"file": p.name, "exists": p.is_file(), "sha256": sha(p)} for p in (training_path, validation_path)]
    state_before = {"training_observations_in_input_files": len(fixtures["training"]),
                    "validation_observations_in_input_files": len(fixtures["validation"]),
                    "registered_type_names": [t["name"] for t in read(registry_path)["types"]],
                    "registered_instance_count": len(read(registry_path)["instances"]),
                    "runtime_files": sorted(p.name for p in runtime.iterdir())}
    # Real unlink, after both processes have exited. No recursive deletion.
    training_path.unlink()
    validation_path.unlink()
    after = [{"file": p.name, "exists": p.exists()} for p in (training_path, validation_path)]
    if any(item["exists"] for item in after) or any(inputs.iterdir()):
        raise AssertionError("Raw run inputs were not removed")
    deletion = {"scope": "run input files and exited learner/validator process memory",
                "before": before, "after": after, "input_directory_empty": True,
                "state_before": state_before,
                "state_after": {**state_before, "training_observations_in_input_files": 0,
                                "validation_observations_in_input_files": 0},
                "training_process_exited": True, "validation_process_exited": True,
                "execution_file_allowlist_after_task_injection": ["registry.json", "runtime.py", "tasks.json"],
                "audit_archive_keeps_observation_values": True,
                "reproduction_fixture_kept_outside_runtime": "data/observations.json",
                "claim": "execution independent of observations; not global secure erasure"}
    save(audit / "deletion.json", deletion)
    event("raw_input_files_deleted", before=before, after=after)
    # Inject tasks only after deletion; worker receives no expected answers.
    save(runtime / "tasks.json", read(ROOT / "data" / "tasks.json"))
    shutil.copyfile(ROOT / "intuition" / "runtime.py", runtime / "runtime.py")
    save(audit / "execution_manifest.json", {"runtime_files_before_execution": sorted(p.name for p in runtime.iterdir()),
         "training_input_exists": training_path.exists(), "validation_input_exists": validation_path.exists(),
         "registry_sha256": sha(registry_path), "task_file_has_expected_answers": False,
         "raw_observation_files_in_runtime": []})
    invoke(["-I", "-S", runtime / "runtime.py", "--registry", "registry.json",
            "--tasks", "tasks.json", "--out", "result.json"], cwd=runtime)
    result = read(runtime / "result.json")
    # Hidden acceptance expectations belong to the observer, never the search
    # process or generated registry/worker/task inputs.
    observed = [item["value"] for item in result["results"]]
    if observed != ["14", "21", "25"]:
        raise AssertionError(f"Unseen-task acceptance failed: {observed}")
    if any(not item["allowed"] for item in result["file_accesses"]):
        raise AssertionError("Unexpected worker file access")
    event("fresh_worker_finished", result_values=observed, raw_observation_reads=result["raw_observation_reads"])
    file_hashes = {"registry": sha(registry_path),
                   "candidates": sha(audit / "candidates.jsonl"),
                   "discovery": sha(audit / "discovery.json"),
                   "formal_certificate": sha(audit / "formal_certificate.json"),
                   "results": sha(runtime / "result.json")}
    save(audit / "artifact_hashes.json", file_hashes)
    return {"run": directory.name, "selected": discovery["selected"],
            "values": observed, "hashes": file_hashes, "input_files_deleted": True}


def write_report(base, reproducibility, checks):
    first = base / "run_01"
    discovery = read(first / "audit" / "discovery.json")
    finite = read(first / "audit" / "finite_checks.json")
    rows = "\n".join(f"| {k} | {v} |" for k, v in discovery["filter_counts"].items())
    ties = "\n".join(f"| `{v['phi']}` | `{v['decoder']}` | {v['status']} | {v['reason'] or '通用域证书通过'} |"
                     for v in discovery["tie_certificates"])
    report = f"""# Intuition 最小结构生成与计算闭环实验 V0.1：实验报告

本次实际运行完成了有限语法中的自动搜索、商集构造、类型注册、运行输入删除和独立执行。结论限定于此原型，不宣称唯一识别了物理概念，也不宣称不依赖人为先验的严格自主数学发现。

## 实测结果

- 候选不变量 {discovery['candidate_count']} 棵，解码表达式 {discovery['decoder_count']} 棵；未做交换律或答案导向剪枝。
- 能在训练集精确恢复的候选对：{discovery['eligible_pair_count']}。
- 选中 φ=`{discovery['selected']['phi_text']}`，g=`{discovery['selected']['decoder_text']}`，描述长度 {discovery['selected']['description_length']}。
- Type_1 三个已观察实例：{', '.join(g['id']+'='+g['value'] for g in discovery['groups'])}。
- 验证数据在注册冻结后读取，3/3 属于既有类且距离预测准确，注册文件哈希前后相同。
- 实际重复 {len(reproducibility['runs'])} 次，候选日志、选择结果、证书、注册文件、执行结果的 SHA-256 逐项相同：{reproducibility['all_equal']}。
- 全部运行的最终输出：14、21、25；运行输入文件删除后才启动执行进程。
- 自动化测试结果：{checks['status']}。完整输出见本次目录 `test_results.txt`。

## 搜索与筛选

语法 φ ::= d | t | φ+φ | φ-φ | φ*φ | φ/φ，最多5个节点（叶与运算符各1）。对应1/3/5节点分别为2/16/256棵，总274。g的叶变量为z,t，其余相同。所有数值是Fraction精确有理数，无常量叶、无物理量纲规则、无Velocity初始类型。

筛选固定为：训练上有定义；拒绝训练常数；至少两个类且每类至少两个不同观察；存在g能零残差恢复全部距离；最小化L(φ)+L(g)；同复杂度按表达式字符串字典序排序；通用全域证书是注册前硬门槛。残差编码只实现“零残差可行/非零不可行”，不是完整概率MDL编码。验证和最终答案不参与排名。

| 过滤结果 | 候选数 |
|---|---:|
{rows}

同最小训练描述长度的候选全部公开：

| φ | g | 域证书 | 原因 |
|---|---|---|---|
{ties}

正训练集不能区分互为倒数的表示。在声明域d≥0,t>0上，含零距离奇点的表示会被通用分母规则拒绝。字典序也属于人为确定的表示约定，不是物理语义证据。

## 商集、表示与证明

X={{(d,t)∈Q² | d≥0,t>0}}，x~y当且仅当φ(x)=φ(y)，Type_1=X/~。三个实例只是观察到的三个类，全域商集并非仅三个元素。数值坐标由φ诱导；注册文件保留关系、域、AST及类型身份，而不保留代表元观察。

形式验证使用独立通用检查器：从AST递归生成精确有理多项式；每次除法证明除数在声明域非零；用交叉多项式为零证明g(φ(d,t),t)=d；再检查任意新τ>0时总定义和Distance非负闭包。等价关系使用等号拉回证明树，表示单射来自关系定义，Travel代表元无关来自g仅接受坐标与时间的因子化。证书写出AST、规范化结果、分母条件、通用推理步骤。信任基础是Python/Fraction、所附多项式实现和这些通用数学模板；不是Lean/Coq等外部证明器。

有限交叉检查另存：自反{finite['reflexivity']['count']}、对称{finite['symmetry']['count']}、传递{finite['transitivity']['count']}、观察时间上的代表元一致性{finite['descent']['count']}项，全部通过。这些有限检查不替代全域证明。

## 删除与独立执行的证据边界

训练、验证各自运行在子进程中，结束后实际unlink训练和验证输入文件，记录文件哈希、删除前后存在状态并断言输入目录为空。之后创建新运行目录中的任务，使用 `python -I -S` 启动只依赖标准库的独立解释器脚本；运行阶段仅加载registry.json和tasks.json。AST解释器实际执行生成的Travel规则，没有任何目标输出查表。I/O审计钩子拒绝打开未声明文件；结果包含实际读取清单。

完整候选日志、分组和可复现种子仍能重建原始数值，位于观察者/审计边界之外。故可支持的结论是“执行脱离原始观察”，不支持“机器上所有原始数值不可恢复地消失”。I/O审计是合作式进程约束，并非操作系统级安全隔离；初始标准库导入发生在审计安装前。

## 六项验收的准确结论

| 项 | 结果与边界 |
|---|---|
| 初始无目标类型/公式 | 搜索器无Velocity、无目标AST预置、无答案特殊分支；通用语法会自然枚举目标表达式 |
| 自动发现不变量 | 有界自动搜索通过；严格无先验自主发现与唯一语义识别未证明 |
| 构造新对象 | 生成Type_1商集定义、三个实例、精确坐标 |
| 数学一致性 | 通用证书与独立有限交叉检查通过，信任基础如上 |
| 独立注册 | 类型化Travel从registry加载；执行脚本不导入发现程序 |
| 未见任务 | 删除运行输入后实际得到14、21、25 |

## 局限与停止条件

搜索语法、距离预测目标、重复类准则、域、命名规则、注册框架由人提供。六个精确训练点不证明真实世界普遍匀速规律；全域恒等式证明的是选中表达式的代数性质。没有噪声、加速度、多维运动、单位换算、外推统计或开放领域发现；不增加这些研究方向。符号符号判定是保守充分条件，可能拒绝实际可证明的复杂表达式。系统只在当前有理数一维边界内运行。

局部可执行闭环已完成并重复验证，因此停止本次V0.1实施。若将“自主发现”理解为完全没有人为归纳偏好，则该更强目标未达到，不能以本报告替代证明。

## 证据导航

本次实验目录：`{base.name}`。每次运行均有：`audit/initial_manifest.json`、`candidates.jsonl`、`discovery.json`、`formal_certificate.json`、`finite_checks.json`、`validation.json`、`deletion.json`、`events.jsonl`、`artifact_hashes.json`，以及 `runtime/registry.json`、`tasks.json`、`runtime.py`、`result.json`。重复一致性见`{base.name}/reproducibility.json`，测试明细见`{base.name}/test_results.txt`。源文件哈希记录于initial_manifest，未作数字签名，不声称日志防篡改。
"""
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are refused")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-tests", action="store_true", help="Still runs experiments; marks tests unexecuted")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three repeats are required")
    base = args.out.resolve() if args.out else ROOT / "runs"
    if args.out is None:
        suffix = 2
        while base.exists():
            base = ROOT / f"runs_{suffix:03d}"
            suffix += 1
    if base.exists():
        parser.error("Output directory exists; use a new directory")
    base.mkdir(parents=True)
    try:
        records = []
        for index in range(1, args.repeats + 1):
            print(f"Running {index}/{args.repeats}...", flush=True)
            records.append(run_once(base / f"run_{index:02d}"))
        equal = all(item["hashes"] == records[0]["hashes"] for item in records)
        reproducibility = {"runs": records, "all_equal": equal,
                           "comparison": "SHA-256 of full deterministic artifacts, not just final answers"}
        save(base / "reproducibility.json", reproducibility)
        if not equal:
            raise AssertionError("Repeat artifacts differ")
        checks = {"status": "NOT RUN (--skip-tests)"}
        if not args.skip_tests:
            env = dict(os.environ)
            env["INTUITION_RUNS"] = str(base)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                  cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            (base / "test_results.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
            checks = {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode}
            save(base / "test_summary.json", checks)
            if proc.returncode:
                raise RuntimeError(f"Automated tests failed: see {base / 'test_results.txt'}")
        write_report(base, reproducibility, checks)
        print(f"PASS: {len(records)} identical runs; outputs 14, 21, 25.\nEvidence: {base}\nReport: {ROOT / 'REPORT.md'}")
    except Exception as exc:
        save(base / "failure.json", {"status": "FAIL", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
