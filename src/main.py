"""1 つの run（= 1 モデル）で SciGym-small を解かせ、eval_inputs と instances.json を書く。"""

import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import yaml


def run_instance(cfg, run_dir, instance):
    out = run_dir / "instances" / instance.name
    if (out / "reactions.json").exists() and json.loads((out / "reactions.json").read_text()).get("timed_out"):
        (out / "evaluation.json").unlink(missing_ok=True)  # フォールバック採点の件は、前の run の出力を持ち込んだときにやり直す
    if (out / "evaluation.json").exists():
        return
    args = {
        "instance_dir": str(instance),
        "out_dir": str(out),
        "model": cfg.run_model,
        "base_url": cfg.base_url,
        "max_iterations": 2 if cfg.mode == "sanity" else cfg.max_iterations,
        "eval_debug_rounds": cfg.eval_debug_rounds,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stdout.txt", "a") as log:
        try:
            subprocess.run([sys.executable, "-m", "src.train", json.dumps(args)], stdout=log, stderr=subprocess.STDOUT,
                           timeout=cfg.instance_timeout)
        except subprocess.TimeoutExpired:
            print(f"[{instance.name}] killed after {cfg.instance_timeout}s", file=log)
    if not (out / "evaluation.json").exists():  # 失敗した run の作業ディレクトリは残らないので原因を標準出力へ
        print(f"[{instance.name}] no evaluation.json; log tail:", *(out / "stdout.txt").read_text().splitlines()[-25:], sep="\n  ")


def cli_args():
    """hydra 形式の key=value を読む。hydra / omegaconf は scigym が固定する petab の antlr 版と衝突する"""
    return {k: yaml.safe_load(v) for k, _, v in (a.partition("=") for a in sys.argv[1:])}


def main():
    cli = cli_args()
    run_id = cli["run"]
    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg.update(cli)
    cfg["run"] = yaml.safe_load(open(f"config/run/{run_id}.yaml"))
    cfg = SimpleNamespace(**cfg, run_model=cfg["run"]["model"])
    run_dir = Path(cfg.results_dir) / run_id
    instances = sorted(p for p in Path(cfg.data_dir).iterdir() if p.is_dir())
    if cfg.mode == "sanity":
        instances = [p for p in instances if p.name == "BIOMD0000000027"]
    elif cfg.mode == "pilot":
        instances = instances[::14]
    # API エラーで evaluation.json が出なかった件は 2 回までやり直す
    for _ in range(3):
        with ThreadPoolExecutor(cfg.workers) as pool:
            list(pool.map(lambda p: run_instance(cfg, run_dir, p), instances))
    # 3 試行とも終わらなかった件は、公式の「有効な SBML が提出されなかった」場合と同じく不完全 SBML の採点値にする
    for instance in instances:
        out = run_dir / "instances" / instance.name
        if not (out / "evaluation.json").exists():
            args = {"instance_dir": str(instance), "out_dir": str(out), "score_partial": True}
            subprocess.run([sys.executable, "-m", "src.train", json.dumps(args)], timeout=600)
    results, predicted, reference = {}, [], []
    for instance in instances:
        out = run_dir / "instances" / instance.name
        # Seyval の出力一覧は 1000 ファイルまで。反復ごとのコードは chat_history.yaml に含まれるので削る
        shutil.rmtree(out / "codes", ignore_errors=True)
        (out / "chat_history_readable.txt").unlink(missing_ok=True)
        if not (out / "evaluation.json").exists():
            continue
        results[instance.name] = json.loads((out / "evaluation.json").read_text())
        reactions = json.loads((out / "reactions.json").read_text())
        results[instance.name]["tokens"] = [reactions["input_tokens"], reactions["output_tokens"]]
        results[instance.name]["timed_out"] = reactions.get("timed_out", False)
        missing, added = set(reactions["missing"]), set(reactions["added"])
        for h in sorted(missing | added):
            reference.append(int(h in missing))
            predicted.append(int(h in added))
    (run_dir / "instances.json").write_text(json.dumps(results, indent=1))
    (run_dir / "eval_inputs").mkdir(exist_ok=True)
    (run_dir / "eval_inputs" / "binary_classification.json").write_text(
        json.dumps({"predicted_labels": predicted, "reference_labels": reference})
    )
    stage = cfg.mode.upper()
    ste = [r["observe_smape"] for r in results.values()]
    missing_instances = len(instances) - len(results)
    if missing_instances or not all(math.isfinite(x) for x in ste):
        print(f"{stage}_VALIDATION: FAIL reason=missing_metrics")
        sys.exit(1)
    summary = {"n_instances": len(results), "ste_mean": sum(ste) / len(ste),
               "n_success": sum(r["success"] for r in results.values())}
    print(f"{stage}_VALIDATION_SUMMARY: {json.dumps(summary)}")
    print(f"{stage}_VALIDATION: PASS")


if __name__ == "__main__":
    main()
