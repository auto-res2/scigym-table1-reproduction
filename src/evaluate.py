"""各 run の instances.json を 137 件平均にまとめ、airas-eval の micro 指標と共に metrics.json へ書く。"""

import json
from pathlib import Path

import sys

import yaml


def main():
    cfg = {k: yaml.safe_load(v) for k, _, v in (a.partition("=") for a in sys.argv[1:])}
    for run_id in cfg["run_ids"]:
        run_dir = Path(cfg["results_dir"]) / run_id
        rows = list(json.loads((run_dir / "instances.json").read_text()).values())
        n = len(rows)
        metrics = {"n_instances": n, "n_success": sum(r["success"] for r in rows)}
        for key, name in [("observe_smape", "ste"), ("rp_precision", "rms_precision"), ("rp_recall", "rms_recall"),
                          ("rp_f1", "rms_f1"), ("rpm_precision", "rmsm_precision"),
                          ("rpm_recall", "rmsm_recall"), ("rpm_f1", "rmsm_f1")]:
            metrics[name] = sum(r[key] for r in rows) / n
        metrics["input_tokens"] = sum(r["tokens"][0] for r in rows)
        metrics["output_tokens"] = sum(r["tokens"][1] for r in rows)
        report = json.loads((run_dir / "evaluation" / "binary_classification.json").read_text())
        for key in ("precision", "recall", "f1"):
            metrics[f"micro_rms_{key}"] = report["metrics"][key]
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(run_id, json.dumps(metrics))


if __name__ == "__main__":
    main()
