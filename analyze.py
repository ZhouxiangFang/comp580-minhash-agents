"""Tables, the offline threshold sweep and the plots for the report.

    python3 analyze.py summary     # accuracy and cost per dataset and configuration
    python3 analyze.py sweep       # task-level reuse against the similarity threshold
    python3 analyze.py plots       # figures from the two tables (needs matplotlib)

`sweep` calls no model. It replays the logged baseline run: a task the policy
does not reuse gets the answer and the verdict the baseline agent produced, and
a reused answer is graded by the dataset's verify.py.
"""

import argparse
import csv
import json
import os
import sys
import tempfile

import run_experiments as rx
from policy import ReusePolicy

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUMMARY_FIELDS = ["dataset", "config", "tasks", "accuracy", "reused", "wrong_reuse",
                  "model_requests", "tool_calls", "input_tokens", "seconds",
                  "minhash_seconds", "content_chars_saved"]
SWEEP_FIELDS = ["dataset", "validate", "threshold", "tasks", "reused", "wrong_reuse", "accuracy"]


def read_rows(out, config, dataset):
    path = os.path.join(out, config, dataset, "rows.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)


def summary(args):
    table = []
    for dataset in args.datasets:
        for config in rx.CONFIGS:
            rows = read_rows(args.out, config, dataset)
            if not rows:
                continue
            total = lambda key: sum(row[key] for row in rows)
            table.append({
                "dataset": dataset, "config": config, "tasks": len(rows),
                "accuracy": round(total("passed") / len(rows), 4),
                "reused": total("reused"),
                "wrong_reuse": sum(1 for row in rows if row["reused"] and not row["passed"]),
                "model_requests": total("model_requests"), "tool_calls": total("tool_calls"),
                "input_tokens": total("input_tokens"), "seconds": round(total("seconds"), 2),
                "minhash_seconds": round(total("policy_seconds") + total("content_seconds"), 3),
                "content_chars_saved": total("content_chars_in") - total("content_chars_out")})
    print("| " + " | ".join(SUMMARY_FIELDS) + " |")
    print("|" + "---|" * len(SUMMARY_FIELDS))
    for row in table:
        print("| " + " | ".join(str(row[field]) for field in SUMMARY_FIELDS) + " |")
    write_csv(os.path.join(args.out, "summary.csv"), SUMMARY_FIELDS, table)


def sweep(args):
    table = []
    with tempfile.TemporaryDirectory() as scratch:
        for dataset in args.datasets:
            baseline = {row["tid"]: row for row in read_rows(args.out, "baseline", dataset)}
            if not baseline:
                print("no baseline rows for %s; run the baseline first" % dataset)
                continue
            history, tasks = rx.load_stream(os.path.join(rx.dataset_dir(args, dataset), "env"))
            tasks = [task for task in tasks if task["tid"] in baseline]
            graded = {}                       # (tid, answer) -> verdict, so each pair is graded once
            for validate in (True, False):
                for threshold in THRESHOLDS:
                    policy = ReusePolicy(history, threshold=threshold, validate=validate,
                                         seed=args.seed)
                    reused = wrong = passed = 0
                    for task in tasks:
                        tid = task["tid"]
                        match = policy.lookup(task)
                        if match:
                            answer, key = match["answer"], (tid, json.dumps(match["answer"], sort_keys=True))
                            if key not in graded:
                                with open(os.path.join(scratch, "answer.json"), "w") as handle:
                                    json.dump(answer, handle)
                                graded[key] = rx.verify(args, dataset, tid, scratch)
                            ok = graded[key]
                            reused += 1
                            wrong += not ok
                        else:
                            ok = baseline[tid]["passed"]
                            answer = rx.read_answer(os.path.join(args.out, "baseline", dataset, "work", tid))
                        passed += ok
                        if ok and answer:
                            policy.add(task, answer)
                    table.append({"dataset": dataset, "validate": validate, "threshold": threshold,
                                  "tasks": len(tasks), "reused": reused, "wrong_reuse": wrong,
                                  "accuracy": round(passed / len(tasks), 4)})
                    print(table[-1], flush=True)
    write_csv(os.path.join(args.out, "sweep.csv"), SWEEP_FIELDS, table)


def plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(name):
        with open(os.path.join(args.out, name), encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(os.path.join(args.out, name), dpi=150)
        print("wrote", os.path.join(args.out, name))

    rows = load("summary.csv")
    datasets = [d for d in args.datasets if any(r["dataset"] == d for r in rows)]
    cell = {(r["dataset"], r["config"]): r for r in rows}
    configs = [c for c in rx.CONFIGS if any(r["config"] == c for r in rows)]
    width = 0.8 / len(configs)

    # 1. Performance across datasets: one panel per metric, one bar per configuration.
    metrics = [("accuracy", "accuracy"), ("input_tokens", "input tokens"),
               ("tool_calls", "tool calls"), ("seconds", "completion time (s)")]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, (key, title) in zip(axes.flat, metrics):
        for i, config in enumerate(configs):
            axis.bar([x + i * width for x in range(len(datasets))],
                     [float(cell[d, config][key]) if (d, config) in cell else 0 for d in datasets],
                     width, label=config)
        axis.set_xticks([x + 0.4 - width / 2 for x in range(len(datasets))], datasets, rotation=30)
        axis.set_title(title)
    axes[0, 0].legend()
    save(figure, "plot_datasets.png")

    # 2. Efficiency against correctness: input tokens saved versus change in accuracy.
    figure, axis = plt.subplots(figsize=(6, 4.5))
    for config in configs:
        if config == "baseline":
            continue
        points = [(1 - float(cell[d, config]["input_tokens"]) / max(1.0, float(cell[d, "baseline"]["input_tokens"])),
                   float(cell[d, config]["accuracy"]) - float(cell[d, "baseline"]["accuracy"]))
                  for d in datasets if (d, config) in cell and (d, "baseline") in cell]
        axis.scatter([p[0] for p in points], [p[1] for p in points], label=config)
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.set_xlabel("fraction of baseline input tokens saved")
    axis.set_ylabel("accuracy minus baseline accuracy")
    axis.legend()
    save(figure, "plot_tradeoff.png")

    # 3. Overhead against savings: seconds spent in MinHash versus seconds saved.
    figure, axis = plt.subplots(figsize=(7, 4.5))
    shown = [d for d in datasets if (d, "both") in cell and (d, "baseline") in cell]
    axis.bar([x - 0.2 for x in range(len(shown))],
             [float(cell[d, "baseline"]["seconds"]) - float(cell[d, "both"]["seconds"]) for d in shown],
             0.4, label="time saved by both layers")
    axis.bar([x + 0.2 for x in range(len(shown))],
             [float(cell[d, "both"]["minhash_seconds"]) for d in shown], 0.4, label="MinHash overhead")
    axis.set_xticks(range(len(shown)), shown, rotation=30)
    axis.set_ylabel("seconds")
    axis.legend()
    save(figure, "plot_overhead.png")

    # 4. Impact of the similarity threshold, with and without the exact check.
    if os.path.exists(os.path.join(args.out, "sweep.csv")):
        sweep_rows = load("sweep.csv")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for validate, style in (("True", "-"), ("False", "--")):
            for key, axis in (("reused", axes[0]), ("wrong_reuse", axes[1])):
                axis.plot(THRESHOLDS, [sum(int(r[key]) for r in sweep_rows
                                           if r["validate"] == validate and float(r["threshold"]) == t)
                                       for t in THRESHOLDS], style,
                          label="with exact check" if validate == "True" else "similarity only")
        axes[0].set_title("tasks answered by reuse")
        axes[1].set_title("wrong answers caused by reuse")
        for axis in axes:
            axis.set_xlabel("similarity threshold")
            axis.legend()
        save(figure, "plot_threshold.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["summary", "sweep", "plots"])
    parser.add_argument("--datasets", nargs="+", default=rx.DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(rx.HERE, "..", "results"))
    parser.add_argument("--assignment-root", default=os.environ.get(
        "ASSIGNMENT_ROOT", os.path.join(rx.HERE, "..", "Assignment2_COMP_580")))
    args = parser.parse_args()
    args.assignment_root = os.path.abspath(args.assignment_root)
    args.out = os.path.abspath(args.out)
    {"summary": summary, "sweep": sweep, "plots": plots}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
