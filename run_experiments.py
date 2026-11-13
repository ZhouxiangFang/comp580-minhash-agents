"""Run the four configurations of Assignment 2 and log one row of metrics per task.

    python3 run_experiments.py --config baseline task content both --seed 0

For every task of a dataset, in stream order:
  1. build a fresh working directory (instructions, history, task.json, scenario files);
  2. with task-level reuse on, ask `policy.ReusePolicy` for a verified answer;
  3. otherwise run the Spartacus agent (with content-level reuse on, its tools are
     wrapped by `content.wrap_tools`);
  4. grade answer.json with the dataset's verify.py, and index the answer only if it passed.

The harness, the datasets and the verifiers are used as they are. This script
never opens a dataset's hidden/ directory; only verify.py does.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from content import ContentCache, wrap_tools
from policy import ReusePolicy

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["tickets", "bugs", "docs", "citations", "hosts", "threads", "revisions", "logs"]
CONFIGS = {"baseline": (False, False), "task": (True, False),
           "content": (False, True), "both": (True, True)}
MODEL = "gemini-3.5-flash-lite"
PROMPT = ("Read INSTRUCTIONS.md and task.json in the working directory, answer the request, "
          "and write the answer to answer.json.")


def dataset_dir(args, dataset):
    return os.path.join(args.assignment_root, "assignment2", "datasets", dataset)


def load_stream(env):
    """(history records, incoming tasks in stream order)."""
    with open(os.path.join(env, "history.jsonl"), encoding="utf-8") as handle:
        history = [json.loads(line) for line in handle if line.strip()]
    tasks = []
    for name in sorted(os.listdir(os.path.join(env, "incoming"))):
        with open(os.path.join(env, "incoming", name), encoding="utf-8") as handle:
            tasks.append(json.load(handle))
    tasks.sort(key=lambda task: (task.get("seq", 0), str(task.get("tid"))))
    return history, tasks


def make_workdir(env, task, workdir):
    """What the agent may see: env/ without the other tasks, plus this task's scenario files."""
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    for name in os.listdir(env):
        if name not in ("incoming", "scenarios"):
            shutil.copy(os.path.join(env, name), workdir)
    with open(os.path.join(workdir, "task.json"), "w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2)
    scenarios = os.path.join(env, "scenarios")
    if os.path.isdir(scenarios) and task.get("scenario"):
        wanted = "".join(c for c in str(task["scenario"]).lower() if c.isalnum())
        for name in os.listdir(scenarios):
            if "".join(c for c in name.lower() if c.isalnum()) == wanted:
                shutil.copytree(os.path.join(scenarios, name), workdir, dirs_exist_ok=True)


def run_agent(args, workdir, session_path, cache):
    """Run the agent on one working directory; return its request, tool and token counts."""
    from spartacus import Harness
    from spartacus.tools import core_tools
    used = {"model_requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def on_event(kind, payload):
        if kind == "assistant":
            used["model_requests"] += 1
            used["input_tokens"] += payload.get("usage", {}).get("input", 0)
            used["output_tokens"] += payload.get("usage", {}).get("output", 0)
        elif kind == "tool_start":
            used["tool_calls"] += 1

    # Sub-agents are off in every configuration: a child harness is built without
    # the caller's tools, so it would read files around the content layer.
    agent = Harness(workdir, model=args.model, on_event=on_event, max_turns=args.max_turns,
                    enable_subagents=False, session_path=session_path,
                    extra_tools=wrap_tools(core_tools(workdir), cache) if cache else None)
    agent.run(PROMPT)
    return used


def use_openai_compatible(base):
    """Reach the model through an OpenAI-compatible gateway (OpenRouter) instead of Google.

    Only used with --api-base; the harness files stay as they are. Three things
    are adapted so the gateway behaves like the harness's own Gemini route:
    a tool result is sent as {"result": text} (the gateway drops one that is not
    a JSON object), the temperature is the 0.4 that route uses, and the model's
    reasoning details and tool-call ids are sent back with its earlier tool calls
    (that route does the same with thought signatures; without them Gemini stops
    calling tools).
    """
    from spartacus import models, provider
    models.PROVIDERS["openai"]["base"] = base
    post = provider._post
    seen = {}                                 # (system prompt, assistant turn) -> (details, call ids)

    def adapted(url, body, headers, retries=5):
        if not url.endswith("/chat/completions"):
            return post(url, body, headers, retries)
        run, turn, messages, ids = body["messages"][0]["content"], 0, [], {}
        for message in body["messages"]:
            if message["role"] == "tool":
                message = dict(message, content=json.dumps({"result": message["content"]}),
                               tool_call_id=ids.get(message["tool_call_id"], message["tool_call_id"]))
            elif message["role"] == "assistant":
                details, real_ids = seen.get((run, turn), (None, []))
                calls = [dict(call) for call in message.get("tool_calls") or []]
                if details and len(calls) == len(real_ids):
                    for call, real in zip(calls, real_ids):
                        ids[call["id"]], call["id"] = real, real
                    message = dict(message, tool_calls=calls, reasoning_details=details)
                turn += 1
            messages.append(message)
        data = post(url, dict(body, temperature=0.4, messages=messages), headers, retries)
        reply = (data.get("choices") or [{}])[0].get("message") or {}
        if reply.get("reasoning_details"):
            seen[run, turn] = (reply["reasoning_details"],
                               [call.get("id") for call in reply.get("tool_calls") or []])
        return data

    provider._post = adapted


def read_answer(workdir):
    try:
        with open(os.path.join(workdir, "answer.json"), encoding="utf-8") as handle:
            answer = json.load(handle)
        return answer if isinstance(answer, dict) else None
    except (OSError, ValueError):
        return None


def verify(args, dataset, tid, workdir):
    """True when the dataset's verifier accepts workdir/answer.json (exit code 0)."""
    script = os.path.join(dataset_dir(args, dataset), "verify.py")
    done = subprocess.run([sys.executable, script, "--tid", str(tid), "--answer",
                           os.path.join(workdir, "answer.json")],
                          cwd=args.assignment_root, capture_output=True, text=True)
    return done.returncode == 0


def run_dataset(args, dataset, config):
    use_policy, use_content = CONFIGS[config]
    env = os.path.join(dataset_dir(args, dataset), "env")
    out = os.path.join(args.out, config, dataset)
    os.makedirs(os.path.join(out, "sessions"), exist_ok=True)
    history, tasks = load_stream(env)
    tasks = tasks[:args.limit] if args.limit else tasks
    rows_path = os.path.join(out, "rows.jsonl")
    done = {}
    if os.path.exists(rows_path) and not args.fresh:
        with open(rows_path, encoding="utf-8") as handle:
            done = {row["tid"]: row for row in map(json.loads, handle)}
    policy = ReusePolicy(history, threshold=args.threshold, seed=args.seed) if use_policy else None

    with open(rows_path, "a" if done else "w", encoding="utf-8") as log:
        for task in tasks:
            tid = task["tid"]
            workdir = os.path.join(out, "work", tid)
            if tid in done:                   # resumed run: restore the index, do not re-run
                if policy and done[tid]["passed"] and read_answer(workdir):
                    policy.add(task, read_answer(workdir))
                continue
            make_workdir(env, task, workdir)
            session_path = os.path.join(out, "sessions", tid + ".jsonl")
            if os.path.exists(session_path):  # a transcript left by an earlier, discarded run
                os.remove(session_path)
            row = {"dataset": dataset, "config": config, "tid": tid, "seq": task.get("seq"),
                   "reused": False, "source": None, "similarity": None, "error": None,
                   "model_requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
                   "policy_seconds": 0.0, "content_seconds": 0.0,
                   "content_chars_in": 0, "content_chars_out": 0, "content_folds": 0}
            started = time.perf_counter()
            match = policy.lookup(task) if policy else None
            row["policy_seconds"] = time.perf_counter() - started
            if match:
                row.update(reused=True, source=str(match["source"]), similarity=match["similarity"])
                with open(os.path.join(workdir, "answer.json"), "w", encoding="utf-8") as handle:
                    json.dump(match["answer"], handle)
            else:
                cache = ContentCache(threshold=args.content_threshold, seed=args.seed) \
                    if use_content else None
                try:
                    row.update(run_agent(args, workdir, session_path, cache))
                except Exception as error:    # an API failure is a failed task, not a crash
                    row["error"] = "%s: %s" % (type(error).__name__, error)
                if cache:
                    row.update(content_seconds=cache.stats["seconds"],
                               content_chars_in=cache.stats["chars_in"],
                               content_chars_out=cache.stats["chars_out"],
                               content_folds=cache.stats["folded"])
            row["seconds"] = time.perf_counter() - started
            row["passed"] = verify(args, dataset, tid, workdir)
            if policy and row["passed"] and read_answer(workdir):  # only verified answers are indexed
                added = time.perf_counter()
                policy.add(task, read_answer(workdir))
                row["policy_seconds"] += time.perf_counter() - added
                row["seconds"] += time.perf_counter() - added
            log.write(json.dumps(row) + "\n")
            log.flush()
            print("%-9s %-9s %s %s %s %.1fs" % (
                config, dataset, tid, "PASS" if row["passed"] else "FAIL",
                "reused from %s" % row["source"] if row["reused"] else
                "%d requests, %d tool calls, %d input tokens" % (
                    row["model_requests"], row["tool_calls"], row["input_tokens"]),
                row["seconds"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seed", type=int, default=0, help="seed of the MinHash hash functions")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api-base", default=None,
                        help="OpenAI-compatible endpoint for 'openai:<model>' ids, e.g. OpenRouter")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="task-level: estimated Jaccard a record needs to be considered")
    parser.add_argument("--content-threshold", type=float, default=0.5,
                        help="content-level: estimated Jaccard an earlier output needs")
    parser.add_argument("--max-turns", type=int, default=40, help="agent turn limit per task")
    parser.add_argument("--limit", type=int, default=0, help="only the first N tasks (pilot run)")
    parser.add_argument("--jobs", type=int, default=1, help="datasets run in parallel")
    parser.add_argument("--fresh", action="store_true", help="ignore rows from an earlier run")
    parser.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    parser.add_argument("--assignment-root", default=os.environ.get(
        "ASSIGNMENT_ROOT", os.path.join(HERE, "..", "Assignment2_COMP_580")))
    parser.add_argument("--spartacus-root", default=os.environ.get(
        "SPARTACUS_ROOT", os.path.join(HERE, "..", "Spartacus")))
    args = parser.parse_args()
    args.assignment_root = os.path.abspath(args.assignment_root)
    args.out = os.path.abspath(args.out)
    sys.path.insert(0, os.path.abspath(args.spartacus_root))
    from spartacus import config
    config.load()                             # read the key file once, before any worker thread
    if args.api_base:
        use_openai_compatible(args.api_base.rstrip("/"))

    for config in args.config:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda dataset: run_dataset(args, dataset, config), args.datasets))


if __name__ == "__main__":
    main()
