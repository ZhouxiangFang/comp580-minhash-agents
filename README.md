# MinHash for efficient agent trajectories (COMP 480/580, Assignment 2)

Two MinHash-based optimisations for the Spartacus agent, and the scripts that
measure them on the assignment's datasets.

| File | What it does |
|---|---|
| `minhash.py` | Three-word shingles, 128-function MinHash signatures, banded LSH index. Standard library only. |
| `policy.py` | Task-level reuse: answer an incoming task from a verified record of the same item, without running the agent. |
| `content.py` | Content-level reuse: replace tool output the model has already seen with an exact reference to it. |
| `run_experiments.py` | Runs the four configurations and logs one row of metrics per task. |
| `analyze.py` | Summary table, offline threshold sweep, plots. |

The harness, the datasets and the verifiers are used unchanged. No file here
reads a dataset's `hidden/` directory; answers are graded only by calling the
dataset's own `verify.py`.

## Method

**Task-level reuse.** LSH retrieves verified records whose text has an estimated
Jaccard similarity of at least the threshold (default 0.5). Similarity only
nominates a record. Its answer is reused only if the task and the record are the
same text up to formatting: the same words and numbers in the same order, or the
same paragraphs in another order. One changed word, number or identifier blocks
reuse. A reused answer is graded like any other; if it fails, the task fails.
Answers enter the index only after they pass verification.

The rule contains no dataset-specific logic. The fields to compare are the ones
a task shares with the history records (dates are skipped), and a record's
answer is whatever history records have that tasks do not.

**Content-level reuse.** Within one agent run, each tool output is compared by
MinHash (numbers masked) with the outputs before it. Against the most similar
one an exact line diff is computed, and runs of identical lines are replaced by
a `[reuse]` line naming where they were shown. Changed lines are always shown in
full, and lines that disappeared are noted. If the result is not at most 70 % of
the original length, the original output is returned.

## Setup

Clone the two provided repositories next to this one (or point to them with
`--assignment-root` / `--spartacus-root`):

```sh
git clone https://github.com/sahiljoshi515/Assignment2_COMP_580.git ../Assignment2_COMP_580
git clone https://github.com/sahiljoshi515/Spartacus.git ../Spartacus
mkdir -p ~/.config/spartacus
echo 'GEMINI_API_KEY=...' > ~/.config/spartacus/env && chmod 600 ~/.config/spartacus/env
```

Python 3.10 or newer. The plots need `matplotlib`; nothing else is installed.

## Reproducing the experiments

Datasets: `tickets`, `bugs`, `docs`, `citations`, `hosts`, `threads`,
`revisions`, `logs`. Model: `gemini-3.5-flash-lite`. Each configuration is run
once; the seed fixes the MinHash hash functions (the harness does not expose a
seed for the model).

```sh
python3 run_experiments.py --config baseline task content both --seed 0
python3 analyze.py summary
python3 analyze.py sweep
python3 analyze.py plots
```

Results go to `results/<config>/<dataset>/rows.jsonl` (one row per task), with
the agent's working directories and transcripts beside it. An interrupted run
continues where it stopped; `--fresh` starts over. `--limit 5` runs only the
first five tasks of each dataset, which is a cheap way to check the setup.
