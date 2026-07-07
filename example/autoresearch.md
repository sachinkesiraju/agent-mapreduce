# example: agentic map-reduce over karpathy/autoresearch

This instantiates `program.md` for [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
— the single-GPU nanochat training setup. Where the original runs one idea at a
time on one branch, this runs the same experiment as a beam search.

Setup: clone karpathy/autoresearch, run `uv run prepare.py` once to build data
and tokenizer, then copy `amr.py`, `program.md`, and this file into the clone.
Execute `program.md` with the parameters and domain rules below.

## Parameters

| param        | value |
|--------------|-------|
| eval_cmd     | `uv run train.py > run.log 2>&1` |
| metric_grep  | `grep '^val_bpb:' run.log` |
| metric       | val_bpb — minimize |
| K            | 4 |
| B            | 2 |
| eval_slots   | number of GPUs (1 on a single-GPU box — implementations may overlap, but training runs must queue) |
| timeout      | 10 minutes per eval (training is budgeted at 5; kill and log as crash beyond 10) |
| cost_guard   | `peak_vram_mb:+0.15` unless you intentionally allow more memory |
| holdout_cmd  | none — `evaluate_bpb` in `prepare.py` is the single ground-truth metric; finishing re-runs eval_cmd on baseline and champion |
| tag          | date-based, e.g. `jul6` |

With eval_slots=1 and 5-minute runs, a generation of ~4-6 candidates takes
~25-35 minutes of wall clock. Breadth costs eval time serially on one GPU; the
win is that implementation overlaps and the search keeps B lineages alive
instead of one greedy line.

## Domain rules (from the original program.md — they still bind every worker)

- Modify `train.py` only. Architecture, optimizer, hyperparameters, batch
  size, model size — all fair game.
- `prepare.py` is read-only: it holds the fixed eval, data loading, tokenizer,
  and training constants. The eval is the ground truth; never touch it.
- No new packages or dependencies beyond `pyproject.toml`.
- VRAM is a cost guard: extract `grep '^peak_vram_mb:' run.log`, log it with
  `--cost peak_vram_mb=<value>`, and reduce with `--cost peak_vram_mb:+0.15`
  unless you intentionally allow more memory.
- Simplicity criterion: a 0.001 gain that adds 20 lines of hacky code is
  probably not worth it; an equal result from deleted code is a keep.

## Worker prompt template

> You are one shard of a map-reduce experiment. Your worktree:
> `<path>`. Implement exactly this one idea in `train.py` and nothing else:
> "<idea>: <rationale>". Commit with a one-line message. Then run
> `uv run train.py > run.log 2>&1` and stop — do not read or report the
> results, do not touch anything outside your worktree, do not modify
> `prepare.py`. If the run crashes on a trivial bug of your own making, fix it
> and re-run once.

The orchestrator (you) then greps `run.log` in each worktree, logs each result
with `amr.py log --cost peak_vram_mb=<value> --region <region>`, and reduces
with `--cost peak_vram_mb:+0.15`.
