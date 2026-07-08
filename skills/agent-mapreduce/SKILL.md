---
name: agent-mapreduce
description: Run a recursive agentic map-reduce experiment - beam search over research branches. Fans out K single-variable ideas per generation into isolated git worktrees, scores each with the same fixed eval command, and keeps the top B as the next frontier. Use when the user wants to systematically optimize a measurable target (training run, prompt, retrieval config, compiler flags) and a command can score each attempt. Triggers on "map-reduce experiment", "beam search my experiments", "fan out ideas and keep the best", "agent mapreduce".
---

# agent-mapreduce

You are about to execute the program in `program.md` from [github.com/sachinkesiraju/agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce). The skill is a thin launcher; the program is the contract.

## Launch

1. Get `amr.py` and `program.md` into the root of the target repo, unless they are already there: clone https://github.com/sachinkesiraju/agent-mapreduce.git to a temporary directory and copy the two files from it. Make sure `results.tsv` and eval logs are gitignored in the target repo.
2. Fill in the parameter table at the top of the copied `program.md`. Derive what you can from the repo itself (eval command, metric name, GPU count). Ask the user for anything you cannot derive - at minimum `eval_cmd`, `metric_grep`, and whether the metric is minimized or maximized.
3. Confirm the filled-in parameters with the user once, then execute `program.md` exactly: resume check, baseline noise margin, then the generation loop. Do not modify `amr.py` or the loop itself - only the parameters and the target they point at.

## Non-negotiables (from the program)

- State lives in `results.tsv` and git only. If `results.tsv` already exists, this is a resume: rebuild the frontier with `python3 amr.py tree` and `reduce`, never from memory.
- Scores are grepped from eval artifacts by the orchestrator. A worker's claim is not a measurement.
- A candidate survives only by beating its own parent by more than the noise margin and passing any cost guards.
- The final champion must revalidate on holdout before it ships.
