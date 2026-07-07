# example: generic prompt optimization

This shows the same recursive agentic map-reduce loop on a generic idea task,
not ML training. The target is a prompt file. The metric is whatever your eval
script reports.

Imagine a repo like:

```
prompt.md          — the prompt the agent edits
eval.jsonl         — fixed examples, never edited
eval.py            — fixed evaluator, prints accuracy
program.md
amr.py
```

The task: improve `prompt.md` against `eval.jsonl`.

## Parameters

| param        | value |
|--------------|-------|
| eval_cmd     | `python3 eval.py --prompt prompt.md --data eval.jsonl > run.log 2>&1` |
| metric_grep  | `grep '^accuracy:' run.log` |
| metric       | accuracy — maximize |
| K            | 6 |
| B            | 2 |
| eval_slots   | 6 if evals are cheap; lower if the evaluator calls paid APIs |
| timeout      | 5 minutes |
| holdout_cmd  | `python3 eval.py --prompt prompt.md --data holdout.jsonl > holdout.log 2>&1` |
| tag          | date-based, e.g. `jul6-prompt` |

Use `--maximize` when reducing:

```bash
python3 amr.py reduce --gen 1 --beam 2 --margin 0.01 --maximize
```

## Domain rules

- Modify `prompt.md` only.
- Do not modify `eval.py`, `eval.jsonl`, or `holdout.jsonl`.
- Each idea must target one prompt strategy, e.g. output format, examples,
  rubric wording, refusal policy, reasoning style, or retrieval instructions.
- Final champion must pass holdout. A map-set win that fails holdout is an
  overfit, not a result.

## Example generation

Baseline:

```bash
python3 amr.py log 0 base000 - 0.710000 ran "baseline prompt run 1"
python3 amr.py log 0 base000 - 0.715000 ran "baseline prompt run 2"
```

Margin is `0.005`.

Generation 1 ideas:

```bash
python3 amr.py log --region output_format 1 jsonfmt base000 0.760000 ran "force JSON schema output"
python3 amr.py log --region examples 1 fewshot base000 0.785000 ran "add three hard few-shot examples"
python3 amr.py log --region brevity 1 terse00 base000 0.720000 ran "shorten instructions"
python3 amr.py log --region rubric 1 rubric0 base000 0.748000 ran "add grading rubric before answer"
python3 amr.py log --region reasoning 1 chain00 base000 0.735000 ran "ask for brief hidden scratchpad summary"
python3 amr.py log --region policy 1 policy0 base000 - crash "prompt exceeded model context"
```

Reduce:

```bash
python3 amr.py reduce --gen 1 --beam 2 --margin 0.005 --maximize
```

Frontier:

```text
fewshot jsonfmt
```

Generation 2 now recurses from both branches. The agent might try:

```text
fewshot -> fewshot + better negative examples
fewshot -> fewshot + compressed examples
jsonfmt -> JSON schema + enum constraints
jsonfmt -> JSON schema + repair instructions
```

Again, every child is compared to its own parent, and the best B globally
survive. This is the same system as the autoresearch example; only `eval_cmd`,
`metric`, and the editable target changed.

## Worker prompt template

> You are one shard of a recursive agentic map-reduce experiment. Your
> worktree: `<path>`. Implement exactly this one prompt idea in `prompt.md` and
> nothing else: "<idea>: <rationale>". Commit with a one-line message. Then run
> `python3 eval.py --prompt prompt.md --data eval.jsonl > run.log 2>&1` and
> stop. Do not read or report the result. Do not touch `eval.py`, `eval.jsonl`,
> or `holdout.jsonl`.
