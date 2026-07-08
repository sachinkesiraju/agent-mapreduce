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
$ python3 amr.py reduce --gen 1 --beam 2 --margin 0.005 --maximize
gen 1: 5 candidates, 5 beat their parent (margin 0.005), keeping 2
KEEP    fewshot  0.785000  (parent base000, delta +0.070000)  add three hard few-shot examples  regions=examples
KEEP    jsonfmt  0.760000  (parent base000, delta +0.045000)  force JSON schema output          regions=output_format
FRONTIER: fewshot jsonfmt
FUSE_CANDIDATE  fewshot  jsonfmt  regions=examples,output_format
```

Five of six ideas beat the baseline, but only the top two survive the beam.
The `FUSE_CANDIDATE` line says the two survivors touch disjoint regions, so
few-shot examples plus JSON schema is worth trying as one extra candidate in
generation 2.

Generation 2 recurses from both branches, each child scored against its own
parent:

```bash
python3 amr.py log --region examples      2 fewneg0  fewshot 0.801000 ran "add negative examples showing common mistakes"
python3 amr.py log --region examples      2 fewcomp  fewshot 0.779000 ran "compress examples to save tokens"
python3 amr.py log --region output_format 2 jsonenum jsonfmt 0.771000 ran "add enum constraints to JSON schema"
python3 amr.py log --region output_format 2 jsonrep0 jsonfmt 0.762000 ran "add self-repair instructions for invalid JSON"
```

```bash
$ python3 amr.py reduce --gen 2 --beam 2 --margin 0.005 --maximize
gen 2: 4 candidates, 2 beat their parent (margin 0.005), keeping 2
KEEP    fewneg0   0.801000  (parent fewshot, delta +0.016000)  add negative examples showing common mistakes  regions=examples
KEEP    jsonenum  0.771000  (parent jsonfmt, delta +0.011000)  add enum constraints to JSON schema            regions=output_format
FRONTIER: fewneg0 jsonenum
FUSE_CANDIDATE  fewneg0  jsonenum  regions=examples,output_format
```

Note `fewcomp` scored 0.779, better than anything in the `jsonfmt` line, but
it dies: it lost to its own parent (0.785). Compression cost accuracy, and the
comparison against the parent catches that where a global leaderboard would
not. The tree so far:

```text
$ python3 amr.py tree --maximize
base000  0.715000  baseline prompt run 1
├── jsonfmt  0.760000  force JSON schema output
│   ├── jsonenum  0.771000  add enum constraints to JSON schema
│   └── jsonrep0  0.762000  add self-repair instructions for invalid JSON
├── fewshot  0.785000  add three hard few-shot examples
│   ├── fewneg0  0.801000  add negative examples showing common mistakes
│   └── fewcomp  0.779000  compress examples to save tokens
├── terse00  0.720000  shorten instructions
├── rubric0  0.748000  add grading rubric before answer
├── chain00  0.735000  ask for brief hidden scratchpad summary
└── policy0  crash  prompt exceeded model context
```

This is the same system as the autoresearch example; only `eval_cmd`,
`metric`, and the editable target changed. Before shipping, the champion must
still beat the baseline on `holdout.jsonl`.

## Worker prompt template

> You are one shard of a recursive agentic map-reduce experiment. Your
> worktree: `<path>`. Implement exactly this one prompt idea in `prompt.md` and
> nothing else: "<idea>: <rationale>". Commit with a one-line message. Then run
> `python3 eval.py --prompt prompt.md --data eval.jsonl > run.log 2>&1` and
> stop. Do not read or report the result. Do not touch `eval.py`, `eval.jsonl`,
> or `holdout.jsonl`.
