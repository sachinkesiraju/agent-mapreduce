# agent-mapreduce

Autoresearch as recursive agentic map-reduce.

Karpathy's autoresearch loop tests one idea at a time:

```text
one idea -> eval -> keep/revert -> repeat
```

Agentic map reduce tests multiple ideas per generation:

```text
K ideas -> K isolated worktrees -> keep top B -> repeat
```

Use it when the search space is noisy and idea-rich or when you suspect a single greedy walk will likely miss optimal paths or discard changes that only pay off in combination. It works for any repo where an agent can apply an idea and a command can score it.

The agent reads `program.md` and does the editing. `amr.py` keeps the log and selects the next frontier.

## How it works

The repo has three important pieces:

- `program.md` — instructions for the agent. Edit this for your domain.
- `amr.py` — stdlib-only helper: `log`, `reduce`, `tree`.
- `example/` — concrete setups for autoresearch and prompt optimization.

The loop:

1. Run the baseline twice. The spread becomes the noise margin.
2. Review artifacts and prior results, then propose K ideas from the current frontier.
3. Create one git worktree per idea.
4. Send one worker agent into each worktree. Every candidate uses the same eval split, seed if applicable, budget, and scoring command.
5. The orchestrator greps scores from artifacts and logs every result.
6. `amr.py reduce` keeps candidates that beat their own parent by more than the margin and satisfy any cost guards.
7. The top B survivors become the next frontier.
8. If survivors touch different regions, `amr.py` prints a combination candidate.
9. Repeat. Final winner must revalidate on holdout before it ships.

With K=4, B=2:

```text
baseline
├── idea A   keep
│   ├── idea A1   keep
│   └── idea A2   discard
├── idea B   keep
│   ├── idea B1   keep
│   └── idea B2   discard
├── idea C   crash
└── idea D   discard
```

## Quick start

Copy `program.md` and `amr.py` into the repo you want to optimize. Fill in the params at the top of `program.md`:

```md
eval_cmd:     <command that runs eval and writes a log>
metric_grep:  <command that extracts the score>
metric:       <name> — minimize or maximize
K:            4
B:            2
eval_slots:   <number of GPUs or contended eval resources>
timeout:      <when to kill a stuck eval>
cost_guard:   <optional, e.g. vram:+0.15>
holdout_cmd:  <optional final eval>
tag:          <run tag>
```

Then start a coding agent in that repo and say:

```text
Read program.md and set up a new recursive agentic map-reduce experiment. Do the setup first, then begin the loop.
```

## amr.py

```bash
python3 amr.py log [--cost name=value] [--region name] <gen> <commit> <parent> <score|-> <ran|crash|note> <description...>
python3 amr.py reduce --gen <N> --beam <B> --margin <M> [--maximize] [--cost name:+tol]
python3 amr.py tree [--maximize]
```

Example:

```bash
python3 amr.py log --cost vram=44.0 0 base000 - 0.998100 ran "baseline run 1"
python3 amr.py log --cost vram=44.0 0 base000 - 0.997900 ran "baseline run 2"

python3 amr.py log --cost vram=44.2 --region optimizer 1 lr00400 base000 0.993200 ran "increase LR to 0.04"
python3 amr.py log --cost vram=44.1 --region architecture 1 gelu000 base000 0.996000 ran "replace SiLU with GeLU"
python3 amr.py log --cost vram=80.0 --region architecture 1 wide000 base000 - crash "double model width OOM"

python3 amr.py reduce --gen 1 --beam 2 --margin 0.0002 --cost vram:+0.15
python3 amr.py tree
```

Every candidate is compared to its own parent, not to a global baseline. `--cost vram:+0.15` means the candidate must also stay within 15% of its parent's VRAM.

## Generic example

Prompt optimization is the same loop with different params:

```md
eval_cmd:     python3 eval.py --prompt prompt.md --data eval.jsonl > run.log 2>&1
metric_grep:  grep '^accuracy:' run.log
metric:       accuracy — maximize
K:            6
B:            2
```

Workers edit `prompt.md`; `eval.py` and eval data stay fixed. Reduce with `--maximize`. See `example/prompt-optimization.md`.

## Why use this over autoresearch?

Traditional autoresearch is a greedy hill-climber:

```text
greedy: keep the first thing that improves
```

This is beam search over research branches:

```text
beam: compare many ideas, keep the best few, recurse
```

This helps when one idea at a time is too path-dependent. Agentic map reduce allows you to compare sibling ideas from the same parent, avoid advancing on tiny noisy wins, keep a few competing branches alive, and discover paths where A alone is weak but A+B wins.

Use this when:

- you want to avoid local optima;
- you care about comparing sibling ideas fairly;
- you want a tree of research, not a linear walk;
- you can afford K evaluations per generation;
- useful changes may only pay off in combination.

Stick with plain autoresearch when evals are very expensive, there is one obvious direction, or you want the smallest possible loop.

## License

MIT
