# agent-mapreduce

Autoresearch as recursive agentic map-reduce.

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) loop tests one idea at a time:

```text
one idea -> eval -> keep/revert -> repeat
```

Agentic map reduce tests multiple ideas per generation:

```text
K ideas -> K isolated worktrees -> keep top B -> repeat
```

Use it when the search space is noisy and idea-rich - when you suspect a single greedy walk will likely miss optimal paths or discard changes that only pay off in combination. It works for any repo where an agent can apply an idea and a command can score it.

The agent reads `program.md` and does the editing. `amr.py` keeps the log and selects the next frontier.

## How it works

The repo has three important pieces:

- `program.md` — instructions for the agent. Edit this for your domain.
- `amr.py` — stdlib-only helper: `log`, `reduce`, `tree`.
- `example/` — three concrete setups: ML training, prompt optimization, and choosing between whole algorithms.

The loop:

1. Run the baseline twice. The spread becomes the noise margin.
2. Review artifacts and prior results, then propose K ideas from the current frontier.
3. Create one git worktree per idea.
4. Send one worker agent into each worktree. Every candidate uses the same eval split, seed if applicable, budget, and scoring command.
5. The orchestrator greps scores from artifacts and logs every result.
6. `amr.py reduce` keeps candidates that beat their own parent by more than the margin and satisfy any cost guards.
7. The top B survivors become the next frontier.
8. If survivors touch disjoint regions, `amr.py` prints a combination candidate.
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

Or install it as a Claude Code skill: copy [skills/agent-mapreduce/](skills/agent-mapreduce/) into `~/.claude/skills/`, then say `/agent-mapreduce` in any repo. The skill is self-contained — it carries its own copies of `program.md` and `amr.py`, copies them into the target repo, fills the params, and starts the loop.

## A worked generation

Suppose the metric is `val_bpb` (minimize) and VRAM is cost-guarded. The agent logs the two baseline runs, then generation 1:

```bash
python3 amr.py log --cost vram=44.0 0 base000 - 0.998100 ran "baseline run 1"
python3 amr.py log --cost vram=44.0 0 base000 - 0.997900 ran "baseline run 2"

python3 amr.py log --cost vram=44.2 --region optimizer    1 lr00400 base000 0.993200 ran "increase LR to 0.04"
python3 amr.py log --cost vram=44.1 --region architecture 1 gelu000 base000 0.996000 ran "replace SiLU with GeLU"
python3 amr.py log --cost vram=80.0 --region architecture 1 wide000 base000 -        crash "double model width OOM"
```

The baseline spread is 0.0002, so that is the margin. Reduce:

```bash
$ python3 amr.py reduce --gen 1 --beam 2 --margin 0.0002 --cost vram:+0.15
gen 1: 2 candidates, 2 beat their parent (margin 0.0002, 0 failed cost guards), keeping 2
KEEP    lr00400  0.993200  (parent base000, delta -0.004700)  increase LR to 0.04     regions=optimizer
KEEP    gelu000  0.996000  (parent base000, delta -0.001900)  replace SiLU with GeLU  regions=architecture
FRONTIER: lr00400 gelu000
FUSE_CANDIDATE  lr00400  gelu000  regions=architecture,optimizer
```

Both survivors advance, and because they touch disjoint regions, `reduce` suggests trying their combination as one extra candidate next generation. Generation 2 now branches from `lr00400` and `gelu000`, each child compared against its own parent. At any point:

```bash
$ python3 amr.py tree
base000  0.997900  baseline run 1
├── lr00400  0.993200  increase LR to 0.04
├── gelu000  0.996000  replace SiLU with GeLU
└── wide000  crash  double model width OOM
```

## amr.py

```bash
python3 amr.py log [--cost name=value] [--region name] <gen> <commit> <parent> <score|-> <ran|crash|note> <description...>
python3 amr.py reduce --gen <N> --margin <M> [--beam <B>] [--maximize] [--cost name:+tol]
python3 amr.py tree [--maximize]
```

**log** appends one row to `results.tsv`. Status `ran` requires a numeric score; `crash` records a failure with `-` as the score; `note` records a law, a written-down negative result such as `"LAW: wider models OOM on this GPU"`. Crashes and laws are results too: they stop the agent from re-proposing pruned families.

**reduce** selects the frontier for a generation:

- Every candidate is compared to its own parent, never to a global baseline.
- `--cost vram:+0.15` means a candidate must also hold its logged `vram` within 15% of its parent's; violators print `COST_FAIL` and are dropped even if their score wins.
- Survivors are ranked globally and the top `--beam` (default 2) print as `KEEP` lines plus a `FRONTIER:` line.
- If two kept survivors carry disjoint `--region` tags, reduce prints `FUSE_CANDIDATE`: try that combination next generation. It competes like any other candidate, and combined winners often don't add.
- If no candidate beats its parent by more than the margin, reduce prints `STALL` and re-lists the parents as the unchanged frontier. The program's response to a stall is to record a law and propose from new hypothesis families, not to retry a pruned family with tweaks.

**tree** prints the whole search tree with each node's best score (crashes included), followed by any recorded laws.

All state lives in `results.tsv` (append-only) and git. A crashed or interrupted run resumes from those two alone.

The loop is domain-agnostic. [example/prompt-optimization.md](example/prompt-optimization.md) tunes a prompt file (workers edit `prompt.md`, reduce with `--maximize`), and [example/text-classification.md](example/text-classification.md) has the agent propose whole different algorithms (lexicon vs Naive Bayes vs logistic regression) and keep whichever the eval prefers. Both are full multi-generation walkthroughs.

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

[MIT](LICENSE)
