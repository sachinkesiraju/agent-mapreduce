# example: agentic map-reduce over solution strategies

The other two examples vary settings inside one approach: hyperparameters in
`train.py`, wording in `prompt.md`. This one is different. Here each idea is a
whole **algorithm**, and the agent writes it from scratch. The target is a
`solver.py` that classifies text; competing ideas are a hand-tuned lexicon, a
Naive Bayes model, a logistic regression trained from scratch, and so on. The
loop is identical; only the size of the thing that changes per idea is bigger.

This is the mode to reach for when the question is "which approach wins?"
rather than "what's the best setting for this approach?". You do not tell the
agent which algorithm to use. It proposes several, implements each in its own
worktree, and the eval decides.

## The fixed harness

Two files the agent never edits. A seeded dataset generator (run once) and a
fixed evaluator. `solver.py` is the editable target; `eval.py` trains it on
`train.tsv` and scores it on a held-out split.

`gen_data.py` builds `train.tsv`, `test.tsv`, and `holdout.tsv` with real
signal, 8% label noise, and negation traps ("not terrible" is positive):

```python
import random
POS = ["great","love","excellent","wonderful","best","amazing","brilliant","enjoyed"]
NEG = ["terrible","hate","awful","worst","boring","bad","disappointing","poor"]
FILLER = ["the","movie","was","film","really","quite","very","i","it","this","story","plot"]

def make(n, seed):
    r = random.Random(seed); rows = []
    for _ in range(n):
        label = r.choice(["pos","neg"])
        lex = POS if label == "pos" else NEG
        opp = NEG if label == "pos" else POS
        words = r.sample(lex, r.randint(2,3)) + r.sample(FILLER, r.randint(3,5))
        if r.random() < 0.20:      words += ["not", r.choice(opp)]   # negation trap
        elif r.random() < 0.15:    words.append(r.choice(opp))       # contamination
        r.shuffle(words)
        if r.random() < 0.08:      label = "neg" if label == "pos" else "pos"  # noise
        rows.append((label, " ".join(words)))
    return rows

for name, n, seed in [("train",400,1),("test",150,2),("holdout",150,3)]:
    with open(f"{name}.tsv","w") as f:
        for lab, txt in make(n, seed): f.write(f"{lab}\t{txt}\n")
```

`eval.py` reloads whatever `solver.py` the worker wrote, trains it, and prints
the metric. This is the ground truth; the agent grabs the score from here, not
from the worker:

```python
import sys, importlib, solver

def load(p):
    with open(p) as f:
        return [line.rstrip("\n").split("\t",1) for line in f if line.strip()]

data = sys.argv[1] if len(sys.argv) > 1 else "test.tsv"
importlib.reload(solver)
solver.train(load("train.tsv"))
rows = load(data)
acc = sum(solver.predict(txt) == lab for lab, txt in rows) / len(rows)
print(f"accuracy: {acc:.6f}")
```

The baseline `solver.py` is deliberately weak, a two-word lexicon, so the first
generation has room to move:

```python
POS = {"great", "love"}; NEG = {"terrible", "hate"}
def train(rows): pass
def predict(text):
    t = text.split()
    return "pos" if sum(w in POS for w in t) - sum(w in NEG for w in t) >= 0 else "neg"
```

## Parameters

| param        | value |
|--------------|-------|
| eval_cmd     | `python3 eval.py > run.log 2>&1` |
| metric_grep  | `grep '^accuracy:' run.log` |
| metric       | accuracy — maximize |
| K            | 4 |
| B            | 2 |
| eval_slots   | 4 — the eval is pure-Python and cheap, so all candidates can run at once |
| timeout      | 2 minutes |
| holdout_cmd  | `python3 eval.py holdout.tsv > holdout.log 2>&1` |
| tag          | date-based, e.g. `jul8-clf` |

The metric's resolution is one test example, 1/150 ≈ 0.0067, so the noise
margin floors there: accuracy differences smaller than a single flipped
example are not signal.

## Domain rules

- The agent may rewrite `solver.py` however it likes, as long as it keeps the
  `train(rows)` and `predict(text)` interface. A whole new algorithm per idea
  is the point.
- `eval.py`, `gen_data.py`, and the three `.tsv` files are read-only.
- One algorithm family per idea. Tag the region: `lexicon`, `model`,
  `negation`, `features`.
- The champion must beat the baseline on `holdout.tsv`, not just `test.tsv`.

## What a real run looked like

Baseline lexicon scored 0.706667. From that single frontier node the agent
proposes four ideas, each a different hypothesis about what drives the label
and each in its own region:

- **full lexicon** (`lexicon`): the baseline only knows four words; score with
  all sixteen.
- **naive bayes** (`model`): stop hand-picking words, let the training counts
  weight every token.
- **negation-aware lexicon** (`negation`): flip a word's polarity after "not"
  or "never" to catch the traps.
- **logistic regression** (`model`): learn per-word weights by gradient descent
  instead of assuming them.

One worker per idea rewrites `solver.py` in its own worktree, commits, and runs
the eval. The orchestrator greps each score from `run.log` and logs it:

```bash
python3 amr.py log --region lexicon  1 fulllex  base 0.920000 ran "full 16-word lexicon vote"
python3 amr.py log --region model    1 nbayes   base 0.920000 ran "naive bayes, laplace smoothing"
python3 amr.py log --region negation 1 negation base 0.866667 ran "negation-aware lexicon"
python3 amr.py log --region model    1 logreg   base 0.860000 ran "logistic regression from scratch"
```

```bash
$ python3 amr.py reduce --gen 1 --beam 2 --margin 0.0067 --maximize
gen 1: 4 candidates, 4 beat their parent (margin 0.0067), keeping 2
KEEP    fulllex  0.920000  (parent base, delta +0.213333)  full 16-word lexicon vote        regions=lexicon
KEEP    nbayes   0.920000  (parent base, delta +0.213333)  naive bayes, laplace smoothing   regions=model
FRONTIER: fulllex nbayes
FUSE_CANDIDATE  fulllex  nbayes  regions=lexicon,model
```

The interesting result is what lost. The negation-aware classifier, the
"smart" idea, scored *below* the plain lexicon. The dataset shuffles word
order, so "not" rarely stays next to the word it should flip, and the handler
mostly mis-fires. A greedy loop that tried negation first might have anchored
on it; comparing siblings against the same parent caught it. The two survivors
sit in disjoint regions, so reduce suggests an ensemble.

Generation 2 recurses. The agent takes each surviving branch and proposes new
ideas *from that parent*, not from the baseline: the `fulllex` line gets a
data-mined lexicon, the `nbayes` line gets bigrams and stopword removal, and
the fuse candidate reduce suggested becomes its own branch off `nbayes`. Note
the parent commit in each row, that is the recursion:

```bash
python3 amr.py log --region lexicon,model 2 ensemble nbayes  0.920000 ran "ensemble: full lexicon + naive bayes"
python3 amr.py log --region lexicon       2 minedlex fulllex 0.920000 ran "data-mined log-odds lexicon"
python3 amr.py log --region model         2 nbbigram nbayes  0.900000 ran "naive bayes + bigrams"
python3 amr.py log --region model         2 nbstop   nbayes  0.920000 ran "naive bayes minus stopwords"
```

```bash
$ python3 amr.py reduce --gen 2 --beam 2 --margin 0.0067 --maximize
gen 2: 4 candidates, 0 beat their parent (margin 0.0067), keeping 0
STALL: no candidate beat its parent by more than the margin. Frontier unchanged.
FRONTIER: nbayes fulllex
```

The ensemble, the data-mined lexicon, and stopword removal all tied 0.92;
bigram features scored 0.90 and hurt. That flat result is the signal: with 8%
label noise, ~0.92 is the ceiling. The honest move is to record it, not to
keep tuning into the noise:

```bash
python3 amr.py log 2 - - - note "LAW: gen-2 refinements all sit at ~0.92 = the 8% label-noise ceiling; no headroom above the gen-1 lexicon/NB tie"
```

The whole search, two generations deep, is one tree:

```text
$ python3 amr.py tree --maximize
base  0.706667  baseline run 1
├── fulllex  0.920000  full 16-word lexicon vote
│   └── minedlex  0.920000  data-mined log-odds lexicon
├── nbayes  0.920000  naive bayes, laplace smoothing
│   ├── ensemble  0.920000  ensemble: full lexicon + naive bayes
│   ├── nbbigram  0.900000  naive bayes + bigrams
│   └── nbstop  0.920000  naive bayes minus stopwords
├── negation  0.866667  negation-aware lexicon
└── logreg  0.860000  logistic regression from scratch
```

The two surviving branches each spawned their own children; the pruned
`negation` and `logreg` ideas stay in the tree as dead ends, never recursed
into. On the holdout split both survivors scored 0.946667 against the
baseline's 0.673333, so the win is real, not overfit to the test set. The
lexicon and Naive Bayes tie, so the simplicity criterion ships the lexicon:
six lines, no training.

## Worker prompt template

> You are one shard of a map-reduce experiment. Your worktree: `<path>`.
> Rewrite `solver.py` to implement exactly this one approach and nothing else:
> "<idea>: <rationale>". Keep the `train(rows)` and `predict(text)` interface.
> Commit with a one-line message. Then run `python3 eval.py > run.log 2>&1`
> and stop. Do not read or report the score. Do not touch `eval.py`,
> `gen_data.py`, or any `.tsv` file.
