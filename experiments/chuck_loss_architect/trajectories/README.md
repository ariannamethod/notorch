# Chuck learns from the worlds his actions reach

The preceding [lived-state run](../lived/README.md) taught two lives from the
parent's visited states. Both regressed against that parent in all four full
deployments. Their action histories show earlier braking, with the policy-taught
student reaching HeVLM's dampening floor at update 61 on both new seeds.

This [fixed protocol](protocol.json) compares experience acquired on that
student's own trajectory with an equal fitting budget on parent trajectories.
Both new lives start from the same sealed `lived-policy` weights. One receives
32 parent-reached worlds; the other receives 32 student-reached worlds. In every
target branch, the same frozen student supplies actions after the first
HOLD/BRAKE/PUSH intervention. Source worlds are the experimental difference.

SimpleLLM (450,688 parameters) and HeVLM (1,123,456) retain their pinned corpora,
F32 arithmetic, context 64, body LR .0003, clip 1 and two SIMD threads. Development
seeds 42/73 supply checkpoints before updates 1, 17, 33, 49, 65, 97, 129 and 193.
Their spacing preserves every sixteen-step source-continuation check. Both
163-parameter refits receive 512 epochs of the existing native conditioned Huber
objective at LR .03: 16,384 fits each. H16 future-four-window loss supplies the
target; held-out loss is recorded separately.

All lives are sealed before seeds 1013/1217. Common-state readouts use student
worlds and the shared student continuation. Six complete 512-step deployments
per body/seed compare canonical Chuck, HOLD, PUSH, the initial student,
`refit-parent` and `refit-self`. Four save/load repeats check continuation.
Full curves, early actions, first dampening-floor entry and regressions remain
in the receipt. This is one acquisition round with the existing policy capacity.

The experimental `--trajectory` host separates the source life from the future
policy. It records both original and grafted branch identities, advances actual
history, and restores the complete source. Its source-policy branch reproduces
the ordinary host's next sixteen transitions. Equal source/continuation weights
produce one measured table with an explicit alias. Ordinary `--lived`, canonical
Chuck and the public numerical APIs retain their existing behavior.

Reproduction uses the authenticated raw lived-state experiment as input:

```sh
python3 experiments/chuck_loss_architect/trajectories/run.py \
  --previous-run /path/to/lived-raw --output /path/to/new-trajectories
```

The fixed full run contains 26,624 body updates, 32,768 policy fits and 576
readouts. Development smoke uses seed 42, two checkpoints, two fit epochs,
17-step source hosts and 32-step deployments. Its receipts remain separate.

## Closed artifacts retain their first identity

The first development smoke stopped when a later read found three previously
parsed traces truncated. The failed directory remains unchanged. Its cause is
undetermined; no full acquisition or new evaluation had started.

The runner now seals each closed file from its first bytes: those same bytes
supply parsing, an immutable hash and an atomic compressed mirror. A later
mismatch preserves the damaged file and failed-integrity receipt. Recovery is
allowed once per file, only from mirror bytes matching the original hash.
A missing or incorrect mirror stops the run. Every restoration is reported;
body updates, fitting and measured outcomes are never rerun for recovery.

## Measured round: source `ac5d651`

The full run completes once from clean commit
`ac5d651d41efa68e79b729a4e3b05058e0161d47` under protocol
`86673afd0abc04b5f389aeb4400765287fbc9a0951963639d961fc70a9c6c121`.
Both new lives start from the same initial student. Here, `refit-self` means
experience reached by that initial student, before refitting; this round does
not feed the newly refitted lives' deployment results back into training.
Both fitted lives are sealed before any seed-1013/1217 world is generated.

The C hosts use F32, two SIMD threads, context 64, LR .0003 and clipping at 1
on AMD EPYC 9V74 / Linux x86_64 / GCC 13.3.0. The process sees nine CPUs with
an eight-CPU quota and an 8 GiB memory limit. The pinned Dracula and Hebrew
corpora, original repository commits, token hashes and 450,688 / 1,123,456
parameter bodies are recorded in [results.json](results.json). These are CPU
measurements. No GPU execution is claimed.

### Actions at the same visited states

Each row covers sixteen new worlds on the initial student's trajectory. The
three first actions share that student's continuation for the remaining fifteen
updates. H16 future-four-window loss determines regret; exact ties count as
optimal. These are measured intervention outcomes, separate from full deployment.

| Body | Readout | Optimal / 16 | Mean H16 regret |
|---|---|---:|---:|
| SimpleLLM | Initial student | 13 | 0.000327423 |
| SimpleLLM | Refit parent | 7 | 0.000294730 |
| SimpleLLM | Refit self | 10 | 0.000216872 |
| SimpleLLM | PUSH | 10 | 0.000379935 |
| HeVLM | Initial student | 5 | 0.001149826 |
| HeVLM | Refit parent | 12 | 0.000038907 |
| HeVLM | Refit self | 12 | 0.000038907 |
| HeVLM | PUSH | 12 | 0.000040784 |

Both refits lower mean regret on both bodies. SimpleLLM's optimal-choice count
falls even as mean regret improves. On HeVLM, both refits choose the same actions
in all sixteen common worlds. All six readouts, including HOLD and BRAKE,
and the development comparisons remain in [common_state.csv](common_state.csv).

### Complete 512-step deployments

Final held-out loss, lower is better. Each row uses the same body initialization,
data windows and seed across all six arms. This repeats actions through the
states each life actually reaches; it is not a sum of common-state regrets.

| Body / seed | Canonical | HOLD | PUSH | Initial student | Refit parent | Refit self |
|---|---:|---:|---:|---:|---:|---:|
| SimpleLLM / 1013 | 2.50756502 | 2.57181811 | 2.49165535 | 2.58611774 | 2.58488512 | 2.60762620 |
| SimpleLLM / 1217 | 2.52432513 | 2.57725716 | 2.52897191 | 2.58628678 | 2.57601786 | 2.58855104 |
| HeVLM / 1013 | 1.57190084 | 1.57658267 | 1.57234776 | 1.58259857 | 1.57894039 | 1.57811940 |
| HeVLM / 1217 | 1.56940532 | 1.55563200 | 1.58516204 | 1.58714998 | 1.57969737 | 1.58079970 |

Parent-source refitting improves the initial student in all four pairs.
Student-source refitting improves both HeVLM seeds and regresses both SimpleLLM
seeds. Against the equal-budget parent-source refit, the student-source refit
wins only HeVLM/1013. Canonical Chuck finishes below every learned life in these
four pairs. Constant PUSH wins SimpleLLM/1013; constant HOLD wins HeVLM/1217.
The [paired differences](paired_final.csv) and
[five-point held-out curves](heldout_curves.csv) retain the exact measurements.

### Where acquired actions separate the worlds

Both refits first select PUSH where the initial student selects BRAKE at
SimpleLLM updates 37/29 and HeVLM updates 3/3 (seeds 1013/1217). All eight first
changes have identical features, observation, pre-action Chuck state and loss.
Different acquired weights therefore change the selected action before the
body trajectories separate.

| Body / seed | Initial student's first .3 dampening | Refit parent | Refit self |
|---|---:|---:|---:|
| SimpleLLM / 1013 | 260 | 288 | 194 |
| SimpleLLM / 1217 | 236 | 283 | 198 |
| HeVLM / 1013 | 55 | 127 | 163 |
| HeVLM / 1217 | 56 | 144 | 162 |

The student-source refit takes fewer BRAKE actions in the first 128 SimpleLLM
updates, yet subsequently reaches the dampening floor earlier and finishes
with more BRAKE actions over the whole run. Its HeVLM trajectory delays the
floor and improves over the initial student. Early action counts alone would
miss that difference. [deployment.csv](deployment.csv) records both early and
complete action counts, first divergence and first floor entry.

This round changes the source of acquired experience while holding the initial
weights, continuation, capacity and fit budget fixed. Its result is mixed:
experience from the student's reached worlds does not consistently improve
its next full trajectory. The next question is how repeated acquisition and
the duration of accumulated action effects change that result. The current
outcomes have not been used to adjust or repeat this fixed round.

### Receipts and gates

The run executes 64 acquisition worlds, 32 new evaluation worlds, 384 branches,
26,624 body updates, 32,768 policy fits and 576 readouts. All 1,536 selected-source
transitions, 3,072 probe-off/on host comparisons, four 256-step source/deployment
prefixes and four complete saved-life continuations are exact. Terminal
verification authenticates 1,081 artifact identities. The final seal additionally
includes the terminal receipt and results. No integrity failure or restoration
occurs in the full run; the failed first smoke and the second smoke's two
restorations remain separate.

The independent [audit.py](audit.py) passes 135,630 checks on 322 selected original
artifacts, reconstructing target association, fit chronology, source/grafted
history, action credit, readout ties and complete deployment comparisons:

```sh
python3 experiments/chuck_loss_architect/trajectories/audit.py \
  --run /path/to/new-trajectories --out /path/to/new-audit \
  --source-commit ac5d651d41efa68e79b729a4e3b05058e0161d47 \
  --results-sha256 99b7f72453ecc36e420e1aa7a13048c1889d7ec6fa717b12b889e09682751416
```

Integration retains all eighteen measured source-file hashes and all eleven
non-shared files from main `0b0444a`. The first combined static/shared build
exposes archive consumers accidentally selecting `libnotorch.so`. Five CPU
consumers now link their declared archives explicitly. All 49 default CPU recipe
commands then pass with the shared library present; the five rebuilt consumers
have no dynamic `libnotorch.so` dependency. The original failure and corrected
build receipts remain in [integration_verification.json](integration_verification.json).

[archives.json](archives.json) records lossless part identities and reconstruction
instructions for the 3,292,477,369-byte full record (2,192 members) and the
1,198,704,640-byte validation/smoke record. The full archive verifies all 1,083
final original identities and every corresponding compressed mirror. Its SHA-256
is `187be0ac0f45557285a28b3a4e3708e03730dbfa3d6ad7c7408c798149e36474`.
The first packaging attempt stopped on disk exhaustion; its failure receipt
is retained. Packaging did not repeat or change numerical work.
