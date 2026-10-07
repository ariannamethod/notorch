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
