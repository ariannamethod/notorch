# Chuck: learned actions and future consequences

Research date: 2026-10-07. The sources below supply experiments to learn from.
Chuck's lineage remains WOLFE's finite executable language, Netta's acquired
experience, and Chuck's persistent training state. The cited experiments were
read, not rerun. Chuck's own measurements are linked separately.

## Experiments worth carrying forward

| Primary source | Actual experiment and observed boundary | Question it gives Chuck |
| --- | --- | --- |
| [Andrychowicz et al., 2016](https://arxiv.org/pdf/1606.04474), §§2–3.2 | Coordinatewise LSTM; 100-step training episodes with 20-step truncated credit. MNIST tests change width/depth/activation and run 200 steps. Width/depth transfer works in the reported settings; sigmoid-to-ReLU transfer fails. | Do acquired choices transfer across bodies and beyond the horizon that taught them? |
| [Wichrowska et al., 2017](https://proceedings.mlr.press/v70/wichrowska17a/wichrowska17a.pdf), §§4–5, Appendix C | Hierarchical recurrent optimizer learns on synthetic landscapes; neural networks are absent from that training set. ImageNet networks initially improve, then progress fails after approximately 10K–20K updates. | Include later and stagnant states. Early improvement and sustained continuation are separate measurements. |
| [Metz et al., 2022](https://proceedings.mlr.press/v199/metz22a/metz22a.pdf), Figure 6, Appendices A/D | A compact MLP has **197 meta-parameters**. Capacity sweeps and cross-task tests expose a compute/capacity tradeoff. Further meta-training can improve the training body while degrading transfer. | Keep Chuck's 163 parameters for the first future-credit experiment. Save the parent before adaptation and measure both lives on untouched runs. |
| [Vicol et al., 2021](https://proceedings.mlr.press/v139/vicol21a/vicol21a.pdf), §§5.1–5.2 | Persistent Evolution Strategies carries perturbation history through partial unrolls. A toy problem gives wrong-sign credit with short truncation; the learned-optimizer test uses 4-step truncations within a 1,000-step horizon. | Persistent state and useful credit horizon are independent requirements. Test the sign of credit on an actual ranking reversal. |
| [AutoLoss, 2018](https://arxiv.org/pdf/1810.02442), §§4–5, Appendix A | A controller selects executable training operations. Its GAN controller is a linear Bernoulli model; task outcomes teach the caller. Tests include feature ablations, fixed schedules and architecture transfer. Synthetic experiments separate controller and target training/validation, plus final test data. | A small action caller is enough to start. Keep policy experience separate from the states used to evaluate acquired choices. |
| [AutoLRS, ICLR 2021](https://arxiv.org/pdf/2105.10762), Algorithm 1, Appendix A.5.4 | Every LR candidate starts from the same model **and optimizer** checkpoint. Short candidate runs forecast a longer validation horizon. Removing forecasting changes late LR choices and worsens final results. Reported training-step speedups exclude search work. | Restore the whole training world before comparing actions. Record all branch computation. Our H16 labels are measured directly. |
| [Wu et al., ICLR 2018](https://arxiv.org/pdf/1803.02021), §§2–3 | Noisy-quadratic and neural-network tests isolate short-horizon bias. Short lookahead favors overly small rates in their setting. Reusing one minibatch through lookahead reverses the problem toward large unstable rates. | Compare horizons on the same probe, and probes at the same horizon. Preserve the actual future window stream. Their rate direction differs from Chuck's measured immediate PUSH preference. |
| [Dynamic Algorithm Configuration, ECAI 2020](https://ecai2020.eu/papers/1237_paper.pdf), §§5–7 | Synthetic Luby/Sigmoid environments compare state-conditioned policies, a time-only schedule and static actions. Different instances requiring opposite actions defeat an instance-blind schedule. | A constant-action reference tests whether choices add value; a later history/time-only ablation can test which observation information matters. |
| [Joulani et al., ICML 2013](https://proceedings.mlr.press/v28/joulani13.pdf), Algorithm 2 | Delayed-feedback theory associates arriving outcomes with originating actions and maintains action-specific queues. Its stochastic assumptions are explicit. | Preserve original decision identity and exact-once credit when adding an online delayed episode. |
| [Celo2, 2026](https://arxiv.org/pdf/2602.19142), §§3–5, Appendix A | Small image-MLP tasks teach a compact policy with 50-step truncations and 100–2,000-step runs. Tests include language models, vision and Atari. Width and input/output normalization receive ablations; the main system also uses matrix orthogonalization and AdamW components. | Test scale sensitivity before changing Chuck's observation normalization or bounded action vocabulary. |
| [ELO, 2026 preprint](https://arxiv.org/html/2607.06772v4), Figure 2, Appendix D | A resume buffer returns to difficult inner states. Buffer-only training becomes unstable; progressive expert supervision is a separate tested component. Base runs span 100–2,000 steps, with 50-step truncations and longer evaluation. | Keep difficult saved worlds as experience. Increasing horizon is its own experiment, with its own failures. |

## The current Chuck experiment

The [common-state measurements](../experiments/chuck_loss_architect/scenarios/README.md)
already supply the question: immediate own-window loss prefers PUSH in all
16 development states; H16 future probes prefer BRAKE in four and tied
HOLD/BRAKE in three. The headline comparison changes both horizon and probe.
The retained H1/H4/H16 matrix also permits each factor to be examined separately.

The [fixed future-credit protocol](../experiments/chuck_loss_architect/future/protocol.json)
uses one selected intervention followed by 15 HOLD updates. All three branches
start from the same complete state and are evaluated on the same four future
windows. Each head learns

`clamp((future_loss[HOLD] - future_loss[action]) / (abs(future_loss[HOLD]) + 1e-6), -1, 1)`.

This is our experiment design, informed by the readings and Chuck's receipts.
The C implementation fits mean three-head Huber loss with simultaneous gradients.
Eight SimpleLLM states teach one saved life; a copy then receives eight HeVLM
states. Both stages run the same preregistered 512 epochs at LR .03. In the
final run, both lives are frozen before seeds 101 and 211 generate outcomes.
A preceding runner smoke already exposed checkpoint 1 of seed 101 on both
bodies. The fixed settings receive no tuning from it; seed 211 supplies the
separate untouched-run confirmation. The full evaluation retains both seeds.

Initial, Simple-trained, adapted, original immediate-feedback, always-HOLD and
always-PUSH readouts see identical source histories. Exact outcome ties remain
ties. Full branch computation and readout regret are recorded separately from
the host's final held-out loss. The original host continues with its existing
online policy, and independent control/probe processes verify that returning
from diagnostic forks preserves every host step and final byte.

## What follows from the result

Replay tests acquisition and transfer on measured source contexts. A subsequent
closed-loop test can execute non-overlapping episodes with the same intervention
and HOLD tail, preserving the identity of the choice until its consequence
arrives. Repeated interventions at every step define a new continuation and
need new measurements. A predeclared longer-horizon subset, reversed body
transfer, and time/history ablations each answer a different next question.

Keep the failure receipts. A fitted policy can change its choices and still
lose against a constant action on untouched states. That result identifies
what its next experience must teach.
