#!/usr/bin/env python3
"""Read-only post-result diagnosis of SPA's frozen conditioned experiment."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
PREFIX = 'experiments/spa_agent/conditioned/'
RECEIPT_SHA = '398ba12f745f38e3365dc184fa49af01538d2e1b8ba6e70981eaf378330ee3d2'
ARCHIVE_SHA = '595bf9f75e68f1b8fe562bc2bbb68c3908d71a4817205d2684714657a2f68dca'
ARMS = ('initial', 'raw_mean8', 'conditioned_mean8', 'shuffled_conditioned')
SOURCE_KEYS = {'agent_rng', 'body_hash', 'episode', 'feature_witness', 'host_rng',
               'life_hash', 'ordinal', 'ordinary_decision_raw_sha256', 'seed',
               'sentence_count', 'snapshot', 'snapshot_raw_sha256',
               'snapshot_witness', 'source_archive_sha256', 'step', 'target'}
FLOOR = struct.unpack('<f', struct.pack('<f', .001))[0]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def source(name):
    path = ROOT / (PREFIX + name)
    return path.read_bytes() if path.exists() else subprocess.check_output(
        ['git', 'show', 'HEAD:' + PREFIX + name], cwd=ROOT)


def load():
    raw = source('receipts.json')
    require(digest(raw) == RECEIPT_SHA, 'receipt SHA')
    receipts = json.loads(raw)
    archive = b''.join(source(f'raw_records.tar.gz.part-{i:02d}') for i in range(3))
    require(digest(archive) == ARCHIVE_SHA, 'archive SHA')
    selected = {}
    pins = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tf:
        names = [m.name for m in tf.getmembers()]
        require(len(names) == len(set(names)), 'duplicate archive member')
        for split in ('training', 'evaluation'):
            for suffix in ('.json', '.readout-1.jsonl', '.readout-2.jsonl'):
                name = split + suffix
                member = tf.getmember(name)
                require(member.isfile(), 'member type')
                data = tf.extractfile(member).read()
                expected = receipts['archive']['members'][name]
                require(len(data) == expected['bytes'] and digest(data) == expected['sha256'],
                        'member identity: ' + name)
                selected[name] = data
                pins[name] = expected
            require(selected[split + '.readout-1.jsonl'] == selected[split + '.readout-2.jsonl'],
                    'independent readout copies')
    datasets = {s: json.loads(selected[s + '.json'])['samples'] for s in ('training', 'evaluation')}
    readouts = {s: [json.loads(line) for line in selected[s + '.readout-1.jsonl'].splitlines()]
                for s in datasets}
    return receipts, datasets, readouts, pins


def validate(split, receipts, samples, readouts):
    require(len(samples) == 48 and len(readouts) == 192, 'complete split')
    sample_map = {(r['seed'], r['snapshot']): r for r in samples}
    require(len(sample_map) == 48, 'duplicate source')
    rows = [r for r in receipts[split]['state_comparisons'] if r['horizon'] == 4]
    comparisons = {(r['seed'], r['snapshot'], r['label']): r for r in rows}
    require(len(comparisons) == len(rows), 'duplicate comparison')
    joined = {}
    for r in readouts:
        require(r['arm'] in ARMS, 'unknown arm')
        require(set(r['source']) == SOURCE_KEYS, 'complete source witness')
        key = (r['source']['seed'], r['source']['snapshot'])
        require(key in sample_map, 'unknown readout source')
        sample = sample_map[key]
        require(all(sample.get(k) == v for k, v in r['source'].items()), 'source witness join')
        require((key, r['arm']) not in joined, 'duplicate readout')
        mask = sample['action_mask']
        target = sample['target']
        expected_mask = 1 | (2 if target > 0 else 0) | (4 if target + 1 < sample['sentence_count'] else 0)
        require(mask == expected_mask and mask == r['action_mask'], 'action mask')
        valid = [k for k in range(3) if mask & (1 << k)]
        scores = r['scores']
        require(len(scores) == 3 and all(math.isfinite(v) for v in scores), 'finite scores')
        require(all(v == f32(v) for v in scores), 'native float scores')
        raw = bytes.fromhex(r['readout_hex'])
        require(len(raw) == 28, 'readout length')
        kind, raw_target, raw_source, raw_mask, *raw_scores = struct.unpack('<4I3f', raw)
        action = r['action']
        require(raw_scores == scores and raw_mask == mask and kind == action['kind']
                and raw_target == target and action['target'] == target, 'readout byte witness')
        chosen = max(valid, key=lambda k: (scores[k], -k))
        require(chosen == kind, 'native masked argmax')
        neighbor = None if kind == 0 else target - 1 if kind == 1 else target + 1
        require(action['source'] == neighbor and raw_source == (0xffffffff if neighbor is None else neighbor),
                'action neighbor')
        c = comparisons[key + (r['arm'],)]
        initial = comparisons[key + ('initial',)]
        require(c['kind'] == kind and c['source_snapshot_sha256'] == sample['snapshot_raw_sha256']
                and c['target'] == target, 'comparison join')
        require(c['actions'] == initial['actions'], 'common consequences')
        require([a['kind'] for a in c['actions']] == valid, 'consequence action set')
        keep_mean = c['actions'][0]['mean_reward']
        for a in c['actions']:
            draws = a['replicate_advantages']
            require(len(draws) == 8 and all(math.isfinite(v) for v in draws), 'paired draws')
            require(abs(statistics.mean(draws) - a['mean_advantage_over_keep']) < 1e-14,
                    'paired mean')
            require(abs(a['mean_reward'] - keep_mean - a['mean_advantage_over_keep']) < 1e-14,
                    'reward and advantage join')
            se = statistics.stdev(draws) / math.sqrt(8)
            require(abs(se - a['paired_standard_error']) < 1e-14, 'paired standard error')
            require(a['positive_replicates'] == sum(v > 0 for v in draws)
                    and a['negative_replicates'] == sum(v < 0 for v in draws), 'paired signs')
        joined[(key, r['arm'])] = (sample, r, c)
    require(set(joined) == {(key, arm) for key in sample_map for arm in ARMS}, 'complete arm/source product')
    return joined


def distribution(values):
    return {'count': len(values), 'min': min(values), 'median': statistics.median(values),
            'max': max(values), 'mean': statistics.mean(values)} if values else {'count': 0}


def huber(error):
    e = abs(error)
    return .5 * e * e if e <= 1 else e - .5


def analyze(joined):
    result = {}
    for arm in ARMS:
        details = []
        state_losses = []
        choices = [0, 0, 0]
        for (key, label), (sample, readout, c) in sorted(joined.items()):
            if label != arm:
                continue
            scores = readout['scores']
            means = {a['kind']: f32(a['mean_reward']) for a in c['actions']}
            targets = {k: f32(v - means[0]) for k, v in means.items()}
            scale = max(FLOOR, max(abs(v) for v in targets.values()))
            conditioned = {k: f32(v / scale) for k, v in targets.items()}
            choices[readout['action']['kind']] += 1
            state_losses.append(statistics.mean(huber(scores[k] - conditioned[k]) for k in means))
            for a in c['actions']:
                k = a['kind']
                if not k:
                    continue
                draws = a['replicate_advantages']
                loo = [(sum(draws) - v) / 7 for v in draws]
                details.append({'seed': key[0], 'snapshot': key[1], 'target': sample['target'],
                    'action': k, 'selected': readout['action']['kind'] == k,
                    'score_margin_over_keep': scores[k] - scores[0],
                    'raw_mean_advantage': a['mean_advantage_over_keep'],
                    'conditioned_target': conditioned[k],
                    'score_minus_conditioned_target': scores[k] - conditioned[k],
                    'paired_standard_error': a['paired_standard_error'],
                    'positive_draws': a['positive_replicates'], 'negative_draws': a['negative_replicates'],
                    'leave_one_draw_out_positive_means': sum(v > 0 for v in loo),
                    'leave_one_draw_out_mean_range': [min(loo), max(loo)]})
        positives = [r for r in details if r['raw_mean_advantage'] > 0]
        groups = {}
        for name, values in [('all', details), ('positive_mean', positives),
                             ('nonpositive_mean', [r for r in details if r['raw_mean_advantage'] <= 0]),
                             ('target_3_cost_only', [r for r in details if r['target'] == 3])]:
            groups[name] = {'alternatives': len(values),
                'positive_score_margins': sum(r['score_margin_over_keep'] > 0 for r in values),
                'negative_score_margins': sum(r['score_margin_over_keep'] < 0 for r in values),
                'zero_score_margins': sum(r['score_margin_over_keep'] == 0 for r in values),
                'selected': sum(r['selected'] for r in values),
                'score_margin': distribution([r['score_margin_over_keep'] for r in values]),
                'raw_mean_advantage': distribution([r['raw_mean_advantage'] for r in values])}
        result[arm] = {'choices_keep_left_right': choices, 'groups': groups,
            'mean_huber_against_associated_conditioned_targets': statistics.mean(state_losses),
            'positive_mean_above_one_paired_se': sum(r['raw_mean_advantage'] > r['paired_standard_error'] for r in positives),
            'positive_mean_above_two_paired_se': sum(r['raw_mean_advantage'] > 2*r['paired_standard_error'] for r in positives),
            'positive_mean_sign_survives_all_leave_one_out_draws': sum(r['leave_one_draw_out_positive_means'] == 8 for r in positives),
            'positive_alternatives': positives,
            'alternatives': details}
    return result


def feature_support(datasets, joined):
    training = datasets['training']
    evaluation = datasets['evaluation']
    bounds = [(min(r['features'][i] for r in training), max(r['features'][i] for r in training))
              for i in range(29)]
    result = {'definition': 'Coordinate min/max coverage of the captured 29-feature vectors; a descriptive box, not a learned support model.',
              'history_coordinates': {'23': 'LEFT action frequency', '26': 'last action was LEFT'},
              'splits': {}}
    for split, samples in datasets.items():
        require(all(len(s['features']) == 29 and all(math.isfinite(v) for v in s['features']) for s in samples),
                'finite feature vectors')
        changed = []
        missed = []
        outside = []
        for sample in samples:
            key = (sample['seed'], sample['snapshot'])
            _, readout, comparison = joined[split][(key, 'conditioned_mean8')]
            out = [i for i, (lo, hi) in enumerate(bounds) if not lo <= sample['features'][i] <= hi]
            witness = {'seed': key[0], 'snapshot': key[1], 'target': sample['target'],
                       'left_frequency': sample['features'][23], 'last_left': sample['features'][26],
                       'outside_training_coordinate_ranges': out}
            if out:
                outside.append(witness)
            if readout['action']['kind'] != 0:
                changed.append({**witness, 'action': readout['action']['kind']})
            if readout['action']['kind'] == 0 and any(a['mean_advantage_over_keep'] > 0 for a in comparison['actions']):
                missed.append(witness)
        result['splits'][split] = {
            'unique_feature_vectors': len({tuple(s['features']) for s in samples}),
            'constant_coordinates': {str(i): samples[0]['features'][i] for i in range(29)
                                     if len({s['features'][i] for s in samples}) == 1},
            'states_with_left_history': sum(s['features'][23] > 0 for s in samples),
            'states_with_last_left': sum(s['features'][26] == 1 for s in samples),
            'changed_actions': changed, 'missed_positive_states': missed,
            'outside_training_coordinate_box': outside}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipts, datasets, readouts, pins = load()
    joined = {s: validate(s, receipts, datasets[s], readouts[s]) for s in datasets}
    report = {'schema': 1, 'analysis': 'post-result frozen-policy margin and paired-credit diagnosis',
        'policy_fits': 0, 'body_generation_runs': 0,
        'script_sha256': digest(Path(__file__).read_bytes()),
        'inputs': {'receipts_sha256': RECEIPT_SHA, 'archive_sha256': ARCHIVE_SHA, 'members': pins},
        'definitions': {'margin': 'score[action] - score[KEEP]; native choice is strict masked argmax with lowest-kind ties',
            'loss': 'mean across states of mean valid-action Huber error against associated conditioned H4 targets; shuffled arm is assessed against associated outcomes, not its acquisition donors',
            'positive_mean': 'original eight-paired-draw H4 advantage > 0',
            'leave_one_out': 'eight descriptive seven-draw means; draws remain nested within each captured state',
            'cohort': '48 states in 12 episodes and two seeds per split; training42/73 and exposed evaluation509/601',
            'scope': 'post-result description; all original choices, rewards and measurement rules retained'},
        'feature_support': feature_support(datasets, joined),
        'splits': {s: analyze(joined[s]) for s in datasets}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({s: r['conditioned_mean8']['groups'] for s, r in report['splits'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
