#!/usr/bin/env python3
"""Independently audit fixed-policy history probes; no fitting or body generation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA = 'af116f2d153e3494cba93eac55033604e8c01851431d1713658bd133c1d25af6'
ARMS = ('initial', 'conditioned_mean8')
SCORE_TOLERANCE = 3e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': sha(raw)}


def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: require(False, 'nonfinite JSON'))


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def fnv(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def load_sources():
    spec = importlib.util.spec_from_file_location('history_audit_transfer',
        ROOT / 'experiments/spa_agent/transfer/analyze.py')
    transfer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transfer)
    receipts, datasets, archived, pins = transfer.load()
    for split in datasets:
        transfer.validate(split, receipts, datasets[split], archived[split])
    return transfer, receipts, datasets, archived, pins


def decode_life(path, expected):
    raw = path.read_bytes()
    require(identity(path) == {k: expected[k] for k in ('bytes', 'sha256')}, 'original life identity')
    require(raw[:8] == b'NTSPA001' and struct.unpack_from('<I', raw, 8)[0] == len(raw), 'life format')
    require(fnv(raw[:-8]) == struct.unpack_from('<Q', raw, len(raw)-8)[0], 'life checksum')
    require(f'{fnv(raw):016x}' == expected['life_hash'], 'original life hash')
    # Canonical codec: 12-byte header, three versions, 56-byte config, config hash.
    values = struct.unpack_from('<267f', raw, 88)
    w1 = [values[i*30:i*30+29] for i in range(8)]
    b1 = [values[i*30+29] for i in range(8)]
    w2 = [values[240+i*9:240+i*9+8] for i in range(3)]
    b2 = [values[248+i*9] for i in range(3)]
    flat = [v for row in w1 for v in row] + b1 + [v for row in w2 for v in row] + b2
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,
            'policy_sha256': sha(struct.pack('<267f', *flat)), 'life_hash': expected['life_hash']}


def forward(policy, features):
    hidden = []
    for row, bias in zip(policy['w1'], policy['b1']):
        z = bias
        for w, x in zip(row, features):
            z = f32(z + f32(w*x))
        hidden.append(f32(math.tanh(z)))
    scores = []
    for row, bias in zip(policy['w2'], policy['b2']):
        z = bias
        for w, x in zip(row, hidden):
            z = f32(z + f32(w*x))
        scores.append(z)
    return scores


def validate(records, datasets, archived, policies):
    """Validate in-memory records and independently reconstruct every aggregate."""
    require(len(records) == 2688, 'complete probe count')
    sources = {(split, s['seed'], s['snapshot']): s
               for split, samples in datasets.items() for s in samples}
    donors = {(s['seed'], s['snapshot']): s for s in datasets['training']}
    old = {(split, r['arm'], r['source']['seed'], r['source']['snapshot']): r
           for split, rows in archived.items() for r in rows if r['arm'] in ARMS}
    witness_keys = set(next(iter(old.values()))['source'])
    expected = set()
    for (split, seed, snapshot), sample in sources.items():
        pool = [k for k, d in donors.items() if (d['target'], d['sentence_count']) ==
                (sample['target'], sample['sentence_count'])]
        require(len(pool) == 12, 'complete donor pool')
        for arm in ARMS:
            expected.update((split, arm, seed, snapshot, v, None) for v in ('identity', 'zero_history'))
            expected.update((split, arm, seed, snapshot, 'training_donor', k) for k in pool)
    seen, state_hashes, groups, maximum_error = set(), {}, {}, 0.0
    for r in records:
        require(r['schema'] == 1 and r['arm'] in ARMS, 'record schema/arm')
        source = r['source']
        sk = (r['split'], source['seed'], source['snapshot'])
        require(sk in sources, 'unknown recipient')
        sample = sources[sk]
        require(source == {k: sample[k] for k in witness_keys}, 'complete recipient witness')
        variant, donor_key = r['variant'], None
        donor = None
        if variant == 'training_donor':
            donor_key = (r['donor']['seed'], r['donor']['snapshot'])
            require(donor_key in donors, 'unknown donor')
            donor = donors[donor_key]
            require(r['donor'] == {k: donor[k] for k in witness_keys}, 'complete donor witness')
            require((donor['target'], donor['sentence_count']) ==
                    (sample['target'], sample['sentence_count']), 'same-coordinate donor')
        else:
            require(variant in ('identity', 'zero_history') and r['donor'] is None, 'variant donor')
        key = (sk[0], r['arm'], sk[1], sk[2], variant, donor_key)
        require(key in expected and key not in seen, 'duplicate/unknown probe cell')
        seen.add(key)
        features = list(sample['features'])
        if variant == 'zero_history':
            features[19:] = [0.0]*10
        elif donor is not None:
            features[19:] = donor['features'][19:]
        fb = struct.pack('<29f', *features)
        require(r['perception_hex'] == fb[:76].hex() and r['history_hex'] == fb[76:].hex(),
                'perception/history byte composition')
        # x86-64 frozen Experience ABI: 3 uint32, 4 pad, uint64, 29 float, 4 pad.
        experience = struct.pack('<III4xQ29f4x', 1, sample['target'], sample['sentence_count'],
                                 int(sample['life_hash'], 16), *features)
        require(r['experience_hex'] == experience.hex(), 'complete experience byte composition')
        baseline = old[(sk[0], r['arm'], sk[1], sk[2])]
        require(r['baseline_kind'] == baseline['action']['kind'], 'original baseline action')
        raw = bytes.fromhex(r['readout_hex'])
        require(len(raw) == 28, 'native readout length')
        kind, target, neighbor, mask, *scores = struct.unpack('<4I3f', raw)
        expected_mask = 1 | (2 if sample['target'] > 0 else 0) | (4 if sample['target']+1 < sample['sentence_count'] else 0)
        require(mask == r['action_mask'] == sample['action_mask'] == expected_mask, 'native action mask')
        require(scores == r['scores'] and all(math.isfinite(v) for v in scores), 'score byte witness')
        valid = [k for k in range(3) if mask & (1 << k)]
        selected = max(valid, key=lambda k: (scores[k], -k))
        require(kind == r['chosen_kind'] == selected, 'strict masked argmax')
        require(target == sample['target'] and neighbor ==
                (0xffffffff if kind == 0 else target-1 if kind == 1 else target+1), 'typed action coordinates')
        require(type(r['changed']) is bool and r['changed'] == (kind != r['baseline_kind']), 'action change flag')
        require(r['donor_last_left'] == (bool(donor['features'][26] == 1) if donor else None), 'donor stratum')
        require(r['consequences'] == ('original archived observation' if variant == 'identity' else 'unmeasured'),
                'counterfactual consequence scope')
        if variant == 'identity':
            require(raw.hex() == baseline['readout_hex'], 'original readout parity')
        if r['arm'] == 'initial':
            require(kind == 0 and scores == [0.0]*3, 'initial zero-head control')
        policy = policies[r['arm']]
        require(r['frozen']['life_hash'] == policy['life_hash'] and
                r['frozen']['policy_sha256'] == policy['policy_sha256'], 'frozen policy identity')
        state_hashes.setdefault(r['arm'], r['frozen']['state_sha256'])
        require(state_hashes[r['arm']] == r['frozen']['state_sha256'] and
                len(r['frozen']['state_sha256']) == 64, 'constant state identity')
        error = max(abs(a-b) for a, b in zip(forward(policy, features), scores))
        require(error <= SCORE_TOLERANCE, 'independent float32 policy scores')
        maximum_error = max(maximum_error, error)
        strata = ['all'] + (['donor_last_left' if r['donor_last_left'] else 'donor_other_last'] if donor else [])
        for stratum in strata:
            group_key = (sk[0], r['arm'], variant, stratum)
            g = groups.setdefault(group_key, {'records': 0, 'changed': 0,
                'choices_keep_left_right': [0, 0, 0], 'changed_sources': set()})
            g['records'] += 1
            g['changed'] += r['changed']
            g['choices_keep_left_right'][kind] += 1
            if r['changed']:
                g['changed_sources'].add((sk[1], sk[2]))
    require(seen == expected, 'complete probe product')
    summary = []
    for (split, arm, variant, stratum), group in sorted(groups.items()):
        group['changed_sources'] = [list(k) for k in sorted(group['changed_sources'])]
        summary.append(dict(split=split, arm=arm, variant=variant, stratum=stratum, **group))
    return summary, maximum_error


def redhands(records, datasets, archived, policies):
    caught = []
    changes = {
        'missing_probe_cell': lambda rows: rows.pop(),
        'duplicate_probe_cell': lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
        'altered_perception': lambda rows: rows[0].__setitem__('perception_hex', '00'*76),
        'forged_donor_witness': lambda rows: next(r for r in rows if r['donor'] is not None)['donor'].__setitem__('life_hash', '0'*16),
    }
    for name, mutation in changes.items():
        rows = copy.deepcopy(records)
        mutation(rows)
        try:
            validate(rows, datasets, archived, policies)
        except ValueError:
            caught.append(name)
        else:
            raise ValueError('accepted red-hand defect: ' + name)
    return caught


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run = args.execution_dir
    protocol_path = ROOT / 'experiments/spa_agent/history_probe/protocol.json'
    require(identity(protocol_path)['sha256'] == PROTOCOL_SHA, 'frozen protocol SHA')
    protocol = parse(protocol_path.read_bytes())
    for name, expected in protocol['source_sha256'].items():
        require(identity(ROOT / name)['sha256'] == expected, 'frozen original source: ' + name)
    transfer, receipts, datasets, archived, pins = load_sources()
    start, recovery, summary = [parse((run / name).read_bytes()) for name in
                                ('start.json', 'recovery.json', 'summary.json')]
    require(start['protocol'] == summary['protocol'] == identity(protocol_path), 'execution protocol identity')
    require(start['inputs'] == {'receipts_sha256': transfer.RECEIPT_SHA,
        'archive_sha256': transfer.ARCHIVE_SHA, 'members': pins}, 'authenticated input identities')
    require(start['source_ids'] == summary['source_ids'] and start['library'] == summary['library'], 'source/library seal')
    for name, expected in start['source_ids'].items():
        require(identity(ROOT / name) == expected, 'executed source identity: ' + name)
    require(recovery['status'] == summary['status'] == 'PASS' and recovery['replay_updates'] == 24576
            and recovery['comparison_receipts'] == 48 and recovery['all_fit_receipts_and_before_after_hashes_exact'] is True
            and recovery['new_training_experiments'] == recovery['body_generation_runs'] == 0, 'exact recovery accounting')
    seal = receipts['fit_repeat']['seals'][0]
    require(recovery['journal']['compressed'] == receipts['archive']['members']['fit-1/fit.jsonl.gz']
            and recovery['journal']['decoded_sha256'] == seal['trace']['uncompressed']['sha256'], 'original replay journal identity')
    policies = {}
    for arm in ARMS:
        expected = seal['lives'][arm]
        require(recovery['lives'][arm] == expected == receipts['fit_repeat']['seals'][1]['lives'][arm], 'recovered life seal')
        policies[arm] = decode_life(run / (arm + '.life'), expected)
    require(summary['probes'] == identity(run / 'probes.jsonl') and summary['recovery'] == identity(run / 'recovery.json'), 'execution artifact identities')
    raw = (run / 'probes.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'complete probe final record')
    records = [parse(line) for line in raw.splitlines()]
    groups, error = validate(records, datasets, archived, policies)
    require(groups == summary['groups'], 'independent aggregate parity')
    require(summary['readouts'] == 2688 and summary['archived_baseline_parity'] == 192
            and summary['policy_updates_during_probe'] == summary['body_generation_runs'] == 0
            and summary['counterfactual_consequences'] == 'unmeasured', 'probe accounting')
    report = {'status': 'PASS', 'schema': 1, 'protocol_sha256': PROTOCOL_SHA,
        'audit_sha256': identity(Path(__file__))['sha256'], 'records': len(records),
        'archived_original_readouts': 192, 'recovery_replayed_updates': 24576,
        'audit_policy_fit_calls': 0, 'new_training_experiments': 0, 'body_generation_runs': 0,
        'independent_float32_score_tolerance': SCORE_TOLERANCE,
        'independent_float32_max_score_error': error,
        'deliberate_corruptions_caught': redhands(records, datasets, archived, policies),
        'artifacts': {name: identity(run / name) for name in
                      ('start.json', 'recovery.json', 'summary.json', 'probes.jsonl')},
        'groups': groups}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'records', 'deliberate_corruptions_caught',
                                           'independent_float32_max_score_error')}, sort_keys=True))


if __name__ == '__main__':
    main()
