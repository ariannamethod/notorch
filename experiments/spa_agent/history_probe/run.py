#!/usr/bin/env python3
"""Frozen-policy native SPA readouts under complete history-block interventions."""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes as C
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


POLICY = module('history_conditioned', 'experiments/spa_agent/conditioned/policy.py')
TRANSFER = module('history_transfer', 'experiments/spa_agent/transfer/analyze.py')
SPA = POLICY.SPA
require, identity = POLICY.require, POLICY.identity
ARMS = ('initial', 'conditioned_mean8')
PROTOCOL_SHA = 'af116f2d153e3494cba93eac55033604e8c01851431d1713658bd133c1d25af6'
REGISTERED = {
    'schema': 1, 'arms': list(ARMS), 'history_coordinates': [19, 29],
    'variants': ['identity', 'zero_history', 'training_donor'],
    'donors_per_state': 12, 'replay_updates': 24576,
    'source_rows': {'training': 48, 'evaluation': 48},
    'training_seeds': [42, 73], 'evaluation_seeds': [509, 601],
    'learning_rate': .03, 'scale_floor': .001, 'epochs': 512}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_protocol(path):
    before = identity(path)
    require(before['sha256'] == PROTOCOL_SHA, 'frozen history protocol SHA')
    protocol = POLICY.read_json(path)
    require(all(protocol.get(k) == v for k, v in REGISTERED.items()),
            'history probe protocol differs from registered experiment')
    require(all(identity(ROOT / name)['sha256'] == value
                for name, value in protocol['source_sha256'].items()),
            'registered implementation source changed')
    require(identity(path) == before, 'protocol changed while reading')
    return protocol, before


def authenticate(raw, expected, label):
    require({'bytes': len(raw), 'sha256': sha(raw)} == expected,
            'authenticated identity changed: ' + label)
    return raw


def journal_bytes(receipts):
    archive = b''.join(TRANSFER.source(f'raw_records.tar.gz.part-{i:02d}') for i in range(3))
    require(sha(archive) == TRANSFER.ARCHIVE_SHA, 'archive SHA')
    name = 'fit-1/fit.jsonl.gz'
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tf:
        names = [m.name for m in tf.getmembers()]
        require(len(names) == len(set(names)), 'duplicate archive member')
        member = tf.getmember(name)
        require(member.isfile(), 'journal member type')
        return authenticate(tf.extractfile(member).read(), receipts['archive']['members'][name], name)


def recover(samples, native, output, receipts, raw_journal):
    """Replay the original updates and require every original byte receipt."""
    initial = SPA.Agent(SPA.Config.default(native=native, mode=SPA.Mode.LEARNED,
                                          seed=1, exploration=0), native=native)
    initial_state = initial.state
    initial.save(output / 'initial.life')
    agent = SPA.Agent.from_file(output / 'initial.life', native=native)
    seal = receipts['fit_repeat']['seals'][0]
    for arm in ARMS:
        require(seal['lives'][arm] == receipts['fit_repeat']['seals'][1]['lives'][arm],
                'original independent life identities differ')
    expected = seal['lives']['initial']
    require(identity(output / 'initial.life') == {k: expected[k] for k in ('bytes', 'sha256')}
            and f'{initial.hash:016x}' == expected['life_hash'], 'initial life parity')
    inputs, count, comparisons = {}, 0, 0
    decoded = hashlib.sha256()
    record_counts = Counter()
    with gzip.GzipFile(fileobj=io.BytesIO(raw_journal)) as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'unterminated original journal record')
            decoded.update(line)
            r = json.loads(line, object_pairs_hook=POLICY.PARENT.unique_object)
            record_counts[r['type']] += 1
            if r['type'] == 'header':
                require(sum(record_counts.values()) == 1 and r['abi'] == SPA._layout()
                        and r['byteorder'] == sys.byteorder and r['epochs'] == 512
                        and r['fits_per_arm'] == 24576 and r['rate'] == .03
                        and r['scale_floor'] == .001
                        and r['protocol_sha256'] == POLICY.FROZEN_PROTOCOL_SHA256,
                        'original journal ABI/config header')
            elif r.get('arm') == 'conditioned_mean8' and r['type'] == 'comparison':
                index = comparisons
                require(index < 48, 'too many conditioned comparisons')
                sample = samples[index]
                captured = POLICY.experience(sample)
                values, references = POLICY.comparisons(sample, sample, 8)
                probe = agent.fit_conditioned(captured, values, 0, .001)
                expected_record = {
                    'type': 'comparison', 'comparison_id': 'conditioned_mean8:' + str(index),
                    'arm': 'conditioned_mean8', 'source': POLICY.source_identity(sample),
                    'source_features': list(captured.features), 'donor': POLICY.source_identity(sample),
                    'repeats': references, 'native_input_hex': [bytes(v).hex() for v in values],
                    'initial_evaluation': POLICY.native_receipt(probe), 'receipt_hex': bytes(probe).hex()}
                require(r == expected_record, 'original comparison parity at row ' + str(index))
                require(bytes(agent.state) == bytes(initial_state), 'rate-zero replay changed life')
                inputs[index] = captured, values
                comparisons += 1
            elif r.get('arm') == 'conditioned_mean8' and r['type'] == 'fit':
                require(comparisons == 48 and count < 24576, 'replay input/update coverage')
                epoch, row = divmod(count, 48)
                before, policy_before = f'{agent.hash:016x}', sha(bytes(agent.state.policy))
                captured, values = inputs[row]
                receipt = agent.fit_conditioned(captured, values, .03, .001)
                POLICY.policy_only(agent, initial_state)
                actual = {'type': 'fit', 'arm': 'conditioned_mean8', 'epoch': epoch,
                    'row': row, 'comparison_id': 'conditioned_mean8:' + str(row),
                    'source_feature_witness': samples[row]['feature_witness'], 'donor_row': row,
                    'repeat_count': 8, 'life_before': before, 'life_after': f'{agent.hash:016x}',
                    'policy_before_sha256': policy_before,
                    'policy_after_sha256': sha(bytes(agent.state.policy)),
                    'receipt_hex': bytes(receipt).hex()}
                require(actual == r, 'original fit parity at update ' + str(count))
                count += 1
    require(record_counts == {'header': 1, 'comparison': 144, 'fit': 73728}
            and count == 24576 and comparisons == 48, 'complete original replay coverage')
    require(decoded.hexdigest() == seal['trace']['uncompressed']['sha256'],
            'decoded original journal identity')
    agent.save(output / 'conditioned_mean8.life')
    expected = seal['lives']['conditioned_mean8']
    require(identity(output / 'conditioned_mean8.life') == {k: expected[k] for k in ('bytes', 'sha256')}
            and f'{agent.hash:016x}' == expected['life_hash'], 'acquired life parity')
    return {'initial': initial, 'conditioned_mean8': agent}, {
        'status': 'PASS', 'replay_updates': count, 'comparison_receipts': comparisons,
        'all_fit_receipts_and_before_after_hashes_exact': True,
        'journal': {'compressed': identity_bytes(raw_journal), 'decoded_sha256': decoded.hexdigest()},
        'lives': {arm: seal['lives'][arm] for arm in ARMS},
        'purpose': 'Exact recovery of previously acquired weights from original updates',
        'new_training_experiments': 0, 'body_generation_runs': 0}


def identity_bytes(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def donor_pool(sample, training):
    # Neither outcomes nor policy readouts enter donor selection.
    donors = [r for r in training if (r['sentence_count'], r['target']) ==
              (sample['sentence_count'], sample['target'])]
    require(len(donors) == 12 and len({(d['seed'], d['snapshot']) for d in donors}) == 12,
            'complete twelve-donor stratum')
    return donors


def make_intervention(sample, variant, donor=None):
    captured = POLICY.experience(sample)
    if variant == 'zero_history':
        require(donor is None, 'zero-history donor')
        for i in range(19, 29):
            captured.features[i] = 0
    elif variant == 'training_donor':
        require(donor is not None and (sample['sentence_count'], sample['target']) ==
                (donor['sentence_count'], donor['target']), 'donor coordinates')
        for i in range(19, 29):
            captured.features[i] = donor['features'][i]
    else:
        require(variant == 'identity' and donor is None, 'unknown history intervention')
    validate_intervention(sample, captured, variant, donor)
    return captured


def validate_intervention(sample, captured, variant, donor):
    source = POLICY.experience(sample)
    want = bytes(source.features)[19*4:] if variant == 'identity' else (
        bytes(40) if variant == 'zero_history' else bytes(POLICY.experience(donor).features)[19*4:])
    require(bytes(captured.features)[19*4:] == want, 'complete history block changed')
    restored = SPA.Experience.from_buffer_copy(bytes(captured))
    for i in range(19, 29):
        restored.features[i] = source.features[i]
    require(bytes(restored) == bytes(source), 'perception or metadata changed')


def validate_readout(sample, readout):
    mask = POLICY.PARENT.action_mask(sample)
    require(readout.action_mask == mask == sample['action_mask'], 'readout action mask')
    valid = [k for k in range(3) if mask & (1 << k)]
    scores = list(readout.scores)
    import math
    require(all(math.isfinite(v) for v in scores), 'nonfinite readout')
    selected = max(valid, key=lambda k: (scores[k], -k))
    require(readout.action.kind == selected, 'masked action selection')
    expected = SPA.Action.at(selected, sample['target'])
    require(bytes(readout.action) == bytes(expected), 'typed action coordinates')


def score_one(agent, sample, captured, native):
    before = bytes(agent.state)
    exp_before = bytes(captured)
    require(native.lib.nt_spa_experience_validate(C.byref(captured)) == 0,
            'native experience refused')
    readout = agent.score(captured)
    require(bytes(agent.state) == before, 'readout changed weights/state/RNG')
    require(bytes(captured) == exp_before, 'readout changed experience')
    validate_readout(sample, readout)
    return readout


def run_probe(datasets, archived, agents, native, stream):
    counts, records = {}, 0
    baselines = {}
    # Verify the complete original readout product before the first intervention.
    for split, samples in datasets.items():
        old = {(r['arm'], r['source']['seed'], r['source']['snapshot']): r for r in archived[split]}
        for sample in samples:
            for arm in ARMS:
                agent = agents[arm]
                base = score_one(agent, sample, make_intervention(sample, 'identity'), native)
                original = old[arm, sample['seed'], sample['snapshot']]
                require(bytes(base).hex() == original['readout_hex']
                        and f'{agent.hash:016x}' == original['life_hash'], 'archived readout parity')
                baselines[split, arm, sample['seed'], sample['snapshot']] = base
    for split, samples in datasets.items():
        for sample in samples:
            variants = [('identity', None), ('zero_history', None)] + [
                ('training_donor', d) for d in donor_pool(sample, datasets['training'])]
            for arm in ARMS:
                agent = agents[arm]
                base = baselines[split, arm, sample['seed'], sample['snapshot']]
                frozen = {'life_hash': f'{agent.hash:016x}', 'state_sha256': sha(bytes(agent.state)),
                          'policy_sha256': sha(bytes(agent.state.policy))}
                for variant, donor in variants:
                    captured = make_intervention(sample, variant, donor)
                    readout = score_one(agent, sample, captured, native)
                    changed = readout.action.kind != base.action.kind
                    record = {'schema': 1, 'split': split, 'arm': arm, 'variant': variant,
                        'source': POLICY.source_identity(sample),
                        'donor': POLICY.source_identity(donor) if donor else None,
                        'history_hex': bytes(captured.features)[19*4:].hex(),
                        'perception_hex': bytes(captured.features)[:19*4].hex(),
                        'experience_hex': bytes(captured).hex(), 'readout_hex': bytes(readout).hex(),
                        'scores': list(readout.scores), 'action_mask': readout.action_mask,
                        'chosen_kind': readout.action.kind, 'baseline_kind': base.action.kind,
                        'changed': changed, 'frozen': frozen,
                        'donor_last_left': bool(donor['features'][26] == 1) if donor else None,
                        'consequences': 'unmeasured' if variant != 'identity' else 'original archived observation'}
                    POLICY.json_line(stream, record)
                    strata = ['all'] + (['donor_last_left' if record['donor_last_left'] else
                                        'donor_other_last'] if donor else [])
                    for stratum in strata:
                        key = (split, arm, variant, stratum)
                        group = counts.setdefault(key, {'records': 0, 'changed': 0,
                            'choices_keep_left_right': [0, 0, 0], 'changed_sources': set()})
                        group['records'] += 1
                        group['changed'] += changed
                        group['choices_keep_left_right'][readout.action.kind] += 1
                        if changed:
                            group['changed_sources'].add((sample['seed'], sample['snapshot']))
                    records += 1
    require(records == 2688, 'complete probe product')
    groups = []
    for (split, arm, variant, stratum), group in sorted(counts.items()):
        group['changed_sources'] = [list(k) for k in sorted(group['changed_sources'])]
        groups.append(dict(split=split, arm=arm, variant=variant, stratum=stratum, **group))
    return {'status': 'PASS', 'readouts': records, 'archived_baseline_parity': 192,
            'groups': groups, 'policy_updates_during_probe': 0, 'body_generation_runs': 0,
            'counterfactual_consequences': 'unmeasured',
            'sampling_unit': '96 source states; twelve correlated donor probes per state'}


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    _, protocol_id = validate_protocol(args.protocol)
    native = SPA.Native(args.library)
    source_paths = ('spa_agent.c', 'spa_agent.h', 'spa_binding.c', 'spa_binding.h', 'python/SPA.py',
        'experiments/spa_agent/history_probe/run.py',
        'experiments/spa_agent/conditioned/policy.py',
        'experiments/spa_agent/replicates/policy.py', 'experiments/spa_agent/transfer/analyze.py')
    source_ids = {p: identity(ROOT / p) for p in source_paths}
    library_id = identity(native.path)
    receipts, datasets, archived, pins = TRANSFER.load()
    for split, samples in datasets.items():
        TRANSFER.validate(split, receipts, samples, archived[split])
        for ordinal, sample in enumerate(samples):
            POLICY.PARENT.validate_source(sample, ordinal)
            require(sample['seed'] == REGISTERED[split + '_seeds'][ordinal // 24], 'cohort order')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'start.json', {'protocol': protocol_id, 'source_ids': source_ids,
        'library': library_id, 'inputs': {'receipts_sha256': TRANSFER.RECEIPT_SHA,
        'archive_sha256': TRANSFER.ARCHIVE_SHA, 'members': pins}})
    agents, recovery = recover(datasets['training'], native, args.output, receipts, journal_bytes(receipts))
    write_json(args.output / 'recovery.json', recovery)
    with (args.output / 'probes.jsonl').open('x') as stream:
        summary = run_probe(datasets, archived, agents, native, stream)
    require(identity(args.protocol) == protocol_id and identity(native.path) == library_id,
            'protocol/library changed during experiment')
    require({p: identity(ROOT / p) for p in source_paths} == source_ids, 'source changed during experiment')
    summary.update(protocol=protocol_id, source_ids=source_ids, library=library_id,
                   probes=identity(args.output / 'probes.jsonl'), recovery=identity(args.output / 'recovery.json'))
    write_json(args.output / 'summary.json', summary)
    print(json.dumps({'status': 'PASS', 'readouts': summary['readouts'], 'replay_updates': 24576}))


if __name__ == '__main__':
    main()
