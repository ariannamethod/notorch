#!/usr/bin/env python3
"""History-composition, masked-action and authenticated-replay refusal gates."""
import argparse
import copy
import ctypes as C
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('history_probe', ROOT / 'experiments/spa_agent/history_probe/run.py')
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
SPA = RUN.SPA


def sample(target=1, seed=42, snapshot=0):
    features = [0.] * 29
    features[12] = C.c_float(target / 3).value
    features[15] = float(target > 0)
    features[16] = float(target < 3)
    features[19:29] = [.2, .3, .4, 0., 1., 0., 0., 1., 0., .5]
    return {'features': features, 'target': target, 'sentence_count': 4,
            'life_hash': '0000000000000001', 'action_mask': 1 | (2 if target else 0) | (4 if target < 3 else 0),
            'seed': seed, 'snapshot': snapshot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    checks, defects = [], []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def reject(name, fn):
        try:
            fn()
        except (ValueError, SPA.Error):
            defects.append(name)
        else:
            raise AssertionError('deliberate defect escaped: ' + name)

    source = sample()
    donor = sample(snapshot=1)
    donor['features'][0] = .75
    donor['features'][19:29] = [.6, .7, .8, 1., 0., 0., 1., 0., 0., -.3]
    for variant, selected in [('identity', None), ('zero_history', None), ('training_donor', donor)]:
        value = RUN.make_intervention(source, variant, selected)
        check(variant + ' first nineteen features', bytes(value.features)[:76] == bytes(RUN.POLICY.experience(source).features)[:76])
        RUN.validate_intervention(source, value, variant, selected)
    check('source immutable', source == sample())
    altered = RUN.make_intervention(source, 'training_donor', donor)
    altered.features[0] = donor['features'][0]
    reject('perception leakage', lambda: RUN.validate_intervention(source, altered, 'training_donor', donor))
    altered = RUN.make_intervention(source, 'training_donor', donor)
    altered.features[28] = source['features'][28]
    reject('partial history swap', lambda: RUN.validate_intervention(source, altered, 'training_donor', donor))
    altered = RUN.make_intervention(source, 'identity')
    altered.source_life_hash = 2
    reject('metadata leakage', lambda: RUN.validate_intervention(source, altered, 'identity', None))
    reject('wrong donor coordinates', lambda: RUN.make_intervention(source, 'training_donor', sample(target=2)))
    reject('unknown intervention', lambda: RUN.make_intervention(source, 'unknown'))
    donors = [sample(seed=42 + i // 6, snapshot=i % 6) for i in range(12)]
    for row in donors:
        row['outcomes'] = object()  # A read of outcomes must never be needed.
    check('all twelve donors source ordered', RUN.donor_pool(source, donors) == donors)
    reject('missing donor', lambda: RUN.donor_pool(source, donors[:-1]))
    reject('duplicate donor', lambda: RUN.donor_pool(source, donors[:-1] + donors[:1]))
    readout = SPA.Readout(action=SPA.Action.at(0, 0), action_mask=5,
                          scores=(C.c_float * 3)(0, 100, -1))
    RUN.validate_readout(sample(target=0), readout)
    check('invalid high head masked', readout.action.kind == 0)
    readout.action = SPA.Action(1, 0, SPA.NO_SOURCE)
    reject('unmasked action', lambda: RUN.validate_readout(sample(target=0), readout))
    readout.action = SPA.Action.at(0, 0)
    readout.action_mask = 7
    reject('changed mask', lambda: RUN.validate_readout(sample(target=0), readout))
    readout.action_mask = 5
    readout.action.target = 1
    reject('wrong typed target', lambda: RUN.validate_readout(sample(target=0), readout))
    expected = RUN.identity_bytes(b'original receipt')
    RUN.authenticate(b'original receipt', expected, 'fixture')
    reject('corrupted receipt', lambda: RUN.authenticate(b'original receipT', expected, 'fixture'))
    reject('truncated receipt', lambda: RUN.authenticate(b'original receip', expected, 'fixture'))
    RUN.validate_protocol(ROOT / 'experiments/spa_agent/history_probe/protocol.json')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'protocol.json'
        value = dict(RUN.REGISTERED, donors_per_state=11)
        path.write_text(json.dumps(value))
        reject('changed frozen protocol', lambda: RUN.validate_protocol(path))
    native_status = 'SKIPPED: --library was not supplied'
    if args.library:
        native = SPA.Native(args.library)
        agent = SPA.Agent(SPA.Config.default(native=native, mode=SPA.Mode.LEARNED, seed=1, exploration=0), native=native)
        before = bytes(agent.state)
        for variant, selected in [('identity', None), ('zero_history', None), ('training_donor', donor)]:
            captured = RUN.make_intervention(source, variant, selected)
            RUN.score_one(agent, source, captured, native)
            check('native accepts ' + variant, bytes(agent.state) == before)
        captured = RUN.make_intervention(source, 'identity')
        captured.features[25] = 1  # Two simultaneous last actions.
        reject('native refuses incoherent one-hot', lambda: RUN.score_one(agent, source, captured, native))
        check('refusal preserves native life', bytes(agent.state) == before)
        # A real scorer wrapped with a state mutation must fail the read-only gate.
        class MutatingAgent:
            @property
            def state(self):
                return agent.state

            def score(self, exp):
                out = agent.score(exp)
                agent.reset_memory()
                policy = agent.state.policy
                policy.b2[0] += .1
                agent.set_policy(policy)
                return out
        captured = RUN.make_intervention(source, 'identity')
        reject('native readout state mutation', lambda: RUN.score_one(MutatingAgent(), source, captured, native))
        native_status = 'PASS: native composition, read-only scoring and malformed history refusal'
    report = {'status': 'PASS', 'checks': checks, 'deliberate_defects_caught': defects,
              'native': native_status}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'deliberate_defects': len(defects), 'native': native_status}))


if __name__ == '__main__':
    main()
