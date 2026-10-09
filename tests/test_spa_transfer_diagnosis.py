#!/usr/bin/env python3
"""Reject forged action/credit joins in the read-only SPA transfer diagnosis."""
import copy
import importlib.util
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transfer', ROOT / 'experiments/spa_agent/transfer/analyze.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
receipts, datasets, readouts, pins = A.load()
for split in datasets:
    A.validate(split, receipts, datasets[split], readouts[split])

caught = []
def reject(name, change):
    r, d, o = copy.deepcopy((receipts, datasets['evaluation'], readouts['evaluation']))
    change(r, d, o)
    try:
        A.validate('evaluation', r, d, o)
    except (ValueError, KeyError):
        caught.append(name)
        return
    raise AssertionError('accepted mutation: ' + name)

reject('missing_readout', lambda r,d,o:o.pop())
reject('duplicate_readout', lambda r,d,o:o.__setitem__(1, copy.deepcopy(o[0])))
reject('source_witness', lambda r,d,o:o[0]['source'].__setitem__('snapshot_raw_sha256','0'*64))
reject('missing_source_witness', lambda r,d,o:o[0]['source'].pop('snapshot_raw_sha256'))
reject('mask', lambda r,d,o:o[0].__setitem__('action_mask',7))
reject('score_byte_mismatch', lambda r,d,o:o[0]['scores'].__setitem__(2,.5))

def selection(r,d,o):
    # Coherent score bytes, but KEEP is now the wrong available action.
    o[0]['scores'][2] = .5
    raw = bytearray.fromhex(o[0]['readout_hex'])
    struct.pack_into('<f',raw,24,.5)
    o[0]['readout_hex'] = raw.hex()
reject('coherent_scores_wrong_action', selection)

def credit(r,d,o):
    for c in r['evaluation']['state_comparisons']:
        if c['horizon']==4 and c['seed']==509 and c['snapshot']==0:
            c['actions'][1]['mean_advantage_over_keep'] *= -1
reject('credit_sign',credit)

def reward_join(r,d,o):
    for c in r['evaluation']['state_comparisons']:
        if c['horizon']==4 and c['seed']==509 and c['snapshot']==0:
            c['actions'][1]['mean_reward'] += .5
reject('reward_advantage_join',reward_join)

def paired_error(r,d,o):
    for c in r['evaluation']['state_comparisons']:
        if c['horizon']==4 and c['seed']==509 and c['snapshot']==0:
            c['actions'][1]['paired_standard_error'] = 0
reject('paired_error',paired_error)

def common_credit(r,d,o):
    c=next(c for c in r['evaluation']['state_comparisons'] if c['horizon']==4 and c['label']=='conditioned_mean8')
    c['actions'][1]['replicate_advantages'][0] += .125
reject('arm_specific_consequence',common_credit)

original_source=A.source
A.source=lambda name: b'{}' if name=='receipts.json' else original_source(name)
try:
    A.load()
except ValueError:
    caught.append('receipt_identity')
else:
    raise AssertionError('accepted forged receipt')
finally:
    A.source=original_source

# Real masked-head trap: positive LEFT at target0 must never enter valid margins.
j=A.validate('training',receipts,datasets['training'],readouts['training'])
sample,readout,c=j[((42,7),'conditioned_mean8')]
assert readout['scores'][1]>0 and readout['action']['kind']==0 and readout['action_mask']==5
report=A.analyze(j)['conditioned_mean8']
assert not any(x['seed']==42 and x['snapshot']==7 and x['action']==1 for x in report['alternatives'])
assert report['choices_keep_left_right']==[45,1,2]
print(json.dumps({'status':'PASS','readouts_validated':384,'deliberate_corruptions_caught':caught,
                  'positive_invalid_head_excluded':True},sort_keys=True))
