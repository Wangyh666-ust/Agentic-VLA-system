"""One documented protocol repair; preserve first failed launch evidence."""
import json
import shutil
import zipfile
from .prepare_simple_pilot import ROOT, OUT, PLAN, sha, write


def main():
    evidence = OUT / 'evidence'
    inputs = evidence / 'inputs.json'
    preflight = evidence / 'preflight.json'
    for source, name in ((PLAN, 'proposal_initial.json'), (inputs, 'inputs_initial.json'), (preflight, 'preflight_initial.json')):
        shutil.copyfile(source, evidence / name)
    data = json.loads(inputs.read_text())
    data['cases'][1]['expected_release'] = True
    for item in data['frozen_files'].values():
        path = ROOT / item['path']
        item.update(bytes=path.stat().st_size, sha256=sha(path))
    write(inputs, data)
    plan = json.loads(PLAN.read_text())
    plan['scope']['max_cases'] = 5
    plan['scope']['stop_rule'] = ('首次wine协议解析失败、收尾动作0，保留原始失败。允许一次明确协议修复wine重试；其余三个独立例各一次。'
                               '应触发未触发则停止本例依赖；不无限重试、不组合、不训练、不子agent。')
    plan['configuration']['inputs_sha256'] = sha(inputs)
    source_map = {rel: sha(ROOT / rel) for rel in plan['configuration']['source_sha256']}
    plan['configuration']['source_sha256'] = source_map
    plan['configuration']['protocol_repair'] = 'archive raw response before parsing; missing schema metadata remains null, no invented positives'
    plan['changed_factors'].append({'name': 'protocol_decoder', 'before': 'strict whole-schema response discard on parse failure',
                                    'after': 'raw response audit; absent fields unknown; normalize optional boxes keys',
                                    'reason': 'first wine had one HTTP call, parse failure, physical helper0; original response/usage not retained by old client, marked unavailable'})
    write(PLAN, plan)
    write(preflight, {'proposal_sha256': sha(PLAN), 'inputs_sha256': sha(inputs), 'source_sha256': source_map})
    with zipfile.ZipFile(evidence / 'source_snapshot_repair.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in source_map: z.write(ROOT / rel, rel)
    print('protocol repair frozen; original launch preserved')


if __name__ == '__main__': main()
