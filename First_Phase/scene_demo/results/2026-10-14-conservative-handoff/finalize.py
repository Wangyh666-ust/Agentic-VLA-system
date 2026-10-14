"""Archive bounded offline evidence without modifying original experiments."""
import hashlib,json,shutil,sys
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
sys.path.insert(0,str(root/'First_Phase'))
from scene_demo.xvla_subtask_evidence import build_prompt

names=['book_before','book_before_contact','book_after','bowl_held','microwave_history','microwave_geometry']
summaries={n:json.loads((out/n/'summary.json').read_text()) for n in names}
expected={'book_before_contact':None,'book_after':False,'bowl_held':True,'microwave_geometry':True}
assert all(summaries[n]['goal_achieved'] is value for n,value in expected.items())
assert summaries['bowl_held']['handoff_review_triggered'] is True
assert summaries['microwave_geometry']['proofs'][0]['kind']=='historical_review'
models=[];prompt_checks=[]
for name in names:
    for response in sorted((out/name/'model_requests').glob('*response.json')):
        raw=json.loads(response.read_text())
        request=response.with_name(response.name.replace('response','request'))
        data=json.loads(request.read_text())
        models.append({'case':name,'file':str(response.relative_to(root)),
            'requested':data['requested_model'],'returned':raw['model']})
        if name in expected and '上次回答有协议缺失' not in data['prompt']:
            packet=json.loads(data['prompt'].splitlines()[-1])
            prompt_checks.append({'case':name,'file':str(request.relative_to(root)),
                'matches_final_builder':build_prompt(packet)==data['prompt']})
assert all(r['requested']==r['returned']=='qwen3.8-max' for r in models)
assert all(r['matches_final_builder'] for r in prompt_checks)
base=root/'First_Phase/scene_demo'
files=['xvla_relation_gate.py','xvla_subtask_evidence.py','xvla_subtask_review.py','xvla_grounding.py','xvla_service.py',
       'tests/test_xvla_relation_gate.py','tests/test_xvla_subtask_review.py','tests/test_xvla_grounding.py','tests/test_xvla_integration.py']
snapshot=out/'source_snapshot'
for name in files:
    target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/name,target)
protected=json.loads((out/'protected_controls.json').read_text())
assert all(r['unchanged'] for r in protected)
summary={'protocol':'xvla_subtask_review_v8','new_robot_actions':0,'subagents':0,
    'cases':[{'name':n,'expected':value,'actual':summaries[n]['goal_achieved'],
              'handoff_review_triggered':summaries[n]['handoff_review_triggered'],
              'time':summaries[n]['target_time']} for n,value in expected.items()],
    'historical_placement':{'observed_at':summaries['microwave_geometry']['proofs'][0]['observed_at'],
        'reviewed_at':summaries['microwave_geometry']['proofs'][0]['reviewed_at'],
        'event_id':summaries['microwave_geometry']['proofs'][0]['event_id']},
    'successful_vlm_requests':len(models),'models':models,'final_prompt_checks':prompt_checks,
    'software':'245 passed, 30 subtests passed','protected_controls':protected,
    'initial_failures':['First implementation still falsely accepted book before release (4 calls), retained.',
                        'First microwave replay recovered placement but misread transparent door as open (3 calls), retained.'],
    'test_isolation_incident':'11 invalid fixture-image HTTP400 requests; no successful vision answers; repaired network isolation; no physical actions.',
    'limits':['Offline results are not online motion success or a success-rate estimate.',
              'All 10 native tasks covered in software; real image cases cover book, bowl and microwave only.',
              'External-camera geometry and original images unchanged; occlusion can still delay handoff.',
              'Individual model descriptions may be imperfect; no claim of exact 3D support measurement.',
              'No subagents; actual root model/reasoning setting not verified by a runtime metadata tool.']}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
registry=base/'experiment_registry.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
entry_id='20261014-conservative-handoff-offline'
assert not any(e['id']==entry_id for e in data['entries'])
evidence=[out/'summary.json',out/'proposal.json',out/'review_check.stdout',out/'software.stdout',out/'software.json',out/'protected_controls.json']
evidence+=list(snapshot.rglob('*.py'))
for name in names:
    evidence += [out/name/'summary.json',out/name/'manifest.json']
    evidence += list((out/name/'model_requests').glob('*.json'))
entry={'id':entry_id,'date':'2026-10-14','title':'通用具体区域与保守接管：真实照片离线回放',
    'question_keys':['book-caddy-checker-model','xvla-handoff-boundary','xvla-checker-grounding'],
    'tags':['offline','qwen3.8-max','handoff','regions','no-new-actions'],
    'conditions':['用户授权通用区域定位及保守接管修复；不调用子agent，不加辅助器，零新仿真动作。',
                  'Qwen3.8原图回放；增加区域边界、三视角关系、可见承接接触与接管复核；原收尾/退离/复位源码不变。'],
    'findings':['书松手前unknown不接管；落盒外False；盘上碗（尚未标准开口）True且接管复核触发。',
                '黄杯入炉历史证据observed_at8.95、reviewed_at63.55复用；通用透明门几何规则后关门True，整体True。',
                '245 passed/30 subtests；全部10原生技能的软件正负例覆盖；真实照片涉及三种场景。',
                f'成功审计VLM请求{len(models)}次均Qwen3.8；首版书误放行与首版门误判原记录保留，不能用最终结果覆盖失败。'],
    'limits':summary['limits']+['测试隔离遗漏导致11次无效图片HTTP400，修复后测试无网络请求；这些请求不计视觉验收样本。'],
    'evidence':[{'path':str(p.relative_to(root)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in evidence],
    'configuration_complete':False}
data['entries'].append(entry);data['updated_at']='2026-10-14'
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (base/'EXPERIMENT_INDEX.md').open('a',encoding='utf-8') as stream:
    stream.write('\n### 20261014-conservative-handoff-offline\n\n'
        '- 具体目的地区域 + 保守接管：书松手前 unknown 拦截，盒外 False；盘上碗未标准开口仍 True，并实际触发接管复核（仅模型分支，无运动）。\n'
        '- 黄杯复用 8.95 秒松手证据；63.55 秒当前门状态按门板几何判 closed，整体 True。初版误放行/透明门误判均保留。\n'
        '- 软件 245 passed、30 subtests；10 原生接口覆盖。真实照片 3 场景，19 次成功 Qwen3.8 请求；不代表在线成功率。零新动作、无子agent、原控制不变。\n'
        '- 测试隔离遗漏曾发出 11 次无效图片 HTTP400，修复并归档，不计视觉样本。\n'
        '- 证据：[summary.json](results/2026-10-14-conservative-handoff/summary.json)。\n')
print(json.dumps({'cases':summary['cases'],'calls':len(models),'registered':entry_id},ensure_ascii=False))
