import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[4]
out=Path(__file__).resolve().parent
run=root/'First_Phase/scene_demo/results/interactive-console/runs/20261014T050837Z_book_caddy_1060c3a0'
cap=run/'cap_01_xvla_book_caddy_131dff1d'
attempt=root/'First_Phase/scene_demo/results/interactive-console/20261014-130348-6b2083a7'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
responses=[]
for p in sorted(run.rglob('*response.json')):
    r=read(p)
    if not r.get('choices'):continue
    request=p.with_name(p.name.replace('_response','_request'))
    responses.append({'path':str(p.relative_to(root)), 'returned_model':r.get('model'),
                      'requested_model':read(request).get('requested_model') if request.exists() else None})
events=lines(cap/'events.jsonl'); independent=lines(cap/'independent_finish.jsonl')
result=read(cap/'pilot_result.json')
summary={'purpose':'全部VLM改为Qwen3.8后复测书收纳；检查到位放行、实际松手和独立入格效果',
 'run_dir':str(run.relative_to(root)), 'attempt_dir':str(attempt.relative_to(root)),
 'models':responses,'all_recorded_vlm_requests_and_responses_qwen38':all(r['requested_model']==r['returned_model']=='qwen3.8-max' for r in responses),
 'source_counts':result['source_counts'],'outcome':result['outcome'],'finish':result['finish_status'],
 'home_ready':result['actual_final_metrics']['ready'],
 'vla_goal_true_samples':sum(e['native_success'] is True for e in events),
 'vla_samples':len(events),'independent_samples':len(independent),
 'independent_goal_true_samples':sum(any(r['predicates'].values()) for r in independent),
 'old_handoff_same_images_prompt_probe':{'model':read(out/'old_handoff_qwen38/summary.json')['model'],
    'actions':0,'calls':1,'answer':'at_destination, supported=true; same false positive'},
 'findings':['模型迁移真实触发；150步Qwen3.8仍误判入格并触发收尾开爪10动作。',
             '1次松手三图已存；松手后outside被识别，一次多余JSON括号经原有一次协议修复后生效；最终也判outside。',
             '原图单次对照也误判到位；不能将旧失败仅归因于模型接入错误。',
             '归位实测通过，但书未入目标后格，最终visual failed，完整流程失败。'],
 'limits':['在线n=1；初始三图哈希与旧运行一致，但执行轨迹非完全相同；不估计成功率。',
           'Hermes使用本次独立profile的qwen3.8-max；上述原始请求/返回审计计数仅覆盖保存的视觉客户端。',
           '首次启动因上次info.json遗留失败，零新任务动作，保留后恢复；仅1次在线动作尝试。',
           '未修改检查提示词、相机、动作/收尾/退离/复位；未加辅助器、未调用子agent。']}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
paths=[out/'summary.json',out/'profile_models.json',out/'unchanged_controls.json',out/'software_validation.json',
       out/'old_handoff_qwen38/call_01_request.json',out/'old_handoff_qwen38/call_01_response.json',
       attempt/'proposal.json',attempt/'review_check.json',attempt/'draft.json',cap/'pilot_result.json',
       cap/'events.jsonl',cap/'finish/model_requests/call_01_request.json',cap/'finish/model_requests/call_01_response.json',
       cap/'finish/model_requests/call_04_response.json',cap/'finish/sensors/cap000104_wrist_checker_sideview.png',
       cap/'finish/sensors/cap000326_wrist_checker_sideview.png']
entry=dict(id='20261014-all-vlm-qwen38-book',title='全VLM切换Qwen3.8与书收纳单次复测',date='2026-10-14',
    question_keys=['book-caddy-checker-model'],tags=['qwen3.8-max','book_caddy','checker','early_release'],
    conditions=['用户要求全VLM切换并复测；book_caddy seed142 state0、原请求/同一能力、预算900；无助手、无子agent。',
                '初识别/复核/Hermes/执行及历史检查均改为qwen3.8-max；提示词/三机位/原控制及收尾不变。'],
    findings=summary['findings'],limits=summary['limits'],configuration_complete=False,
    evidence=[{'path':str(p.relative_to(root)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
registry=root/'First_Phase/scene_demo/experiment_registry.json'; data=read(registry)
assert not any(e['id']==entry['id'] for e in data['entries'])
data['entries'].append(entry);data['updated_at']='2026-10-14'
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
index=root/'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
with index.open('a',encoding='utf-8') as f:
    f.write('\n- `20261014-all-vlm-qwen38-book`：[全VLM切换与书收纳复测](results/2026-10-14-all-vlm-qwen38/summary.json)。初识别/复核/Hermes/全部检查统一Qwen3.8；原图对照1次仍误报到位。在线1次VLA150步仍提前允许松手，收尾291步，书在盒外；新模型松手后与最终均识别outside，整链final visual failed；归位通过不替代入格。原提示词/机位/收尾与复位不变，无助手/子agent。\n')
print(json.dumps({k:summary[k] for k in ['all_recorded_vlm_requests_and_responses_qwen38','source_counts','outcome','home_ready','vla_goal_true_samples','independent_goal_true_samples']}))
print('recorded VLM request/response pairs:',len(responses))
