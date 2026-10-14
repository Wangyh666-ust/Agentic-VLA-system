"""Register the preserved failed control before the authorized one-case repeat."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
registry = root / 'First_Phase/scene_demo/experiment_registry.json'
run = 'First_Phase/scene_demo/results/interactive-console/runs/20261014T030501Z_book_caddy_274c55f3'
cap = run + '/cap_01_xvla_book_caddy_b6725ed9'
paths = [cap+'/pilot_result.json', cap+'/events.jsonl', cap+'/finish/model_requests/call_01_response.json',
         cap+'/finish/model_requests/call_04_response.json', cap+'/finish/sensors/cap000095_wrist_checker_sideview.png',
         cap+'/finish/sensors/cap000326_wrist_checker_sideview.png',
         'First_Phase/scene_demo/results/interactive-console/20261014-105935-857a55b5/draft.json']
entry = dict(id='20261014-book-caddy-old-checker', title='用户书收纳在线失败：旧检查器提前放行', date='2026-10-14',
    question_keys=['book-caddy-checker-model'], tags=['book_caddy','xvla','checker','qwen3-vl-plus'],
    conditions=['原用户请求：请帮我把书和杯子整理一下，放好；book_caddy seed142 state0；书为可执行部分，杯子未覆盖。',
                '初识别/Hermes=qwen3.8-max；执行检查=qwen3-vl-plus；无抓取辅助器，原收尾/退离/home。'],
    findings=['VLA150步，native入格谓词未真；视觉误判完全入格，收尾主动开爪9动作。',
              '松手过程图与末态书均在盒外，视觉仍判到位；收尾290动作，最终独立筛查失败，归位实测通过。'],
    limits=['单次失败不证明只由模型选择造成；机位遮挡/隔间绑定仍可能影响结果；不能预测同轨迹Qwen3.8表现。'],
    evidence=[dict(path=p,sha256=hashlib.sha256((root/p).read_bytes()).hexdigest()) for p in paths],
    configuration_complete=False)
data=json.loads(registry.read_text(encoding='utf-8-sig'))
if not any(e['id']==entry['id'] for e in data['entries']):
    data['entries'].append(entry)
    data['updated_at']='2026-10-14'
    registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('preserved baseline registered:',entry['id'])
