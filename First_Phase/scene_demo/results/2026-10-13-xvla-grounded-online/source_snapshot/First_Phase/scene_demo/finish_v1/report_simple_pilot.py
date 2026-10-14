"""Archive actual results and model inputs; never infer unexecuted actions."""
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'First_Phase/scene_demo/results/2026-10-10-simple-checker'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    rows, inputs, videos, keep = [], [], [], []
    previous = BASE / 'evidence/video_validation.json'
    cached = {r['path']: r for r in read(previous)} if previous.exists() else {}
    names = [('wine', 'wine_exit', '酒瓶放架'), ('bowl', 'bowl_held83', '碗放盘子'),
             ('sauce', 'sauce_negative', '番茄酱罐入篮'), ('stove', 'stove_exit', '灶台开火')]
    for label, case, title in names:
        directory = 'pilot_wine_repair' if label == 'wine' else 'pilot_' + label
        case_dir = BASE / directory / case
        result_path = case_dir / 'result.json'
        result = read(result_path)
        model_rows = []
        check_file = case_dir / 'runtime_checks/checks.jsonl'
        if check_file.exists():
            for line in check_file.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                model_rows.append(row)
                for raw_path in row.get('image_paths', []):
                    path = Path(raw_path)
                    inputs.append({'case': case, 'call_index': row['call_index'],
                                   'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
                    keep.append(path)
        first = model_rows[0] if model_rows else {}
        final = model_rows[-1] if model_rows else {}
        snapshot = result.get('final_snapshot') or {}
        finish = result.get('finish') or {}
        initial_snapshot = None
        oracle_file = case_dir / 'oracle.jsonl'
        if oracle_file.exists():
            for line in oracle_file.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row.get('tag') == 'initial': initial_snapshot = row['snapshot']
        actions = result.get('phase_counts') or {}
        goal = first.get('raw_verdict', {}).get('operation_achieved')
        item = {'case': case, 'title': title, 'status': result['status'],
                'initial_goal_achieved': goal, 'initial_decision': first.get('initial_check', {}).get('decision'),
                'initial_visual': first.get('raw_verdict'), 'initial_oracle': initial_snapshot,
                'actual_open_actions': result.get('release_open_actions', 0),
                'actual_settle_actions': finish.get('release_settle_actions', 0),
                'actions': actions, 'finish_reason': finish.get('reason'),
                'final_visual_decision': final.get('initial_check', {}).get('decision'),
                'final_oracle_goal': snapshot.get('gold', {}).get('predicates'),
                'final_held_objects': snapshot.get('gold', {}).get('held_objects'),
                'final_hand_contacts': len(snapshot.get('contacts', [])),
                'home_ready': snapshot.get('home', {}).get('ready'),
                'physical_streak': result.get('physical_streak'), 'errors': result.get('errors'),
                'qwen_calls': read(case_dir.parent / 'summary.json')['qwen_calls'],
                'qwen_tokens': sum(read(p).get('usage', {}).get('total_tokens', 0)
                                  for p in (case_dir / 'runtime_checks/model_requests').glob('*_response.json')),
                'video': str(case_dir / 'all_actions.mp4')}
        rows.append(item)
        video = case_dir / 'all_actions.mp4'
        if video.exists():
            fingerprint = sha(video)
            old = cached.get(str(video.relative_to(ROOT)))
            if old and old['sha256'] == fingerprint:
                frame_count = old['frames']
            else:
                reader = imageio.get_reader(video)
                try: frame_count = sum(1 for _ in reader)
                finally: reader.close()
            videos.append({'case': case, 'path': str(video.relative_to(ROOT)), 'frames': frame_count,
                           'sha256': sha(video), 'decoded_to_end': True})
        for name in ('result.json', 'oracle.jsonl', 'helper.jsonl', 'events.jsonl', 'checks.jsonl', 'all_actions.mp4'):
            path = case_dir / name
            if path.exists(): keep.append(path)
        if check_file.exists(): keep.append(check_file)
        request_dir = case_dir / 'runtime_checks/model_requests'
        if request_dir.exists(): keep.extend(p for p in request_dir.iterdir() if p.is_file())
        for name in ('summary.json',): keep.append(case_dir.parent / name)
    initial_calls = read(BASE / 'pilot_wine/summary.json')['qwen_calls']
    summary = {'cases': rows, 'qwen_calls': sum(r['qwen_calls'] for r in rows) + initial_calls,
               'qwen_tokens': sum(r['qwen_tokens'] for r in rows), 'new_vla': 0, 'new_hermes': 0,
               'subagents': 0, 'physical_success_cases': sum(r['status'] == 'physical_success' for r in rows),
               'case_count': 4, 'simulation_launches': 5,
               'initial_wine_protocol_failure': {'qwen_calls': initial_calls, 'qwen_tokens': None, 'raw_response': 'unavailable; old base client discarded validation failure', 'actual_helper_actions': 0},
               'tokens_are_lower_bound': True,
               'scope': 'historical VLA terminal replay plus NEW actual check/open/withdraw/home; not fresh full VLA rollout'}
    write(BASE / 'evidence/summary.json', summary)
    write(BASE / 'evidence/model_inputs.json', inputs)
    write(BASE / 'evidence/video_validation.json', videos)
    body = ['# 两问检查器：四项真实收尾与归位小试', '',
            '**本轮没有新增完整归位成功。酒瓶、碗卡在检查器接管判断；灶台开爪已实际执行，但退离未达到路径终点；番茄目标原本未入篮。**', '',
            '从已有 VLA 轨迹真实恢复末态后，测试检查→开爪→退离→原归位的接管流程。实际动作见表：本轮所有归位均未启动。这是收尾接管测试，不是新 VLA 从头运行或多任务组合测试。', '',
            '| 子任务 | 初次判断目的达成 | 实际开爪 | 退离动作 | 归位动作 | 最终标准归位 | 结果/停止位置 |',
            '|---|---|---:|---:|---:|---|---|']
    reasons = {'wine_exit': '瓶子已放架，Qwen误判悬空，未触发收尾',
               'bowl_held83': '碗已到盘子上仍夹持，Qwen误判悬空，未触发收尾',
               'sauce_negative': '物品尚未入篮，目的未达，拒绝收尾',
               'stove_exit': '开爪成功；退离120次后预算耗尽，归位未启动'}
    for r in rows:
        body.append(f"| {r['title']} | {'是' if r['initial_goal_achieved'] else '否'} | {r['actual_open_actions']} | {r['actions'].get('movement', 0)} | {r['actions'].get('home', 0)} | {'是' if r['home_ready'] else '否'} | {reasons[r['case']]} |")
    body.extend(['', f"实际 Qwen 调用 {summary['qwen_calls']} 次，已保存 usage 合计 {summary['qwen_tokens']} tokens；首次 wine 协议失败的一次调用 usage 未保留，故 token 总数未知、这里只是下限。新 VLA/Hermes/子agent 均为 0。旧动作回放与新收尾动作分开统计。",
                 '', '酒瓶与碗的初始独立目标记录为真。碗仍被持有，酒瓶没有被持有。Qwen两例都误报悬空，因此不能据零收尾动作评价原归位流程。碗先前已完成收尾归位的成功记录仍有效。',
                 '', '番茄酱（LIBERO 中为罐头）旧轨迹没有达到入篮目的；不能人为标成放置成功再强制收尾。拒绝收尾不等于该任务完整成功。Qwen还把未被持有的番茄酱罐说成被夹着，所以这里只能说“未达目的”的判断正确，不能称整份检查正确。',
                 '', '灶台17表示17个张开控制周期，不是17次开合。实测夹爪达到标准开口，开火保持、手部接触为空；仍未完成程序规定的退离路径，因此归位动作0。三项应接管正例中，接管并实际开爪1/3，完整归位0/3；番茄未达目标的负例不纳入这个分母。',
                 '', '首次酒瓶调用因返回接口格式未通过解析，随后只修复一次协议并保留失败记录。最终同版四项各一次，失败停止本例，其余独立样例继续；未测试串联组合，演示界面自动收尾未启用。独立效果使用仿真目标、未持物、无手部接触及标准归位连续五个实际动作后采样；这些信息没有输入 Qwen。',
                 '', '下一步故障分开处理：酒瓶/碗先消除视觉与编码器之间的错误因果解释；灶台分析退离目标跟踪，不能再把它报告成归位失败或直接无限增加预算。暂不训练。', ''])
    for r in rows:
        video = r['video']
        rel = str(Path(video).relative_to(ROOT)).replace('\\', '/')
        body.extend([f"{r['title']}：", '', f"![{r['title']}](D:/FYP/{rel})", ''])
    (BASE / 'README.md').write_text('\n'.join(body), encoding='utf-8')
    for path in (BASE / 'evidence').iterdir():
        if path.is_file() and path.name != 'manifest.json': keep.append(path)
    keep.append(BASE / 'README.md')
    for path in (BASE / 'pilot_wine').rglob('*'):
        if path.is_file() and 'sensors' not in path.parts: keep.append(path)
    unique = sorted(set(keep))
    write(BASE / 'evidence/manifest.json', {'tracked_files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in unique],
                                         'local_intermediate_sensor_files': 'retained locally, not all committed',
                                         'model_inputs_preserved': True})
    print(json.dumps({k: summary[k] for k in ('qwen_calls', 'qwen_tokens', 'physical_success_cases', 'case_count')}))


if __name__ == '__main__': main()
