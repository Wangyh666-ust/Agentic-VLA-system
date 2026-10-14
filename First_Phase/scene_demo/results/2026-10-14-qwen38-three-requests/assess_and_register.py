"""Register the completed attempts without converting partial outcomes to success."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = read(OUT / 'audit.json')
    assert audit['all_cases_finished'] and all(audit['protected_source_unchanged'].values())
    table = audit['cases']['table_tidy_stove']
    heat = audit['cases']['heat_yellow_mug']
    discard = audit['cases']['discard_two_cans']
    assert table['physical_user_goal_success'] is True and all(j['handoff_ok'] for j in table['jobs'])
    assert [j['reference_capability'] for j in table['jobs'][:2]] == ['wine_to_rack', 'stove_on']
    assert table['jobs'][-1]['reference_kind'] == 'global_home'
    assert heat['pipeline_terminal'] == 'blocked' and heat['jobs'][0]['outcome']['reason'] == 'independent_finish_failed'
    assert discard['status'] == 'program_error' and discard['source_counts']['vla'] == 0
    native = [r for p in (OUT / 'heat_yellow_mug').rglob('native_steps.jsonl') for r in rows(p) if r['source'] == 'vla']
    first_true = next(i + 1 for i, r in enumerate(native) if r['native_success'])
    assert first_true == 256 and all(r['native_success'] for r in native[255:])
    independent = [r for p in (OUT / 'heat_yellow_mug').rglob('independent_finish.jsonl') for r in rows(p)]
    streak = longest = 0
    for r in independent:
        streak = streak + 1 if r['ready'] else 0
        longest = max(longest, streak)
    assert longest == 4 and independent[-1]['ready'] is False
    assert independent[-1]['metrics']['ready'] is True and independent[-1]['contacts'] == []
    assert all(independent[-1]['predicates'].values())
    hermes = read(OUT / 'hermes_logged_usage.json')
    assessment = {
        'manual_user_intent_assessment': {'table_tidy_stove': True, 'heat_yellow_mug': False, 'discard_two_cans': None},
        'heat_component_native_goal': {'first_true_vla_step': first_true, 'final_vla_step': len(native), 'all_remaining_true': True},
        'heat_final_independent': {'longest_consecutive_ready': longest, 'final_ready': False, 'robot_reference_ready': True,
                                   'raw_goals_true': True, 'hand_contacts': [], 'exact_completion_probe_failure_reason': 'not recorded'},
        'source_counts': {k: sum(c['source_counts'][k] for c in audit['cases'].values()) for k in ('settling', 'prepare', 'vla', 'finish')},
        'direct_qwen_successful_responses': sum(len(c['actual_qwen_responses']) for c in audit['cases'].values()),
        'direct_qwen_failed_attempts': 1, 'failed_attempt_server_usage': None,
        'hermes_api_calls': sum(c['api_calls'] for c in hermes.values()),
        'known_successful_response_tokens': sum(c['direct_qwen_tokens'] for c in audit['cases'].values()) + sum(c['total_tokens'] for c in hermes.values()),
        'max_observed_images_per_request': max(r['image_count'] for c in audit['cases'].values() for r in c['actual_qwen_responses']),
        'recorded_release_events': sum(c['release_events'] for c in audit['cases'].values()),
        'server_cleanup': read(OUT / 'runtime/server_cleanup.json'),
        'subagents': 0, 'grasp_transport_helpers_added': False,
    }
    (OUT / 'assessment.json').write_text(json.dumps(assessment, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files = {OUT / 'summary.md', OUT / 'assessment.json', ROOT / 'First_Phase/scene_demo/plans/20261014-qwen38-three-requests.json'}
    for p in OUT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.name == 'registry_validation.json':
            continue
        if p.suffix in ('.json', '.jsonl', '.stdout', '.stderr', '.log', '.py') or p.name == 'initial_state.npy':
            files.add(p)
        if p.name.startswith('call_') and p.name.endswith('_request.json'):
            for image in read(p)['images']:
                image_path = ROOT / image['path'].split('/FYP/', 1)[1]
                assert sha(image_path) == image['sha256']
                files.add(image_path)
        if p.name == 'process_events.jsonl':
            for event in rows(p):
                for image in event['images']:
                    image_path = ROOT / image['path'].split('/FYP/', 1)[1]
                    assert sha(image_path) == image['sha256']
                    files.add(image_path)
    # Also retain the third initial triplet even though no request audit was produced.
    files.update((OUT / 'discard_two_cans').rglob('cap000002*.png'))
    entry = {
        'id': '20261014-qwen38-three-requests',
        'title': 'Qwen3.8三条原文：整理整链通过；加热误匹配与独立确认失败；丢弃初识别网络中断',
        'date': '2026-10-14', 'question_keys': ['qwen38-three-user-requests', 'hermes-intent-planning', 'next-subtask-start-reset'],
        'tags': ['Qwen3.8', 'X-VLA', 'online', 'natural-language', 'next-reference', 'semantic-mismatch', 'transport-error'],
        'conditions': ['三个独立场景各一次，用户原文不改；seed142/init0，Hermes及初识别Qwen3.8-Max、末端Qwen3-VL-Plus，原控制/收尾/复位未改。',
                       '没有强加capability_ids顺序或重试；无抓取搬运辅助器、训练、回放或子agent。',
                       'X-VLA45请求/1350步，原收尾938/prepare10/settle30步；直接Qwen25响应+1失败尝试，Hermes8日志API调用。'],
        'findings': ['整理真实提交碗→酒瓶→开灶，三个用户目标最终同时为真；两次下一原生起点交接及末任务globalhome、全部handoff和五连通过。',
                     '加热错误匹配成入炉关门，库无启动加热能力；原生AND从256至900步持续为真，但900步才触发收尾。',
                     '黄杯收尾356步、机器人reference ready且原生goals为真，释放/静止独立筛查只4连而非5连，末帧False，整链blocked/independent_finish_failed；具体cup速度/抓取代理未记录，不断言根因。',
                     '丢弃第一轮初识别Qwen网络失败、零响应/零Hermes/零VLA；照片到报错约94.35s，90s超时是疑点而非已证实原因，意图拒绝效果未验证。',
                     '成功请求图片及body哈希匹配；6个松手事件各三图保存；黄杯首次松手历史证据confirmed真实引用旧图。受保护九项源码哈希未变，进程已清理。'],
        'limits': ['每项一次、场景/请求不同、随机流未固定，不合并成功率或把改善归因于单一模型变化。',
                   '初识别失败原异常被包装，无法确认超时或连接中断，失败请求服务端推理/用量未知。',
                   '黄杯末帧缺完整释放/静止探针值，不能判定是速度阈值、抓取代理或未知观测。',
                   '旧max_input_images15字段未约束当前v7.1，实际最多33图；依据原请求统计。',
                   '根线程实际模型/推理档位未获工具确认，不用名称冒充调用证据。'],
        'configuration_complete': False,
        'evidence': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sorted(files)],
    }
    registry_path = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
    registry = read(registry_path)
    assert not any(e['id'] == entry['id'] for e in registry['entries'])
    registry['entries'].append(entry)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (ROOT / 'First_Phase/scene_demo/EXPERIMENT_INDEX.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 2026-10-14：Qwen3.8三条自然语言任务\n\n- 实验 `20261014-qwen38-three-requests`：[报告与原图](results/2026-10-14-qwen38-three-requests/summary.md)。\n- 整理碗、酒瓶并开灶整链通过，两次下一起点复位与末次home真实执行。\n- 加热被误匹配为入炉关门；原生AND256至900为真，独立释放/静止只4连而整体阻断。丢弃初识别网络失败，未进入规划，不记作正确拒绝。\n- 当前流程未现场修改，无新增抓取搬运辅助器/子agent；完整请求、过程原图与失败记录保留。\n')
    print(json.dumps({'registered': entry['id'], 'evidence_count': len(files), 'assessment': assessment}, ensure_ascii=False))


if __name__ == '__main__':
    main()
