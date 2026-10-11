"""Summarize only fresh online evidence; no inference or robot commands."""
import json
from pathlib import Path

from .online_combo_pilot import ROOT, OUT, PLAN
from .combo_pilot import sha, write


def main():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    rows, keep = [], [PLAN, OUT / 'source_snapshot.zip']
    for name in ('bowl_wine', 'bowl_stove'):
        directory = OUT / name
        if not (directory / 'result.json').exists():
            continue
        result = json.loads((directory / 'result.json').read_text())
        jobs = []
        with imageio.get_writer(str(directory / 'sequence.mp4'), fps=20) as writer:
            for sub in result['subtasks']:
                job, finish = sub['job'], sub['finish'] or {}
                run = Path(job['run_dir'])
                action_file = run / 'finish_actions.jsonl'
                actions = [json.loads(s) for s in action_file.read_text().splitlines()] if action_file.exists() else []
                independent_file = run / 'independent_finish.jsonl'
                independent = [json.loads(s) for s in independent_file.read_text().splitlines()] if independent_file.exists() else []
                checks = []
                for request in sorted((run / 'finish/model_requests').glob('*_request.json')):
                    raw = json.loads(request.read_text())
                    response = json.loads(request.with_name(request.name.replace('_request', '_response')).read_text())
                    ctx = raw['context']
                    stage = ctx['actuator_measurements']['stage']
                    online_number = sum(c['kind'] == 'online' for c in checks)
                    checks.append({'kind': 'online' if stage == 'executing' else 'after_cleanup',
                        'vla_step': (150, 300)[online_number] if stage == 'executing' else job['steps'],
                        'timestamp': ctx['actuator_measurements']['timestamp'],
                        'content': response['choices'][0]['message']['content'], 'usage': response.get('usage'),
                        'request_path': request.relative_to(ROOT).as_posix(),
                        'images': raw['images'], 'input_image_count': len(raw['images']),
                        'stage_sent': '"stage"' in raw['prompt'] or '"phase"' in raw['prompt']})
                    text = raw['prompt'] + '\n' + '\n'.join(
                        f"Image frame={fr['frame']}; view={fr['view']}; timestamp={fr['timestamp']}\n"
                        f"[original image: {img['path']}; sha256={img['sha256']}]"
                        for fr, img in zip(ctx['sensor_frames'], raw['images'])) + '\n'
                    prompt = request.with_suffix('.txt')
                    prompt.write_text(text, encoding='utf-8')
                    keep.append(prompt)
                    keep.extend(Path(img['path']) for img in raw['images'])
                def refs(value):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key in ('rgb_path', 'depth_path', 'calibration_path', 'json_path') and isinstance(item, str) and Path(item).is_file():
                                keep.append(Path(item))
                            refs(item)
                    elif isinstance(value, list):
                        for item in value:
                            refs(item)
                refs(finish)
                count, last = 0, None
                if job.get('rollout_path'):
                    reader = imageio.get_reader(job['rollout_path'])
                    try:
                        for frame in reader:
                            last = frame
                            preview = Image.fromarray(frame)
                            draw = ImageDraw.Draw(preview)
                            draw.rectangle((0, 0, preview.width, 18), fill='black')
                            draw.text((3, 3), 'NEW ONLINE 150/300 | ' + sub['capability'], fill='white')
                            writer.append_data(__import__('numpy').asarray(preview))
                            count += 1
                    finally:
                        reader.close()
                    if last is not None:
                        imageio.imwrite(run / 'final_frame.png', last)
                jobs.append({'capability': sub['capability'], 'new_vla': job['steps'],
                    'success': job['success'], 'reason': job['ended_reason'],
                    'actual_open': sum(a.get('phase') == 'release' and a.get('finger_target') is not None for a in actions),
                    'counts': finish.get('counts', {}), 'home_streak': finish.get('home_streak', 0),
                    'independent_streak': sub['independent_streak'],
                    'final_independent': independent[-1] if independent else None,
                    'finish_reason': finish.get('reason'), 'qwen': checks, 'decoded_frames': count})
        rows.append({'case': name, 'state': result['plan']['state'], 'plan_success': result['plan']['plan_success'],
            'jobs': jobs, 'wall_seconds': result['wall_seconds'], 'timings': result['timings'],
            'video': (directory / 'sequence.mp4').relative_to(ROOT).as_posix()})
        keep.extend(p for p in directory.rglob('*') if p.is_file() and 'sensors' not in p.parts)
        keep.append(OUT / (name + '_launch.log'))
    summary = {'id': '20261011-online-150-300-combos', 'cases': rows,
        'new_vla': sum(j['new_vla'] for r in rows for j in r['jobs']),
        'qwen_calls': sum(len(j['qwen']) for r in rows for j in r['jobs']),
        'qwen_tokens': sum(c['usage'].get('total_tokens', 0) for r in rows for j in r['jobs'] for c in j['qwen']),
        'replay_actions': 0, 'subagents': 0, 'scope': 'fixed user sequence, no new Hermes planning call',
        'not_run': {n: 'shared first bowl failed; registered stop rule' for n in ('bowl_wine', 'bowl_stove')
                    if not (OUT / n / 'result.json').exists()}}
    write(OUT / 'summary.json', summary)
    keep.append(OUT / 'summary.json')
    summary['helper_actions'] = sum(sum(j['counts'].values()) for r in rows for j in r['jobs'])
    write(OUT / 'summary.json', summary)
    body = ['# 150/300步真实在线组合试点', '',
        '**结论：两个组合都实际启动并执行了第二任务，但完整流程均未通过。两次碗任务均正确到位并完成收尾、归位和独立验收。酒瓶第150步仍在桌面，Qwen误报已在架上；实际退离归位完成，但放架失败。灶台第150步已经打开，Qwen正确触发收尾；开爪20周期后仍未达到开口门槛，因此退离和归位均未启动。不能把这次称为两个组合已跑通。**', '',
        '本次从头运行预训练SmolVLA，生成全新录像和原始记录。没有复用旧60步判定、回放旧动作或再次调用保存图片检查。每个组合仅初始化一次场景；子任务间保留物体状态，实际收尾并归位、重置策略队列、读取新观察后执行下一步。', '',
        '| 组合 | 子任务 | 新VLA动作 | 在线检查步 | 实际开爪周期 | 退离周期 | 归位周期 | 独立连续通过 | 结果 |',
        '|---|---|---:|---|---:|---:|---:|---:|---|']
    for row in rows:
        for j in row['jobs']:
            steps = ', '.join(str(c['vla_step']) for c in j['qwen'] if c['kind'] == 'online')
            body.append(f"| {row['case']} | {j['capability']} | {j['new_vla']} | {steps} | {j['actual_open']} | {j['counts'].get('movement', 0)} | {j['counts'].get('home', 0)} | {j['independent_streak']} | {j['reason']} |")
    body += ['', '## 具体发生了什么', '']
    for row in rows:
        body += [f"### {row['case']}：{row['state']}", '']
        for j in row['jobs']:
            body += [f"**{j['capability']}**：归位连续通过 {j['home_streak']}，全目标独立验收连续通过 {j['independent_streak']}。", '']
            for c in j['qwen']:
                label = f"第{c['vla_step']}步在线检查" if c['kind'] == 'online' else '收尾后最终检查（不是新的VLA检查点）'
                body += [label + '：', '', '```json', c['content'], '```', '']
            if j['final_independent']:
                body += ['最终独立位置/开关判据：`' + json.dumps(j['final_independent']['predicates'], ensure_ascii=False) + '`。', '']
        body += [f"![本次新执行 {row['case']}]({ROOT / row['video']})", '']
    body += ['## 配置与结论边界', '',
        '在线检查只在每个子任务的第150、300个实际VLA动作发生；150步判断达到目标后立即停止VLA并接管，因而不再运行到300步。收尾后的最终确认属于另一个环节，不是第三个VLA定时检查点。每次Qwen接收同一时刻腕部第一视角、外部侧视第三视角两张原图及同步夹爪编码器；完整提示词中不发送phase/stage标签。',
        '检查器判断目标位置/开关状态，不要求已经松手或退离才允许接管。实际收尾由原有控制器执行；归位实现和参数保持原版本。本轮没有单独复测归位，没有训练或微调，没有子agent。',
        '独立验收使用启动前固定的目标：碗在盘子上、酒瓶在架子目标区域、灶台打开；第二步同时验证前一步目标仍成立，并检查无持物、无手接触、归位连续五个真实动作后采样。仿真谓词和接触真值不输入Qwen。不以Qwen说完成或单纯归位代替物理目标。',
        '此前保存负例中Qwen的误判仍保留为失败。本次根据用户明确的新请求直接验证真实在线配置，没有把旧负例伪造为通过。两个顺序由用户指定，没有新Hermes规划调用，因此本次验证执行/检查/交接，不衡量Hermes的规划增益；小范围结果不代表可靠性。',
        '录像为本次真实动作，顶部标注NEW ONLINE 150/300及当前子任务。沿用原始记录的缩时采样：VLA每动作一帧，辅助动作每五步一帧，统一20fps，不含模型等待。原始未加标注录像也保存。',
        f"本轮新VLA {summary['new_vla']}动作，辅助 {summary['helper_actions']}周期，历史回放0，Qwen {summary['qwen_calls']}次/{summary['qwen_tokens']}tokens（只统计有原始响应的Qwen调用，不代表整个Codex会话用量）。", '']
    body += ['## 两处剩余故障与下一步', '',
        '酒瓶：第150步输入两张原图均显示瓶子仍在桌面前方。Qwen误把它判断为“架子下层”，收尾后又误报到位；独立验收正确拦截。这里有位置关系识别错误，也有放架尚未完成；由于误判在150步就停止VLA，本轮不能断言继续到300步仍会失败，更不能据此认定训练缺陷。下一步先处理这个空间关系误判，复用本次已保存的错误输入，不扩大物理测试。',
        '灶台：开火目标真实达到，目标触发不是问题。最后开爪反馈为[0.0214956,-0.0380312]，标准为[0.0387306,-0.0387278]；一根手指明显未全开。独立接触记录显示gripper0_finger1_collision仍接触flat_stove_1_g5。当前收尾器先要求全开，未达到后在20周期停止，而未进入解除接触的退离。这说明当前末态仍存在“先全开才能退离，但接触妨碍全开”的阻断；不能仅由预算耗尽推断增加等待就会解决。Qwen另称夹爪未接触，与独立接触记录不一致。下一步修的是带开爪指令的受保护脱离入口，复用已成功的归位模块；本轮不修改、不重试。',
        '本轮程序结果保存后，两个进程退出时各出现EGL_NOT_INITIALIZED析构告警；运行结果和视频已保存，未将退出清理告警当成物理失败原因。', '']
    for name, reason in summary['not_run'].items():
        body += [f'未启动：{name}，原因：{reason}。', '']
    readme = OUT / 'README.md'
    readme.write_text('\n'.join(body), encoding='utf-8')
    keep.append(readme)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    write(OUT / 'manifest.json', {'tracked_files': [
        {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sorted(set(keep)) if p.is_file()]})


if __name__ == '__main__':
    main()
