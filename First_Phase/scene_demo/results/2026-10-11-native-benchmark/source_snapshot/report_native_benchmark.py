"""Read-only benchmark evidence consumer; never controls benchmark worker."""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

OUT = Path('/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark')
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
VENV = '/home/yhwang/fyp/libero_demo/venv/bin/python'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def probe(path):
    code = ('import av,json,sys; c=av.open(sys.argv[1]); '
            'fps=float(c.streams.video[0].average_rate); '
            'n=sum(1 for _ in c.decode(video=0)); c.close(); '
            'print(json.dumps([n,fps]))')
    frames, fps = json.loads(subprocess.check_output([VENV, '-c', code, str(path)], text=True))
    return int(frames), float(fps)


def videos(tasks):
    delivery = read(OUT / 'delivery_progress.json') or {'videos': [], 'issues': []}
    try:
        ffmpeg = subprocess.check_output([VENV, '-c',
            'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],text=True).strip()
        if not Path(ffmpeg).exists():
            raise FileNotFoundError(ffmpeg)
    except Exception as exc:
        delivery['issues'] = [f'已安装ffmpeg无法定位：{exc}；保留原视频，未安装工具。']
        save(OUT / 'delivery_progress.json', delivery)
        return
    delivery['issues'] = []
    known = {item['dest'] for item in delivery['videos']}
    for task in tasks:
        if not task['complete']:
            continue
        lang = task['language'].lower().strip()
        representative = task['key'] in ['libero_spatial_00', 'libero_10_00', 'libero_10_09'] or \
            (task['suite'] == 'libero_goal' and ('wine bottle' in lang or 'turn on the stove' in lang or
                                               lang == 'put the bowl on the plate'))
        if not representative:
            continue
        successes = task['successes']
        chosen = []
        for flag in [True, False]:
            if flag in successes:
                chosen.append(successes.index(flag))
        for episode in chosen:
            if len(delivery['videos']) >= 12:
                break
            source = Path(task['video_paths'][episode])
            if not source.is_absolute():
                source = Path('/mnt/d/FYP') / source
            dest = OUT / 'presentation' / task['key'] / f'eval_episode_{episode}.mp4'
            if str(dest) in known:
                continue
            if dest.exists():
                delivery['issues'].append(f'已有未验证副本，跳过不重复转码：{dest}')
                continue
            if not source.exists():
                delivery['issues'].append(f'原视频尚不可用：{source}')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                frames, source_fps = probe(source)
                subprocess.run([ffmpeg, '-v', 'error', '-r', '20', '-i', str(source),
                    '-an', '-c:v', 'libx264', '-r', '20', str(dest)], check=True)
                target_frames, target_fps = probe(dest)
                if frames != target_frames or target_fps != 20:
                    dest.unlink()
                    raise ValueError(f'frame/fps mismatch: {frames}/{target_frames}, {target_fps}')
                delivery['videos'].append({'source': str(source), 'dest': str(dest),
                    'framecount': frames, 'target_framecount': target_frames,
                    'source_fps': source_fps, 'target_fps': target_fps,
                    'success': successes[episode]})
                known.add(str(dest))
            except Exception as exc:
                if dest.exists():
                    dest.unlink()
                delivery['issues'].append(str(exc))
    save(OUT / 'delivery_progress.json', delivery)


def update():
    progress = read(OUT / 'progress.json') or {}
    error = read(OUT / 'error.json')
    launch = read(OUT / 'launch.json') or {}
    pid = launch.get('pid')
    alive = bool(pid and Path(f'/proc/{pid}/cmdline').exists() and
                 b'native_benchmark.py' in Path(f'/proc/{pid}/cmdline').read_bytes())
    abnormal = bool(error) or (not alive and progress.get('status') != 'complete')
    tasks = []
    issues = []
    for suite in SUITES:
        for task_id in range(10):
            key = f'{suite}_{task_id:02d}'
            folder = OUT / key
            audit, info = read(folder / 'audit.json'), read(folder / 'eval_info.json')
            task = {'suite': suite, 'task_id': task_id, 'key': key, 'complete': False,
                    'language': '', 'successes': [], 'steps': [], 'video_paths': []}
            if audit and info:
                successes = info['per_task'][0]['metrics']['successes']
                valid = audit.get('valid') is True and len(successes) == 5 and \
                    successes == audit.get('native_successes') and len(audit.get('episodes', [])) == 5 and \
                    all(isinstance(v, bool) for v in successes) and \
                    [e['success'] for e in audit['episodes']] == successes
                if valid:
                    task.update(complete=True, language=audit['language'][0], successes=successes,
                                steps=[e['scored_steps'] for e in audit['episodes']],
                                video_paths=info['per_task'][0]['metrics']['video_paths'])
                else:
                    issues.append(f'{key}计分审计不一致，未计入分母')
            tasks.append(task)
    completed = [t for t in tasks if t['complete']]
    n = len(completed)*5
    successes = sum(sum(t['successes']) for t in completed)
    finished = progress.get('status') == 'complete' and len(completed) == 40
    status = '异常' if abnormal or issues else '完成' if finished else '运行中'
    suites = {s: {'completed_tasks': sum(t['complete'] for t in tasks if t['suite']==s),
                   'valid_episodes': sum(t['complete'] for t in tasks if t['suite']==s)*5,
                   'planned_episodes': 50,
                   'successes': sum(sum(t['successes']) for t in completed if t['suite']==s)} for s in SUITES}
    save(OUT / 'per_suite.json', suites)
    fields = ['suite','task_id','status','language'] + [f'success_{i}' for i in range(5)] + \
             [f'steps_{i}' for i in range(5)] + ['n_success','valid_episodes']
    with (OUT / 'native_tasks.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in tasks:
            row = dict(suite=t['suite'],task_id=t['task_id'],status='已完成' if t['complete'] else '未完成',
                       language=t['language'],n_success=sum(t['successes']) if t['complete'] else '',
                       valid_episodes=5 if t['complete'] else 0)
            for i in range(5):
                row[f'success_{i}'] = int(t['successes'][i]) if t['complete'] else ''
                row[f'steps_{i}'] = t['steps'][i] if t['complete'] else ''
            writer.writerow(row)
    ratio = f'{successes/n:.1%}' if n else '暂无有效样本'
    lines = [f'# 原生 SmolVLA LIBERO 评测：{status}', '',
        f'有效次数：{n}/计划200；已完成任务：{len(completed)}/40；成功数：{successes}。',
        f'{"总成功率" if finished else "累计已完成样本比例"}：{ratio}。未完成或异常样本不填失败。', '',
        '| Suite | 完成任务 | 有效次数/计划 | 成功数 |', '|---|---:|---:|---:|']
    for s, v in suites.items():
        lines.append(f'| {s} | {v["completed_tasks"]}/10 | {v["valid_episodes"]}/50 | {v["successes"]} |')
    lines += ['', '计分使用原生 `is_success`：单次为真即成功，不附加松手或归位要求。无 Hermes、Qwen、helper 或训练。',
        'AMP关闭，沿用下载默认配置；动作上限依次为280/280/300/520。每任务使用官方前5个初始状态、环境seed1000..1004、模型seed1000。',
        '当前MuJoCo3.8.1、hf-libero0.1.4，checkpoint为32层；尚未证明与论文设置相同。因此结果测量当前权重与环境组合，低分不直接证明训练缺陷，每任务5次仅作初筛。', '',
        '原始SDK视频metadata为80fps，control为20Hz，因此源视频约4倍速播放；不补动作，原视频保留。', '',
        '| Task | 成功/有效 | 五次成功 | 五次计分步数 | 原始完整指令 |', '|---|---:|---|---|---|']
    for t in tasks:
        flags = ','.join('T' if v else 'F' for v in t['successes']) or '未完成'
        steps = ','.join(map(str,t['steps'])) or '—'
        score = f'{sum(t["successes"])}/5' if t['complete'] else '未完成'
        language = t['language'].replace('|','\\|').replace('\n',' ')
        lines.append(f'| {t["key"]} | {score} | {flags} | {steps} | {language} |')
    if abnormal or issues:
        lines += ['', '异常信息：', json.dumps(error,ensure_ascii=False) if error else 'worker已结束或审计证据不一致。'] + issues
    videos(tasks)
    delivery = read(OUT / 'delivery_progress.json') or {'videos': [], 'issues': []}
    if delivery['videos']:
        lines += ['', '已验证正常速度代表视频：源目标帧数相同，副本20fps呈现全部原帧。', '']
        for item in delivery['videos']:
            windows_path = item['dest'].replace('/mnt/d/', 'D:/', 1)
            link = f'<{windows_path}>' if ' ' in windows_path else windows_path
            lines.append(f'- [{Path(item["dest"]).parent.name}/{Path(item["dest"]).name}]({link})'
                         f'：{item["framecount"]}帧，20fps。')
    else:
        lines += ['', '正常速度副本尚未生成。']
    if delivery['issues']:
        lines += ['', '视频交付限制：'] + delivery['issues']
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':status,'completed_tasks':len(completed),'valid_episodes':n,
                      'successes':successes,'ratio':ratio},ensure_ascii=False),flush=True)
    return finished or abnormal or bool(issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    while True:
        stop = update()
        if args.once or stop:
            break
        time.sleep(60)
