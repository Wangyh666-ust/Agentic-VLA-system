"""Summarize new bounded runs from raw evidence, without overwriting history."""
import json
import subprocess
from pathlib import Path
import imageio.v2 as imageio
from .prepare_goal_pilot import ROOT, OUT, PLAN
from .prepare_simple_pilot import sha, write


def main():
    rows, keep = [], []
    old_summary = OUT/'evidence/summary.json'
    old_manifest = OUT/'evidence/manifest.json'
    cached_rows = {row['run']:row for row in json.loads(old_summary.read_text()).get('runs',[])} if old_summary.exists() else {}
    cached_hashes = {row['path']:row['sha256'] for row in json.loads(old_manifest.read_text()).get('tracked_files',[])} if old_manifest.exists() else {}
    names = {'wine_exit':'酒瓶放架','bowl_held83':'碗放盘','stove_exit':'灶台开火','sauce_negative':'番茄酱罐负例'}
    for result_path in sorted(OUT.glob('pilot_*/*/result.json')):
        result = json.loads(result_path.read_text())
        case_dir = result_path.parent
        finish = result.get('finish') or {}
        snapshot = result.get('final_snapshot') or {}
        replies = []
        for response in sorted((case_dir/'runtime_checks/model_requests').glob('*_response.json')):
            raw = json.loads(response.read_text())
            replies.append({'response':str(response.relative_to(ROOT)), 'content':raw['choices'][0]['message']['content'], 'usage':raw.get('usage')})
        row = {'run':case_dir.parent.name,'case':case_dir.name,'title':names.get(case_dir.name,case_dir.name),
               'status':result['status'],'reason':finish.get('reason'),'counts':result.get('phase_counts'),
               'actual_open':result.get('release_open_actions',0),'home_ready':snapshot.get('home',{}).get('ready'),
               'physical_streak':result.get('physical_streak'),'goal':snapshot.get('gold',{}),
               'contacts':snapshot.get('contacts'),'terminal_sha':result.get('terminal_state_sha'),
               'movement_budget':finish.get('movement_budget'),'movement_progress':finish.get('movement_progress'),
               'qwen':replies,'video':str((case_dir/'all_actions.mp4').relative_to(ROOT))}
        launch = json.loads((OUT/'evidence'/('launch_'+case_dir.parent.name.removeprefix('pilot_')+'.json')).read_text())
        row['launch_start_utc'] = launch['start_utc']
        row['partial_open_withdrawal'] = finish.get('withdraw_with_partial_open')
        video = case_dir/'all_actions.mp4'
        replay_frames = result.get('phase_counts',{}).get('replay',0)
        helper_total = sum(result.get('phase_counts',{}).get(k,0) for k in ('release','movement','confirm','home'))
        cached = cached_rows.get(row['run'])
        if cached and cached_hashes.get(video.relative_to(ROOT).as_posix()) == sha(video) and (case_dir/'final_frame.png').exists():
            count = cached['decoded_video_frames']
        else:
            reader = imageio.get_reader(video)
            clip = imageio.get_writer(str(case_dir/'finish_only.mp4'),fps=20) if helper_total else None
            count, last = 0, None
            try:
                for frame in reader:
                    if clip is not None and count >= replay_frames:clip.append_data(frame)
                    count, last = count+1, frame
            finally:
                reader.close()
                if clip is not None:clip.close()
            imageio.imwrite(case_dir/'final_frame.png',last)
        if helper_total:row['finish_video'] = str((case_dir/'finish_only.mp4').relative_to(ROOT))
        row['decoded_video_frames'] = count
        rows.append(row)
        keep += [p for p in case_dir.iterdir() if p.is_file()]
        keep += [p for p in (case_dir/'runtime_checks/model_requests').glob('*') if p.is_file()]
        keep += [case_dir/'runtime_checks/checks.jsonl',case_dir.parent/'summary.json']
        def referenced_files(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('rgb_path','depth_path','calibration_path','json_path') and isinstance(item,str):
                        path = Path(item)
                        if path.is_file():keep.append(path)
                    referenced_files(item)
            elif isinstance(value,list):
                for item in value:referenced_files(item)
        referenced_files(result)
        for request in (case_dir/'runtime_checks/model_requests').glob('*_request.json'):
            record = json.loads(request.read_text())
            keep += [Path(i['path']) for i in record['images']]
            if request.name == 'call_01_request.json':
                texts = [record['prompt']]
                for frame, image in zip(record['context']['sensor_frames'],record['images']):
                    texts.append(f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}")
                    texts.append('[image_url: '+image['path']+'; sha256='+image['sha256']+']')
                prompt_file = case_dir/'complete_prompt.txt'
                prompt_file.write_text('\n'.join(texts)+'\n',encoding='utf-8')
                keep.append(prompt_file)
    rows.sort(key=lambda row:row['launch_start_utc'])
    original = subprocess.check_output(['git','show','HEAD:First_Phase/scene_demo/finish_v1/control.py'],cwd=ROOT).decode().replace('\r\n','\n')
    current = (ROOT/'First_Phase/scene_demo/finish_v1/control.py').read_text().replace('\r\n','\n')
    marker = '        points = jh.joint_waypoints'
    assert original.split(marker,1)[1] == current.split(marker,1)[1]
    summary = {'runs':rows,'qwen_calls':sum(len(r['qwen']) for r in rows),
               'qwen_tokens':sum(c.get('usage',{}).get('total_tokens',0) for r in rows for c in r['qwen']),
               'new_vla':0,'subagents':0,'original_home_execution_equal':True,
               'scope':'恢复旧VLA末态后实际检查/松手/退离/归位；不是新完整VLA或组合实验'}
    summary['latest_results_by_case'] = {row['case']:{'run':row['run'],'status':row['status'],'reason':row['reason']} for row in rows}
    posthoc = []
    incomplete = []
    for directory in sorted(OUT.glob('posthoc_primary*')):
        if not (directory/'result.json').exists():
            incomplete.append({'run':directory.name,'outcome':'unknown','robot_actions':0,
                               'reason':'process/output unavailable; no raw response or verdict retained; no retry'})
            keep += list(directory.glob('*'))
            continue
        result = json.loads((directory/'result.json').read_text())
        response = json.loads((directory/'call_01_response.json').read_text())
        request = json.loads((directory/'call_01_request.json').read_text())
        posthoc.append({'run':directory.name,'decision':result['decision'],'two_question_verdict':result['two_question_verdict'],
                        'case':result.get('case','wine_home'),'usage':result['usage'],'created':response['created'],'robot_actions':0})
        (directory/'complete_prompt.txt').write_text(request['prompt']+'\n'+'\n'.join(
            f"Image frame={fr['frame']}; view={fr['view']}; timestamp={fr['timestamp']}\n[image_url: {img['path']}; sha256={img['sha256']}]"
            for fr,img in zip(request['context']['sensor_frames'],request['images']))+'\n',encoding='utf-8')
        keep += list(directory.glob('*'))
    posthoc.sort(key=lambda row:row['created'])
    summary['posthoc_checks'] = posthoc
    summary['incomplete_checks'] = incomplete
    summary['unresolved_attempts'] = len(incomplete)
    summary['qwen_calls'] += len(posthoc)
    summary['qwen_tokens'] += sum(row['usage'].get('total_tokens',0) for row in posthoc)
    write(OUT/'evidence/summary.json',summary)
    body = ['# 目的检查与退离预算修复', '',
            '**结论：灶台在线完整通过；酒瓶实际松手→退离→归位及独立五连验收通过，原在线最后视觉判错。最终检查修正后，在同一真实终态上复核通过，没有再跑归位。不能称最终统一版整链联测通过。**', '',
            '实际旧提示词仍添加了“仅悬在上方不算”和单独承托门槛，未落实用户要求。新检查仅输出目的区域是否达到、是否仍夹持/接触、简短依据。接管与最终放置验收分开。', '',
            '灶台旧120周期仅走完约14/20个2mm路径段，并未停滞。新上限为 `min(480,max(120,16×路径段数+40))`，40周期无误差改善停止，保留路径筛查和到达容差。原归位控制器和streaming执行代码未变。', '',
            '酒瓶进一步暴露开爪前退离门槛问题：一侧指尖接近架面，实测开口约标准92%，关节指令误差约1度。近全开且手指/手臂停稳时允许从实际姿态规划有筛查的退离，继续全开命令；这不是松手成功证据，归位前仍要求精确标准开口和无障碍。', '',
            '最终接管采用原子恢复：首次目的信号为真后连续开爪、退离，按实时开口/速度/RGBD几何筛查进入原归位。中途不反复用波动的视觉目的判断打断恢复；最终重新拍照验收，失败仍保留，不拿初次目的判定冒充最终成功。', '',
            '酒瓶最终视觉修正：改用清晰主视图；子任务使用catalog已有自然语言，内部region ID不作为额外视觉要求；删除“目的地区域/层板目标位置”措辞，字面核对是否在架子上，开启1024预算的Qwen短推理。最后回答goal_achieved=true、holding_or_contact=false。视角/措辞/推理多次修正，不能分别归因。', '',
            '| 运行 | 子任务 | 实际开爪周期 | 退离周期 | 归位周期 | 最终归位 | 完整验收 | 停止原因 |',
            '|---|---|---:|---:|---:|---|---|---|']
    for row in rows:
        counts = row['counts'] or {}
        body.append(f"| {row['run']} | {row['title']} | {row['actual_open']} | {counts.get('movement',0)} | {counts.get('home',0)} | {row['home_ready']} | {row['status']} | {row['reason']} |")
    body += ['', f"实际Qwen调用 {summary['qwen_calls']} 次、记录usage {summary['qwen_tokens']} tokens；子agent0、新VLA动作0。失败及修复版本均保留。", '',
             '录像前段为原动作真实回放，后段为本轮新增收尾动作；阶段计数不把回放计为新VLA。独立目标、接触、归位从oracle日志验收，没有输入Qwen。', '',
             '酒瓶与灶台分别在修复过程中验证；灶台已有成功后复用，不为了统一源码版本重复归位。各运行绑定对应冻结源码，不能当成最终同版两任务或组合联测。碗、番茄罐及演示自动收尾没有在本轮启用。', '',
             '[上轮实际完整提示词](evidence/actual_old_prompt.txt)；[本轮逐阶段原始摘要与模型答复](evidence/summary.json)。', '']
    for row in rows: body += [f"## {row['run']}：{row['title']}",'',f"![{row['title']}]({ROOT/row['video']})",'']
    body += ['## 当前终态复核', '', '| 复核版本 | 目的达成 | 仍夹持/接触 | 判定 | 新机器人动作 |', '|---|---|---|---|---:|']
    for row in posthoc:
        verdict = row['two_question_verdict']
        body.append(f"| {row['run']} | {verdict.get('goal_achieved')} | {verdict.get('holding_or_contact')} | {row['decision']['state']} | 0 |")
    body += ['', '[最终实际完整提示词（文字原样，图片以文件/哈希索引表示）](posthoc_primary_literal_thinking/complete_prompt.txt)。',
             '补充番茄罐负例的一次启动没有保留可核实的模型响应，结果及usage未知，不计作检查通过、不重试；上方调用/token数只统计有原始响应的记录。' if incomplete else '', '']
    (OUT/'README.md').write_text('\n'.join(body),encoding='utf-8')
    keep += [OUT/'README.md',PLAN,ROOT/'First_Phase/scene_demo/plans/2026-10-11-home-primary-existing.json']+list((OUT/'evidence').glob('*'))
    keep = sorted(set(p for p in keep if p.is_file() and p.name != 'manifest.json'))
    write(OUT/'evidence/manifest.json',{'tracked_files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in keep]})
    print(json.dumps({'physical_runs':len(rows),'posthoc_checks':len(posthoc),'qwen_calls':summary['qwen_calls'],'qwen_tokens':summary['qwen_tokens'],'latest_results_by_case':summary['latest_results_by_case']},ensure_ascii=False))


if __name__ == '__main__': main()
