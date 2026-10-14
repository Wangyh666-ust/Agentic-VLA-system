"""Archive actual two-step results and selected source inputs, never rerun motion."""
import json
from pathlib import Path
from .combo_pilot import ROOT, OUT, PLAN, sha, write


def main():
    import imageio.v2 as imageio
    rows, keep = [], [PLAN, OUT/'source_snapshot.zip', OUT/'startup_proposal.json', OUT/'startup_source.zip', OUT/'startup_failure.log']
    for name in ('bowl_wine','bowl_stove'):
        directory = OUT/name
        if not (directory/'result.json').exists(): continue
        result = json.loads((directory/'result.json').read_text())
        writer = imageio.get_writer(str(directory/'sequence.mp4'),fps=20)
        jobs = []
        try:
            for sub in result['subtasks']:
                job = sub['job']; run = Path(job['run_dir']); finish = sub['finish'] or {}
                actions = [json.loads(line) for line in (run/'finish_actions.jsonl').read_text().splitlines()] if (run/'finish_actions.jsonl').exists() else []
                opening = sum(row.get('phase')=='release' and row.get('finger_target') is not None for row in actions)
                independent = [json.loads(line) for line in (run/'independent_finish.jsonl').read_text().splitlines()] if (run/'independent_finish.jsonl').exists() else []
                checks = []
                for request in sorted((run/'finish/model_requests').glob('*_request.json')):
                    raw = json.loads(request.read_text()); response = json.loads(request.with_name(request.name.replace('_request','_response')).read_text())
                    checks.append({'content':response['choices'][0]['message']['content'],'usage':response.get('usage'), 'request_path':request.relative_to(ROOT).as_posix()})
                    text = raw['prompt']+'\n'+'\n'.join(f"Image frame={fr['frame']}; view={fr['view']}; timestamp={fr['timestamp']}\n[image: {img['path']}; sha256={img['sha256']}]" for fr,img in zip(raw['context']['sensor_frames'],raw['images']))+'\n'
                    prompt = request.with_suffix('.txt');prompt.write_text(text,encoding='utf-8');keep.append(prompt)
                    keep += [Path(img['path']) for img in raw['images']]
                def refs(value):
                    if isinstance(value,dict):
                        for key,item in value.items():
                            if key in ('rgb_path','depth_path','calibration_path','json_path') and isinstance(item,str) and Path(item).is_file():keep.append(Path(item))
                            refs(item)
                    elif isinstance(value,list):
                        for item in value:refs(item)
                refs(finish)
                count,last=0,None
                if job['rollout_path']:
                    reader=imageio.get_reader(job['rollout_path'])
                    try:
                        for frame in reader:writer.append_data(frame);count+=1;last=frame
                    finally:reader.close()
                    imageio.imwrite(run/'final_frame.png',last)
                jobs.append({'capability':sub['capability'],'new_vla':job['steps'],'success':job['success'],
                             'reason':job['ended_reason'],'actual_open':opening,'counts':finish.get('counts',{}),
                             'home_streak':finish.get('home_streak',0),'independent_streak':sub['independent_streak'],
                             'final_independent':independent[-1] if independent else None,
                             'finish_reason':finish.get('reason'),'qwen':checks,'decoded_frames':count})
        finally:writer.close()
        row={'case':name,'state':result['plan']['state'],'plan_success':result['plan']['plan_success'],'jobs':jobs,
             'wall_seconds':result['wall_seconds'],'timings':result['timings'],'video':(directory/'sequence.mp4').relative_to(ROOT).as_posix()}
        rows.append(row)
        keep += [p for p in directory.rglob('*') if p.is_file() and 'sensors' not in p.parts]
        keep += [OUT/(name+'_launch.log')]
    summary={'cases':rows,'new_vla':sum(j['new_vla'] for row in rows for j in row['jobs']),
             'qwen_calls':sum(len(j['qwen']) for row in rows for j in row['jobs']),
             'qwen_tokens':sum(c['usage'].get('total_tokens',0) for row in rows for j in row['jobs'] for c in j['qwen']),
             'replay_actions':0,'subagents':0,'scope':'fixed user-specified order; no new Hermes planning invocation'}
    summary['not_run'] = {'bowl_stove': 'common first bowl step failed independent acceptance; stop rule'} if not (OUT/'bowl_stove/result.json').exists() else {}
    negative = OUT/'negative_recheck'
    if (negative/'result.json').exists():
        diag = json.loads((negative/'result.json').read_text())
        summary['negative_recheck'] = {'two_question_verdict':diag['two_question_verdict'],'usage':diag['usage'],'robot_actions':0,'effect':'still false-positive; added grounding sentence reverted'}
        summary['qwen_calls'] += 1
        summary['qwen_tokens'] += diag['usage'].get('total_tokens',0)
        keep += list(negative.glob('*'))
    keep += list((OUT/'startup_failure').glob('*'))
    write(OUT/'summary.json',summary);keep.append(OUT/'summary.json')
    body=['# 通用检查器两步组合试点','',
          '**结论：本轮没有组合通过。碗→酒瓶的首步骤在第60个新VLA动作被Qwen误判到位而提前松手；实际归位完成，但碗不在盘上，独立验收阻断酒瓶。碗→灶台因相同首步骤未通过而未启动。不能称酒瓶/灶台或归位失败，也不能称两个组合已经测完。**','',
          '本轮直接使用预训练SmolVLA进行新执行，不回放历史动作。两个顺序由用户指定；没有新增Hermes规划调用，不能据此衡量规划增益。每组合场景仅初始化一次，子任务之间保留物体状态，真实松手/退离/归位，重置策略队列并读取新观察。', '',
          '| 组合 | 子任务 | 新VLA | 实际开爪周期 | 退离 | 归位 | 独立连续达标 | 结果 |',
          '|---|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        for job in row['jobs']:
            body.append(f"| {row['case']} | {job['capability']} | {job['new_vla']} | {job['actual_open']} | {job['counts'].get('movement',0)} | {job['counts'].get('home',0)} | {job['independent_streak']} | {job['reason']} / {job['finish_reason']} |")
    body += ['', '## 判据与限制','',
             'Qwen检查只接收子任务自然语言、当前图像及同步编码器，不接收物体谓词/接触真值。首次目的true触发收尾；结束后再检查目的与分离。独立效果使用运行前固定的literal目标，并检查全部已执行目标、无持物、无手接触、归位连续五个真实post；不会把Hermes计划作为标准答案。原生精确位置分和视觉语义分分别保留。这次主/侧视图明确可见盘子与碗分开，不是精确位置阈值过严或额外支撑门槛。',
             '在线定时检查每60个VLA动作一次，不使用原生谓词首次为真作为检查触发器；最多5次前检查及1次最终检查。应触发但未触发时不执行依赖交接。首碗失败时另一组合不继续；若首碗通过而酒瓶失败，碗→灶台是独立后续技能对照。',
             '归位源码及参数保持46f197e。0.01rad路径点、20Hz约0.2rad/s目标推进，加kp50阻尼跟踪、到位停稳和五采样确认；此外RGBD采集/几何筛查/最终Qwen请求增加墙钟时间，不能把这些全算为关节运动慢。实际调用耗时见summary.timings。未做加速对照。',
             '视频按既有service保存：VLA每动作一帧，辅助动作每5步一帧，统一20fps，收尾视觉上加速；不包含模型等待，不代表真实墙钟速度。',
             '首次启动把完成模式写成release，构造器拒绝；物理/模型调用为零。随后仅修正为实际支持的release_verified再启动，原日志/方案/源码保留。',
             '首例失败后，仅对第60步已保存输入增加一条通用物体定位说明做零动作诊断；Qwen仍错误声称腕部/侧视图碗已在盘上。该改词未解决问题，生产提示词恢复用户确认版。此补充分析未另做预登记门禁，记录为事后诊断，不能充当有效性证明；没有再跑归位或新VLA。',
             f"有响应Qwen {summary['qwen_calls']}次/{summary['qwen_tokens']}tokens，新VLA {summary['new_vla']}，历史回放0，子agent0。软件检查24+26项通过，软件通过不代替上表效果。",'']
    for row in rows:
        body += [f"## {row['case']}：{row['state']} / plan_success={row['plan_success']}",'',f"![{row['case']}]({ROOT/row['video']})",'']
    body += ['## 下一步','', '先验证到位检查能拒绝“碗仍在盘旁”这种明确负例，再恢复这两个原定组合；不重新调已成功归位、不以延迟检查到固定动作数掩盖误判、不微调。优先检验视角选择/物体定位，保存原始两问输出和当前输入。', '']
    (OUT/'README.md').write_text('\n'.join(body),encoding='utf-8');keep.append(OUT/'README.md')
    files=sorted(set(p for p in keep if p.is_file()))
    write(OUT/'manifest.json',{'tracked_files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in files]})
    print(json.dumps({'cases':[{ 'case':r['case'],'state':r['state'],'plan_success':r['plan_success'],'jobs':[{k:v for k,v in j.items() if k not in ('qwen','final_independent')} for j in r['jobs']]} for r in rows],'qwen_calls':summary['qwen_calls'],'new_vla':summary['new_vla']},ensure_ascii=False))


if __name__=='__main__':main()
