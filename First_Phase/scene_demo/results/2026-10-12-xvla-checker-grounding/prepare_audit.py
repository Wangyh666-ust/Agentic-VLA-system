"""Prepare nine fixed saved RGB examples without inference or simulation."""
import copy
import hashlib
import json
from pathlib import Path
from scene_demo.xvla_checker import build_prompt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=HERE.parent/'2026-10-12-xvla-persistent-pilot'
JOBS={
 'wine':BASE/'v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4',
 'soup':BASE/'v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33',
 'white':BASE/'v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a',
 'micro':BASE/'v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f'}
SPECS=[('wine_negative','wine',None,1,15,[False]),('wine_arrived','wine',0,91,15,[True]),('wine_final','wine',1,262,15,[True]),
 ('soup_negative','soup',None,1,16,[False]),('soup_final','soup',5,646,16,[True,None]),
 ('white_negative','white',None,1,5,[False]),('white_arrived','white',4,463,5,[True]),
 ('micro_negative','micro',None,1,5,[False]),('micro_arrived','micro',4,463,5,[True,None])]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    cases=[];sources={}
    for name,key,index,start,prepare,allowed in SPECS:
        job=JOBS[key];checkfile=job/'finish/checks.jsonl';checks=rows(checkfile);source=checks[0 if index is None else index]
        context=copy.deepcopy(source['ctx'])
        if index is None:
            native=job.parent/'native/native_steps.jsonl';sources[str(native)]=sha(native)
            matched=[r for r in rows(native) if r['source']=='vla' and r['counts']['vla']==5];assert len(matched)==1
            r=matched[0];assert r['counts']['settling']==10 and r['counts']['prepare']==prepare and r['counts']['finish']==0
            timestamp=r['sim_time'];fingers=r['finger_qpos']
            context['actuator_measurements']['timestamp']=timestamp;context['actuator_measurements']['finger_qpos']=fingers
            context['actuator_measurements']['max_abs_arm_qvel']=max(abs(v) for v in r['arm_qvel'])
            context['sensor_frames']=[{'frame':0,'timestamp':timestamp,'view':'wrist'},{'frame':1,'timestamp':timestamp,'view':'agentview'}]
        else:
            timestamp=context['actuator_measurements']['timestamp'];fingers=source['robot']['finger_qpos']
            assert all(f['timestamp']==timestamp for f in context['sensor_frames'])
            assert all(abs(a-b)<=1e-9 for a,b in zip(fingers,context['actuator_measurements']['finger_qpos']))
        paths={view:str(job/f'finish/sensors/cap{start+offset:06d}_{view}.png') for offset,view in enumerate(['agentview','wrist','sideview'])}
        for path in paths.values():assert Path(path).is_file();sources[path]=sha(path)
        source_paths=source['image_paths'] if index is not None else [paths['wrist'],paths['agentview']]
        assert all(p in paths.values() for p in source_paths)
        capture={'timestamp':timestamp,'frames':[{'view':'wrist','rgb_path':paths['wrist']},{'view':'agentview','rgb_path':paths['agentview']}],
                 'verification_frames':[{'view':'sideview','rgb_path':paths['sideview']}],'robot':{'finger_qpos':fingers}}
        actual=copy.deepcopy(context);actual['sensor_frames']=[{'frame':i,'timestamp':timestamp,'view':view} for i,view in enumerate(['wrist','agentview','sideview'])]
        cases.append({'name':name,'source_context':context,'source_image_paths':source_paths,'capture':capture,'actual_prompt':build_prompt(actual),'expected_allowed':allowed,
                      'explanation':'固定step5负例' if index is None else ('局部遮挡，身份无法辨认允许unknown' if None in allowed else '保存明确视觉正例')})
        sources[str(checkfile)]=sha(checkfile)
    sourcefiles=[ROOT/'First_Phase/scene_demo/xvla_checker.py',HERE/'prepare_audit.py',HERE/'audit.py',ROOT/'First_Phase/scene_demo/catalog.py']
    for p in sourcefiles:sources[str(p)]=sha(p)
    assets=Path('/home/yhwang/fyp/xvla_native/LIBERO/libero/libero/assets')
    for category in ['wine_bottle','porcelain_mug','white_yellow_mug','alphabet_soup','wine_rack','basket','plate','microwave']:
        for directory in assets.glob('*/'+category):
            for p in directory.iterdir():
                if p.is_file() and p.suffix in ['.xml','.png']:sources[str(p)]=sha(p)
    (HERE/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    sources[str(HERE/'cases.json')]=sha(HERE/'cases.json')
    (HERE/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    ids=['20261012-xvla-persistent-pilot','20261011-final-input-audit','20261011-neutral-encoder-checker','20261011-takeover-stove-cue']
    local=[BASE/'summary.json',BASE/'checker_agentview/result.json',BASE/'checker_rack_description/result.json',ROOT/'First_Phase/scene_demo/xvla_checker.py',HERE/'source_manifest.json']
    proposal={'schema_version':1,'id':'20261012-xvla-checker-grounding','intent':'analyze_existing','question_key':'xvla-checker-view',
      'purpose':'验证静态物品定义/三同步视角/单任务on-in语义是否消除保存样例的身份与额外条件误判，零新动作',
      'prior_evidence':ids,'history_review':[{'experiment_id':i,'disposition':'reuse','reason':r} for i,r in zip(ids,['复用目标与终态视觉冲突、保存画面和两次offline失败。','复用原输入审计与同步证据。','复用编码器融合定义，开口不替代目标。','复用灶台静态定义与既有收尾，候选不改在线。'])],
      'new_physical_actions':0,'analysis_steps':['只读九个固定保存case，原SHA和同采样三图核验。','真实Qwen上限9，一case一次，无重试，transport/parse失败停止。','soup/micro遮挡正例允许null，其他样例按预登记allowed判据。','静态外观、三视角、单任务关系三因素共同变化，不能分离归因；随机流未配对。','仅离线检查器效果，不以offline结果宣称online持久链成功。'],
      'local_evidence':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in local]}
    (HERE/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2))
    print(json.dumps({'ok':True,'case_count':len(cases),'source_count':len(sources),'qwen_calls':0,'xvla_calls':0,'hermes_calls':0,'new_physical_actions':0}))


if __name__=='__main__':main()
