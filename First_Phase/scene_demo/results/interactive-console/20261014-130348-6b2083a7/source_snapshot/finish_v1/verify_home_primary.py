"""One changed-view check of the actual successful home endpoint; no motion."""
import argparse
import copy
import json
import subprocess
import sys
from .prepare_goal_pilot import ROOT, OUT
from .prepare_simple_pilot import sha, write
from .simple_checker import SimpleQwenClient, decide

PLAN = ROOT/'First_Phase/scene_demo/plans/2026-10-11-home-primary-existing.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--revision',default='')
    parser.add_argument('--negative',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        registry = json.loads((ROOT/'First_Phase/scene_demo/experiment_registry.json').read_text())
        history = [e for e in registry['entries'] if 'release-stop' in e['question_keys']]
        write(PLAN,{'schema_version':1,'id':'20261011-home-primary-existing','intent':'analyze_existing','question_key':'release-stop',
                    'purpose':'已真实完成酒瓶开爪/退离/home并独立五连通过，复用其实际终态与同步编码器，移除被手臂遮挡的侧视图修正最后视觉判读，不重复归位。',
                    'prior_evidence':[e['id'] for e in history],
                    'history_review':[{'experiment_id':e['id'],'disposition':'reuse','reason':'复用已归位原始真实末帧与同步反馈；只改变最终检查视角，保留错误原判断，不充当新物理实验。'} for e in history],
                    'new_physical_actions':0,
                    'analysis_steps':['读取本轮wine_atomic原始最终请求和actual post五连验收。','按最终运行期同一规则保留agentview两时刻原图及其同步编码器，无对象oracle进入模型。','最多一次新增Qwen调用；保存完整原始请求/响应、版本和usage，未知/误判不补造。','新检查通过只说明终态复核修正，不改原blocked结果或宣称最终同版全链联测。'],
                    'configuration':{'changed_condition':'home place: two main original frames instead of main+occluded side',
                                     'runtime_sha256':sha(ROOT/'First_Phase/scene_demo/finish_v1/runtime.py')}})
        print(str(PLAN));return
    review = ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'
    subprocess.run([sys.executable,'-B',str(review),'check','--proposal',str(PLAN)],cwd=ROOT,check=True)
    request = (ROOT/'First_Phase/scene_demo/results/2026-10-10-simple-checker/pilot_sauce/sauce_negative/runtime_checks/model_requests/call_01_request.json'
               if args.negative else OUT/'pilot_wine_atomic/wine_exit/runtime_checks/model_requests/call_02_request.json')
    raw = json.loads(request.read_text())
    context = copy.deepcopy(raw['context'])
    chosen = [(fr,img) for fr,img in zip(context['sensor_frames'],raw['images']) if args.negative or fr['view']=='agentview']
    context['sensor_frames'] = [dict(fr,frame=i) for i,(fr,_) in enumerate(chosen)]
    paths = [img['path'] for _,img in chosen]
    for _,img in chosen:assert sha(__import__('pathlib').Path(img['path']))==img['sha256']
    directory = OUT/('posthoc_primary'+args.revision)
    directory.mkdir(exist_ok=False)
    client = SimpleQwenClient(max_calls=1)
    client.audit_dir = directory
    result = client.analyze(context,paths)
    result['decision'] = decide(context,result['verdict'])
    result['source_request_sha256'] = sha(request)
    result['context'] = context
    result['robot_actions'] = 0
    result['case'] = 'sauce_negative' if args.negative else 'wine_home'
    result['limitation'] = 'changed final-view recheck of existing physical endpoint, not new whole-flow run'
    write(directory/'result.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
