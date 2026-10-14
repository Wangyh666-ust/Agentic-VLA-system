"""New bounded combo pilot: visual-associated side grasp and open withdrawal."""
import argparse
import json
import zipfile
from . import combo_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT/'First_Phase/scene_demo/results/2026-10-11-assisted-combos'
PLAN = ROOT/'First_Phase/scene_demo/plans/2026-10-11-assisted-combos.json'
SOURCES = list(dict.fromkeys(pilot.SOURCES + [
    'First_Phase/scene_demo/finish_v1/assisted_combo_pilot.py',
    'First_Phase/scene_demo/finish_v1/calibrated_grasp.py',
] + ['First_Phase/scene_demo/'+name+'.py' for name in
     ('collision_grasp','collision_geometry','local_grasp','side_grasp','preparation_diagnostics','placement_experiments')]))


def prepare():
    if PLAN.exists() or OUT.exists():
        raise RuntimeError('Do not overwrite existing campaign')
    proposal = json.loads((ROOT/'First_Phase/scene_demo/plans/2026-10-11-online-150-300-combos.json').read_text())
    registry = json.loads((ROOT/'First_Phase/scene_demo/experiment_registry.json').read_text())
    ids = [entry['id'] for entry in registry['entries'] if set(entry['question_keys']) &
           {'release-stop','visual-completion','joint-home-handoff','local-grasp','collision-approach','grasp-adaptation'}]
    ids = list(dict.fromkeys(ids + ['20261009-collision-approach','20261009-collision-chain','20261009-assist-repair']))
    proposal.update(id='20261011-assisted-combos',
        purpose='验证碗→酒瓶的视觉关联/几何校准能真实触发旧有效侧抓辅助，及碗→灶台能边开爪边安全退离，再调用原归位；不开子agent、不微调。',
        prior_evidence=ids,
        history_review=[{'experiment_id':i,'disposition':'reuse','reason':
            '复用原归位和旧几何侧抓/避碰路线；最近在线酒瓶空抓及灶台开口阻断是待修的不同故障。旧成功不证明新视觉关联或同时开爪退离成功；旧检查器误报保留为风险。'} for i in ids],
        hypothesis='侧抓前确认目标并校准到已验证相对位姿可改善空抓；解除灶台接触时允许受筛查的同时开爪退离可解除入口死锁。',
        information_gain='各一条真实连续序列，区分视觉辅助触发/实际夹持抬升/放架与同时开爪退离/实测开口/原归位；非相同随机流配对，不能分离多个干预的贡献。',
        success_criteria='酒瓶第一次近距离闭爪提议先请求当前同步wrist+side图/编码器，视觉定位与投影几何关联通过、实际helper>0、confirmed且瓶子抬升；灶台开口受限时concurrent_open_retreat触发且actual open_and_retreat>0，之后实测标准开口并进入原home。每子任务只在实际VLA150/300检查，收尾后全部固定literal目标、无持物/手接触、原归位连续五个真实样本均通过；前碗成果保留。',
        failure_criteria='应触发辅助却零动作、视觉关联拒绝或未发生同时开爪退离记触发链路未完成；执行后未夹到/目标不成立/退离或归位失败分别记效果失败。不得用归位替代目标物体成功，不放宽独立判据。',
        unknown_criteria='视觉null/协议失败/初始化不明停止并保留错误；未启动不计VLA失败；只有视觉定位关联并非真机RGBD六维估计。',
        changed_factors=[
            {'name':'wine_pregrasp','before':'native VLA only; close proposals misaligned','after':'one dual-view visual association + current simulator geometry calibration, validated collision side-grasp helper then fresh-policy transport','reason':'修正抓取位姿，保留旧有效避碰与保护检查'},
            {'name':'turn_release','before':'exact/near-full opening required before movement; one finger constrained','after':'after failed opening allow only turn tasks to withdraw with explicit open command and measured/half/open CAD-screened path; exact original home gate remains','reason':'解除接触阻止张开、张开阻止退离的循环'},
            {'name':'action_accounting','before':'job.steps counts only native VLA because no grasp helper','after':'separate actual VLA/local grasp/finish; checks at actual VLA150/300; max300 VLA+200 grasp per subtask','reason':'辅助动作不能提前触发完成检查'}],
        fixed_factors={'home':'same home controller/streaming .01rad kp50 max360 five samples and exact-open entry',
            'checker':'unchanged user-approved generic prompt; current wrist+side and synchronized encoder; 150/300 actual VLA actions',
            'environment':'goal_table LIBERO goal8, Panda robosuite/MuJoCo seed0 init0 one reset per sequence',
            'policy':'same pretrained SmolVLA weights, no training or environment reset between subtasks',
            'comparison':'historical condition reference only; policy random stream not paired'} ,
        scope={'max_cases':2,'max_vla_actions_per_case':600,'max_helper_actions_per_case':1960,
            'stop_rule':'各组合一次；先核对首碗通过才执行第二技能。首碗未通过停止其后续及另一组合。酒瓶预抓分支未触发/拒绝/失败不继续下游放架调参；首碗已通过时灶台为独立故障试点仍可继续。灶台同时开爪退离未执行则不单测归位。无现场重试，不微调、不扩大、不开子agent。'},
        new_physical_actions=5120,
        repeat={'needed':False,'reason':'new bounded visual pregrasp and concurrent turn withdrawal interventions'})
    proposal['configuration'].update(source_sha256={s:pilot.sha(ROOT/s) for s in SOURCES},
        wine_helper='collision_grasp unchanged validated three-waypoint side grasp',
        pregrasp='qwen3-vl-plus, temperature0, thinking false, max1 call; no simulator pose in prompt; projected association tolerance .03 normalized',
        geometry_source='precise pose, orientation and contact protection remain simulator geometry; real RGBD estimation deferred',
        grasp_budget_per_subgoal=200, physical_policy_loop_budget_per_subgoal=500,
        vla_budget_per_subgoal=300, check_schedule='actual VLA150/300; local grasp counted separately',
        release='turn-only measured/half/open shape-screened concurrent opening after stationary20 exhausted; exact home gate unchanged',
        execution='single primary agent + local tools; user explicitly forbids subagents; no claimed model effort override',
        max_session_actions=2000,
        authorization='latest user authorizes fixes and two small sequences, no subagents/no training')
    pilot.write(PLAN,proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT/source,source)
    print('Prepared assisted combos; zero physical/model calls',flush=True)


def run(case):
    if case == 'bowl_stove':
        first = json.loads((OUT/'bowl_wine/result.json').read_text())
        if not first['subtasks'] or not first['subtasks'][0]['outcome'].get('ok'):
            raise RuntimeError('First bowl failed; dependent combo stopped')
    from .calibrated_grasp import CalibratedAssistService
    pilot.OUT, pilot.PLAN = OUT, PLAN
    pilot.run(case, service_class=CalibratedAssistService, budget_per_subgoal=500)


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--case',choices=pilot.CASES)
    args=parser.parse_args()
    prepare() if args.prepare else run(args.case)


if __name__ == '__main__':
    main()
