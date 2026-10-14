"""Bounded correction of the hypothetical return-screen start, one combo."""
import argparse
import json
import zipfile
from . import stove_exit_repair_pilot as previous
from . import combo_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-stove-candidate-start'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-stove-candidate-start.json'
SOURCES = previous.SOURCES + ['First_Phase/scene_demo/finish_v1/stove_candidate_start_pilot.py']


def prepare():
    if PLAN.exists() or OUT.exists():
        raise RuntimeError('Do not overwrite an existing campaign')
    proposal = json.loads(previous.PLAN.read_text())
    proposal['prior_evidence'].append('20261011-stove-exit-repair')
    proposal['history_review'].append({'experiment_id': '20261011-stove-exit-repair',
        'disposition': 'reuse', 'reason': '实际开爪后退离0：假想回程误加入当前接触姿态为第0点。保留失败，仅纠正候选起点，原执行home默认实测起点保持。'})
    proposal.update(id='20261011-stove-candidate-start',
        purpose='修正候选退出回程的起点语义，让碗→灶台真正退离并接原归位；一个组合，原失败保留。',
        hypothesis='假想回程显式从候选退出姿态采样，不插入尚未退离的当前位置，可避免错误阻断；实际退出和原home继续检查实测起点。',
        information_gain='上一首修订退离0，未验证退出端回程资格的物理效果；本次仅改假想路径起点，不删碰撞、不改原home执行。',
        changed_factors=[{'name': 'hypothetical_return_start',
            'before': 'home_screen prepends actual current contact pose to candidate return',
            'after': 'candidate_home_screen uses scratch path with explicit start_q=candidate; actual exit and home keep default measured start',
            'reason': '真实接口的默认起点与候选预测语义不同'}],
        repeat={'needed': False, 'reason': 'Corrected hypothetical start; prior zero-target trial preserved'})
    proposal['configuration'].update(source_sha256={s: pilot.sha(ROOT / s) for s in SOURCES},
        candidate_screen='explicit hypothetical start_q only; no contact exemptions, actual return default unchanged')
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for s in SOURCES:
            archive.write(ROOT / s, s)
    print('Prepared candidate-start correction; zero actions and model calls', flush=True)


def run():
    previous.OUT, previous.PLAN = OUT, PLAN
    previous.run()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == '__main__':
    main()
