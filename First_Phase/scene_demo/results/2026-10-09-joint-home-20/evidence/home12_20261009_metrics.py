from pathlib import Path
import argparse, csv, hashlib, io, json

ROOT = Path(r'D:\FYP')
TMP = ROOT / 'First_Phase/tmp'
SCENE = ROOT / 'First_Phase/scene_demo'
RESULT = SCENE / 'results/2026-10-09-joint-home-20'
REPORT = RESULT / 'evidence/summary12.json'
RUNNER = TMP / 'home20_20261009_run.py'
METRICS = RESULT / 'evidence/metrics12.json'
CSV = RESULT / 'evidence/case_summary12.csv'
PROOF = TMP / 'home12_20261009_metrics_proof.json'
GOLD = {
    'bowl_to_plate': ['on', 'akita_black_bowl_1', 'plate_1'],
    'wine_to_rack': ['on', 'wine_bottle_1', 'wine_rack_1_top_region'],
    'stove_on': ['turnon', 'flat_stove_1'],
    'soup_to_basket': ['in', 'alphabet_soup_1', 'basket_1_contain_region'],
    'sauce_to_basket': ['in', 'tomato_sauce_1', 'basket_1_contain_region'],
    'white_mug_left': ['on', 'porcelain_mug_1', 'plate_1'],
    'yellow_mug_right': ['on', 'white_yellow_mug_1', 'plate_2'],
}
COLUMNS = ['case_id', 'scene_id', 'init_state_index', 'model_seed', 'capability_ids', 'success', 'reason', 'failure_category', 'vla_actions', 'home_actions', 'total_actions', 'stages_measured', 'home_attempts', 'home_ready', 'continuation_jobs', 'continuation_successes']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def windows_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('source path missing')
    if value.startswith('/mnt/d/'):
        return Path('D:/' + value[len('/mnt/d/'):])
    return Path(value)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def write_fresh(path, data):
    with path.open('xb') as file:
        file.write(data)

def encode(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n').encode('utf-8')

def streak(values):
    tail = maximum = 0
    for value in values:
        tail = tail + 1 if value is True else 0
        maximum = max(maximum, tail)
    return {'true_samples': sum(v is True for v in values), 'false_samples': sum(v is False for v in values), 'unknown_samples': sum(type(v) is not bool for v in values), 'max_streak': maximum, 'tail_streak': tail}

def main(argv=None):
    argparse.ArgumentParser(description='Read-only mechanical metrics for completed home20 campaign.').parse_args(argv)
    require(not any(p.exists() for p in (METRICS, CSV, PROOF)), 'fresh outputs required; no overwrite')
    frozen = {}
    d = {'ok': False, 'metrics_invocations': 1, 'physical_calls': 0, 'model_calls': 0, 'api_calls': 0, 'hermes_calls': 0, 'git_mutations': 0}
    try:
        def read_bytes(path):
            path = Path(path)
            raw = path.read_bytes()
            key = str(path)
            fingerprint = {'bytes': len(raw), 'sha256': sha(raw)}
            if key in frozen:
                require(frozen[key] == fingerprint, 'source changed during read: ' + key)
            frozen[key] = fingerprint
            return raw
        report = json.loads(read_bytes(REPORT).decode('utf-8-sig'))
        read_bytes(RUNNER)
        require(report.get('ok') is True and report.get('executed_cases') == 12 and report.get('planned_cases') == 12, 'completed protocol-OK 12-case report required')
        require(isinstance(report.get('cases'), list) and len(report['cases']) == 12 and report.get('not_run') == [], '12 measured cases and no not_run required')
        cases_out, csv_rows = [], []
        groups = {'by_scene': {}, 'by_sequence': {}}
        totals = dict(attempted_cases=12, physical_successes=0, first_subtask_failures=0, continuation_subtask_failures=0, home_failures=0, home_attempts=0, home_ready=0, after_home_jobs=0, after_home_job_successes=0, effective_home_continuation_jobs=0, effective_home_continuation_successes=0, vla_actions=0, home_actions=0, total_physical_actions=0)
        for case in report['cases']:
            spec = case['spec']
            require(case['case_id'] == spec['case_id'], 'case identity mismatch')
            capabilities = spec['capability_ids']
            require(all(c in GOLD for c in capabilities), 'unknown registered capability')
            literal_gold = [GOLD[c] for c in capabilities]
            require(spec['gold_goals'] == literal_gold, 'independent literal gold mismatch')
            stages, observations = case['stages'], []
            measured_vla = measured_home = home_attempts = home_ready = continuation_successes = effective_jobs = effective_successes = 0
            for index, stage in enumerate(stages):
                cap = stage['capability_id']
                require(cap == capabilities[index], 'stage capability order mismatch')
                n = stage['job']['steps']
                require(type(n) is int and 0 <= n <= 336, 'invalid measured job steps')
                ac = stage['assistcounts']
                require(ac.get('vla_source_actions') == n and ac.get('prepare_source_actions') == 0 and ac.get('local_grasp_source_actions') == 0, 'actual action sources are not all new VLA')
                path = windows_path(stage['observations_path'])
                require(path.is_absolute() and not path.is_symlink() and path.resolve().is_relative_to((RESULT / 'pilot').resolve()), 'observations path outside fresh campaign')
                lines = read_bytes(path).decode('utf-8-sig').splitlines()
                require(len(lines) == n, 'observation row count != job.steps: ' + str(path))
                rows = [json.loads(line) for line in lines]
                require(all(type(row.get('step')) is int and row['step'] == j for j, row in enumerate(rows, 1)), 'observation steps not exact 1..n')
                require(all(isinstance(row.get('snapshot'), dict) for row in rows), 'snapshot not dict')
                key = '|'.join(GOLD[cap])
                object_id = GOLD[cap][1]
                predicate_values = [r['snapshot'].get('predicates', {}).get(key) for r in rows]
                object_entries = [r['snapshot'].get('objects', {}).get(object_id) for r in rows]
                objects_present = any(isinstance(o, dict) for o in object_entries)
                grasped = [o.get('grasped') if isinstance(o, dict) else None for o in object_entries]
                current_ready = [r.get('current_ready', {}).get('ready') for r in rows]
                final_ready = [r.get('final_ready', {}).get('ready') for r in rows]
                data = {
                    'capability_id': cap, 'literal_gold': GOLD[cap], 'job': stage['job'], 'score': stage['score'],
                    'observations_path': str(path), 'observations_sha256': frozen[str(path)]['sha256'], 'row_count': len(rows),
                    'current_predicate_ever_true': any(v is True for v in predicate_values),
                    'current_predicate_final': predicate_values[-1] if predicate_values else None,
                    'current_predicate_unknown_samples': sum(type(v) is not bool for v in predicate_values),
                    'object_grasped_true_samples': sum(v is True for v in grasped) if objects_present else None,
                    'object_grasped_unknown_samples': sum(type(v) is not bool for v in grasped) if objects_present else None,
                    'held_and_goal_true_samples': sum(g is True and p is True for g, p in zip(grasped, predicate_values)) if objects_present else None,
                    'grasp_observation_definition': 'snapshot.objects[target_object].grasped; absent static object yields null',
                    'current_ready': streak(current_ready), 'final_ready': streak(final_ready),
                    'assistcounts': stage.get('assistcounts'), 'home_after': stage.get('home_after'),
                }
                observations.append(data)
                measured_vla += n
                if 'home_after' in stage:
                    h = stage['home_after']
                    require(isinstance(h, dict) and type(h.get('actions')) is int and 0 <= h['actions'] <= 360, 'invalid actual home record')
                    measured_home += h['actions']
                    home_attempts += 1
                    home_ready += h.get('ready') is True and h.get('restore', {}).get('ok') is True
                success = stage['job'].get('success') is True and stage['score'].get('physical_gold_success') is True
                if index > 0:
                    continuation_successes += success
                    prior_home = stages[index - 1].get('home_after')
                    if isinstance(prior_home, dict) and prior_home.get('ready') is True and prior_home.get('restore', {}).get('ok') is True:
                        effective_jobs += 1
                        effective_successes += success
            require(case['vla_actions'] == measured_vla and case['home_actions'] == measured_home and case['total_physical_actions'] == measured_vla + measured_home, 'case action totals differ from stage observations')
            require(case.get('assist_mode') == 'disabled' and case.get('new_hermes_calls') == 0, 'configuration differs from registered no-assist/no-Hermes')
            record = {'case_id': case['case_id'], 'spec': spec, 'physical_success': case['physical_success'], 'failure_category': case.get('failure_category'), 'failure_reason': case.get('failure_reason'), 'unknown': case.get('unknown'), 'operational_errors': case.get('operational_errors'), 'stages': observations, 'vla_actions': measured_vla, 'home_actions': measured_home, 'total_physical_actions': measured_vla + measured_home, 'home_attempts': home_attempts, 'home_ready': home_ready, 'continuation_jobs': max(0, len(stages) - 1), 'continuation_successes': continuation_successes, 'effective_home_continuation_jobs': effective_jobs, 'effective_home_continuation_successes': effective_successes}
            cases_out.append(record)
            csv_rows.append(dict(zip(COLUMNS, [case['case_id'], spec['scene_id'], spec['init_state_index'], spec['model_seed'], '|'.join(capabilities), case['physical_success'], case.get('failure_reason'), case.get('failure_category'), measured_vla, measured_home, measured_vla + measured_home, len(stages), home_attempts, home_ready, max(0, len(stages) - 1), continuation_successes])))
            totals['physical_successes'] += case['physical_success'] is True
            totals['first_subtask_failures'] += case.get('failure_category') == 'first_subtask'
            totals['continuation_subtask_failures'] += case.get('failure_category') == 'continuation_subtask'
            totals['home_failures'] += case.get('failure_category') == 'home'
            for k in ('home_attempts', 'home_ready', 'vla_actions', 'home_actions', 'total_physical_actions', 'effective_home_continuation_jobs', 'effective_home_continuation_successes'):
                totals[k] += record[k]
            totals['after_home_jobs'] += record['continuation_jobs']
            totals['after_home_job_successes'] += continuation_successes
            for grouping, group_key in (('by_scene', spec['scene_id']), ('by_sequence', '|'.join(capabilities))):
                g = groups[grouping].setdefault(group_key, {'attempted_cases': 0, 'physical_successes': 0, 'first_subtask_failures': 0, 'continuation_subtask_failures': 0, 'home_failures': 0, 'vla_actions': 0, 'home_actions': 0, 'home_attempts': 0, 'home_ready': 0, 'effective_home_continuation_jobs': 0, 'effective_home_continuation_successes': 0})
                g['attempted_cases'] += 1
                g['physical_successes'] += case['physical_success'] is True
                g['first_subtask_failures'] += case.get('failure_category') == 'first_subtask'
                g['continuation_subtask_failures'] += case.get('failure_category') == 'continuation_subtask'
                g['home_failures'] += case.get('failure_category') == 'home'
                for k in ('vla_actions', 'home_actions', 'home_attempts', 'home_ready', 'effective_home_continuation_jobs', 'effective_home_continuation_successes'):
                    g[k] += record[k]
        compared = ['physical_successes', 'first_subtask_failures', 'continuation_subtask_failures', 'home_failures', 'home_attempts', 'home_ready', 'after_home_jobs', 'after_home_job_successes']
        checks = {k: report.get(k) == totals[k] for k in compared}
        require(all(checks.values()), 'summary count differs from frozen report: ' + ','.join(k for k, v in checks.items() if not v))
        totals['case_success_fraction'] = {'numerator': totals['physical_successes'], 'denominator': 12}
        totals['effective_home_continuation_success_fraction'] = {'numerator': totals['effective_home_continuation_successes'], 'denominator': totals['effective_home_continuation_jobs']}
        after = {p: {'bytes': Path(p).stat().st_size, 'sha256': sha(Path(p).read_bytes())} for p in frozen}
        require(after == frozen, 'raw source bytes changed during statistics')
        metric = {'schema_version': 1, 'configuration': {'all_vla_jobs_are_new_rollouts': True, 'historical_action_replay': False, 'new_hermes_calls': 0, 'assist_mode': 'disabled', 'helper_actions': 0, 'vla_cap': 336, 'home_cap': 360, 'attempted_cases_denominator': 12, 'effective_home_jobs_denominator_separate': True, 'no_causal_conclusions': True}, 'report_sha256': frozen[str(REPORT)]['sha256'], 'runner_sha256': frozen[str(RUNNER)]['sha256'], 'totals': totals, 'groups': groups, 'cases': cases_out, 'source_files': frozen, 'report_count_checks': checks, 'raw_unchanged': True}
        text = io.StringIO(newline='')
        writer = csv.DictWriter(text, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        write_fresh(METRICS, encode(metric))
        write_fresh(CSV, text.getvalue().encode('utf-8'))
        d.update(ok=True, status='HOME12_MECHANICAL_METRICS_PASS', cases=12, stages=sum(len(c['stages']) for c in cases_out), totals=totals, source_files_before=frozen, source_files_after=after, raw_unchanged=True, sumcounts_match_report=True, report_count_checks=checks, metrics_sha256=sha(METRICS.read_bytes()), csv_sha256=sha(CSV.read_bytes()), report_sha256=frozen[str(REPORT)]['sha256'], runner_sha256=frozen[str(RUNNER)]['sha256'])
    except Exception as exc:
        d.update(status='HOME12_METRICS_FAILURE_PRESERVED', error=repr(exc), source_files_before=frozen)
    write_fresh(PROOF, encode(d))
    print(json.dumps({k: d.get(k) for k in ('status', 'ok', 'cases', 'stages', 'totals', 'raw_unchanged', 'sumcounts_match_report', 'metrics_sha256', 'csv_sha256', 'error')}))
    return 0 if d['ok'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
