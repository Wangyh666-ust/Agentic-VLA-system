from pathlib import Path
import copy,hashlib,json
T=Path(r'D:\FYP\First_Phase\tmp');R=Path(r'D:\FYP\First_Phase\scene_demo\results\2026-10-09-joint-home-20');E=R/'evidence'
REPORT=R/'pilot/report.json';STOP=T/'home12_20261009_stop_proof.json';LAUNCH=T/'home20_20261009_launch_proof.json'
SUMMARY=E/'summary12.json';SCOPE=E/'scope12.json';PROOF=T/'home12_20261009_derive_proof.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(x,msg):
    if not x:raise ValueError(msg)
def fresh(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def main():
    require(not any(p.exists() for p in (SUMMARY,SCOPE,PROOF)),'fresh derived outputs required')
    before={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (REPORT,STOP,LAUNCH)}
    require(sha(REPORT)=='9d8d796ffdcd5c2c4f42081902432fc4febe2168bf2e6c967c90878007a0e569','original report SHA mismatch')
    require(sha(STOP)=='13d9b771ce4da7e704bb423efb11550616d1322a0838be7fdb0471836fa29aa7','stop proof SHA mismatch')
    original=json.loads(REPORT.read_text(encoding='utf-8'));stop=json.loads(STOP.read_text(encoding='utf-8'));launch=json.loads(LAUNCH.read_text(encoding='utf-8'))
    ids=['case_%02d'%i for i in range(1,13)]
    require(original.get('ok') is False and original.get('planned_cases')==20 and original.get('executed_cases')==12 and original.get('unknown')==[] and original.get('operational_errors')==[],'original stopped scope mismatch')
    require([c['case_id'] for c in original['cases']]==ids,'complete case IDs mismatch')
    require(stop.get('ok') is True and stop['effective_analysis_case_ids']==ids and stop['launch_ok'] is False and launch.get('ok') is False and sha(LAUNCH)==stop['launch_proof_sha256'],'stop/launch provenance mismatch')
    partial=stop['partial_case13'];require(partial['observer_last_step']==164 and partial['excluded_from_success_and_failure_statistics'] is True and partial['data_preserved'] is True,'partial case13 exclusion mismatch')
    require([x['case_id'] for x in stop['unstarted_cases']]==['case_%02d'%i for i in range(14,21)] and all(x['directory_exists'] is False for x in stop['unstarted_cases']),'case14..20 not-started proof mismatch')
    freeze={k:stop.get(k) for k in ('source_inputs_plan_profile_unchanged','inputs_unchanged','user8_unchanged','no_further_case_or_runner_invocations')};require(all(v is True for v in freeze.values()),'stop freeze mismatch')
    cases=original['cases'];stages=[s for c in cases for s in c['stages']];homes=[s['home_after'] for s in stages if 'home_after' in s]
    counts={'physical_successes':sum(c.get('physical_success') is True for c in cases),'first_subtask_failures':sum(c.get('failure_category')=='first_subtask' for c in cases),'continuation_subtask_failures':sum(c.get('failure_category')=='continuation_subtask' for c in cases),'home_failures':sum(c.get('failure_category')=='home' for c in cases),'home_attempts':len(homes),'home_ready':sum(h.get('ready') is True and h.get('restore',{}).get('ok') is True for h in homes),'after_home_jobs':sum(max(0,len(c['stages'])-1) for c in cases),'after_home_job_successes':sum(s.get('job',{}).get('success') is True and s.get('score',{}).get('physical_gold_success') is True for c in cases for s in c['stages'][1:])}
    expected={'physical_successes':3,'first_subtask_failures':4,'continuation_subtask_failures':2,'home_failures':3,'home_attempts':8,'home_ready':5,'after_home_jobs':5,'after_home_job_successes':3};require(counts==expected,'derived counts mismatch: '+repr(counts))
    summary=copy.deepcopy(original);summary.update(ok=True,planned_cases=12,executed_cases=12,not_run=[],ok_meaning='completed twelve-case analysis scope only; original twenty-case campaign was user-stopped and did not complete',original_planned_cases=20,original_report_sha256=sha(REPORT),original_report_ok=False,original_not_run=copy.deepcopy(original['not_run']),scope12_path='scope12.json',**counts)
    require(summary['cases']==original['cases'],'case content changed')
    scope={'analysis_scope':'first_twelve_complete_cases_only','user_instruction':stop['user_instruction'],'user_instruction_source':'stop proof user_instruction verbatim','complete_case_ids':ids,'original_registered_case_count':20,'analysis_case_count':12,'original_twenty_case_campaign_completed':False,'original_report_ok':False,'original_launch_ok':False,'original_cached_not_run':copy.deepcopy(original['not_run']),'original_cached_not_run_is_final_execution_state':False,'case13':{'case_id':'case_13','status':partial['status'],'actual_actions':partial['observer_last_step'],'excluded_from_success_and_failure_statistics':True,'physical_success':None,'data_preserved':True},'unstarted_cases':copy.deepcopy(stop['unstarted_cases']),'original_report_sha256':sha(REPORT),'stop_proof_sha256':sha(STOP),'launch_proof_sha256':sha(LAUNCH),'stop_freeze_checks':freeze,'original_twenty_case_plan_unchanged':True,'new_physical_actions':0,'new_model_calls':0,'new_api_calls':0,'git_writes':0}
    require(all({'bytes':Path(p).stat().st_size,'sha256':sha(Path(p))}==x for p,x in before.items()),'raw sources changed before output')
    fresh(SUMMARY,summary);fresh(SCOPE,scope)
    after={p:{'bytes':Path(p).stat().st_size,'sha256':sha(Path(p))} for p in before};require(before==after,'raw sources changed');require(json.loads(SUMMARY.read_text(encoding='utf-8'))['cases']==original['cases'],'saved cases unequal')
    proof={'ok':True,'status':'HOME12_DERIVED_ANALYSIS_SCOPE_PASS','command':['D:\\miniconda\\python.exe','-B',str(T/'home12_20261009_derive.py')],'exit_code':0,'counts':counts,'planned_cases':12,'executed_cases':12,'original_report_ok':False,'original_launch_ok':False,'raw_sources_before':before,'raw_sources_after':after,'raw_sources_unchanged':True,'cases_equal':True,'summary_sha256':sha(SUMMARY),'scope_sha256':sha(SCOPE),'case13_actual_actions_excluded':164,'case14_20_unstarted':True,'new_environment_calls':0,'new_model_calls':0,'new_api_calls':0,'git_writes':0}
    fresh(PROOF,proof);print(json.dumps({'status':proof['status'],'counts':counts,'summary_sha256':sha(SUMMARY),'scope_sha256':sha(SCOPE),'proof_sha256':sha(PROOF),'raw_sources_unchanged':True,'cases_equal':True}));return 0
if __name__=='__main__':raise SystemExit(main())
