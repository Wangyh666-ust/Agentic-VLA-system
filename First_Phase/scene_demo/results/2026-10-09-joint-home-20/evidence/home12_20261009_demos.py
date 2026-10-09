from pathlib import Path
import argparse, hashlib, json, shutil
import imageio.v2 as imageio

ROOT = Path('/mnt/d/FYP')
RESULT = ROOT / 'First_Phase/scene_demo/results/2026-10-09-joint-home-20'
REPORT = RESULT / 'evidence/summary12.json'
DEMOS = RESULT / 'demos12'
PROOF = ROOT / 'First_Phase/tmp/home12_20261009_demos_proof.json'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def fingerprint(path):
    path = Path(path)
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def write_fresh(path, data):
    with path.open('xb') as file:
        file.write((json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + '\n').encode('utf-8'))

def source_path(value):
    require(isinstance(value, str) and value, 'actual-motion video path missing')
    path = Path(value)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'source video must be existing absolute regular file')
    require(path.resolve().is_relative_to((RESULT / 'pilot').resolve()), 'source video outside fixed pilot')
    return path

def decode(path):
    reader = imageio.get_reader(str(path))
    count = 0
    shape = None
    try:
        fps = float(reader.get_meta_data()['fps'])
        require(abs(fps - 20) <= 1e-6, 'video fps not20: ' + str(path))
        for frame in reader:
            if shape is None:
                shape = list(frame.shape)
            require(list(frame.shape) == shape, 'video frame dimensions changed: ' + str(path))
            count += 1
    finally:
        reader.close()
    require(count > 0, 'empty video: ' + str(path))
    return {'frame_count': count, 'fps': fps, 'shape': shape}

def main(argv=None):
    parser = argparse.ArgumentParser(description='Concatenate only actual recorded home12 motion, without simulator imports.')
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args(argv)
    require(not DEMOS.exists() and not PROOF.exists(), 'fresh demos directory/proof required; no retry')
    frozen = {}
    proof = {'ok': False, 'demos_invocations': 1, 'physical_calls': 0, 'model_calls': 0, 'api_calls': 0, 'hermes_calls': 0, 'git_mutations': 0, 'case_videos': []}
    try:
        frozen[str(REPORT)] = fingerprint(REPORT)
        report = json.loads(REPORT.read_text(encoding='utf-8-sig'))
        require(report.get('ok') is True and report.get('executed_cases') == 12 and report.get('planned_cases') == 12 and report.get('not_run') == [], 'completed protocol-OK12 report required')
        cases = report.get('cases')
        require(isinstance(cases, list) and len(cases) == 12, 'exact12 cases required')
        require([c.get('case_id') for c in cases] == ['case_%02d' % i for i in range(1, 13)], 'case IDs not registered order')
        DEMOS.mkdir(parents=True)
        for case in cases:
            segments = []
            zero_motion = []
            vla_actions = home_actions = 0
            for stage_index, stage in enumerate(case['stages']):
                cap = stage['capability_id']
                job = stage['job']
                n = job['steps']
                require(type(n) is int and n >= 0, 'actual job action count invalid')
                vla_actions += n
                if n > 0:
                    segments.append({'kind': 'capability', 'capability_id': cap, 'stage_index': stage_index, 'path': source_path(job.get('rollout_path')), 'actions': n})
                else:
                    zero_motion.append({'kind': 'capability', 'capability_id': cap, 'stage_index': stage_index, 'actions': 0, 'reason': job.get('ended_reason'), 'video_added': False})
                if 'home_after' in stage:
                    home = stage['home_after']
                    require(isinstance(home, dict) and type(home.get('actions')) is int and home['actions'] >= 0, 'actual home action count invalid')
                    hn = home['actions']
                    home_actions += hn
                    if hn > 0:
                        segments.append({'kind': 'home', 'capability_id': cap, 'stage_index': stage_index, 'path': source_path(home.get('video')), 'actions': hn})
                    else:
                        zero_motion.append({'kind': 'home', 'capability_id': cap, 'stage_index': stage_index, 'actions': 0, 'ready': home.get('ready'), 'reason': home.get('reason'), 'video_added': False, 'zeroaction_or_blocked': True})
            require(case['vla_actions'] == vla_actions and case['home_actions'] == home_actions and case['total_physical_actions'] == vla_actions + home_actions, 'case reported actual action sums disagree')
            expected_actions = vla_actions + home_actions
            record = {'case_id': case['case_id'], 'physical_success': case.get('physical_success'), 'failure_category': case.get('failure_category'), 'failure_reason': case.get('failure_reason'), 'vla_actions': vla_actions, 'home_actions': home_actions, 'total_actions': expected_actions, 'segments': [], 'zero_motion': zero_motion, 'video': None, 'fps': 20}
            if not segments:
                require(expected_actions == 0, 'positive actions but no actual source video')
                record.update(reason='no_actual_motion', frame_count=0)
                proof['case_videos'].append(record)
                continue
            shape = None
            for segment in segments:
                path = segment['path']
                frozen.setdefault(str(path), fingerprint(path))
                require(fingerprint(path) == frozen[str(path)], 'source video changed before decoding')
                metadata = decode(path)
                require(metadata['frame_count'] == segment['actions'] + 1, 'source frames != actual actions+1: ' + str(path))
                if shape is None:
                    shape = metadata['shape']
                require(metadata['shape'] == shape, 'segment dimensions differ')
                segment.update(metadata)
            target = DEMOS / (case['case_id'] + '.mp4')
            if len(segments) == 1:
                require(not target.exists(), 'fresh destination required')
                shutil.copyfile(segments[0]['path'], target)
                require(fingerprint(target) == frozen[str(segments[0]['path'])], 'single-segment binarycopy mismatch')
                output_frames = segments[0]['frame_count']
            else:
                writer = imageio.get_writer(str(target), fps=20, macro_block_size=None)
                output_frames = 0
                try:
                    for part, segment in enumerate(segments):
                        reader = imageio.get_reader(str(segment['path']))
                        source_frames = 0
                        try:
                            for i, frame in enumerate(reader):
                                require(list(frame.shape) == shape, 'source dimensions changed during synthesis')
                                source_frames += 1
                                if part > 0 and i == 0:
                                    continue
                                writer.append_data(frame)
                                output_frames += 1
                        finally:
                            reader.close()
                        require(source_frames == segment['actions'] + 1, 'source frame count changed during synthesis')
                finally:
                    writer.close()
            require(output_frames == expected_actions + 1, 'combined output frames != total actual actions+1')
            decoded = decode(target)
            require(decoded['frame_count'] == output_frames and decoded['shape'] == shape, 'combined decode count/dimensions mismatch')
            cursor = 0
            for part, segment in enumerate(segments):
                appended = segment['actions'] + (1 if part == 0 else 0)
                record['segments'].append({**{k: v for k, v in segment.items() if k != 'path'}, 'source_path': str(segment['path']), 'source_sha256': frozen[str(segment['path'])]['sha256'], 'source_bytes': frozen[str(segment['path'])]['bytes'], 'output_frame_start': 0 if part == 0 else cursor - 1, 'output_frame_end': cursor + appended - 1, 'appended_output_frame_start': cursor, 'appended_frame_count': appended, 'dropped_initial_boundary_frame': part > 0})
                cursor += appended
            require(cursor == output_frames, 'segment frame mapping mismatch')
            record.update(video=str(target), frame_count=output_frames, shape=shape, sha256=fingerprint(target)['sha256'], bytes=target.stat().st_size, single_segment_raw_copy=len(segments) == 1)
            proof['case_videos'].append(record)
            print(json.dumps({'case_id': case['case_id'], 'frames': output_frames, 'actual_actions': expected_actions, 'segments': len(segments), 'physical_success': case.get('physical_success')}), flush=True)
        after = {p: fingerprint(p) for p in frozen}
        require(after == frozen, 'raw report/source videos changed')
        index = {'schema_version': 1, 'case_count': 12, 'case_videos': proof['case_videos'], 'report_sha256': frozen[str(REPORT)]['sha256'], 'raw_unchanged': True, 'fps': 20, 'includes_actual_motion_only': True, 'no_frame_interpolation_or_speed_change': True, 'boundary_rule': 'first segment includes initial frame; each later source frame0 is discarded as duplicated boundary', 'source_files_before': frozen, 'source_files_after': after}
        write_fresh(DEMOS / 'index.json', index)
        proof.update(ok=True, status='HOME12_ACTUAL_MOTION_DEMOS_PASS', cases=12, videos=sum(c['video'] is not None for c in proof['case_videos']), raw_unchanged=True, source_files_before=frozen, source_files_after=after, report_sha256=frozen[str(REPORT)]['sha256'], index_sha256=fingerprint(DEMOS / 'index.json')['sha256'])
    except Exception as exc:
        proof.update(status='HOME12_DEMOS_FAILURE_PRESERVED', error=repr(exc), source_files_before=frozen)
    write_fresh(PROOF, proof)
    print(json.dumps({k: proof.get(k) for k in ('status', 'ok', 'cases', 'videos', 'raw_unchanged', 'index_sha256', 'error')}), flush=True)
    return 0 if proof['ok'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
