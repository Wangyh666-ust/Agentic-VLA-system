"""Summarize original records and concatenate native videos; zero model/action calls."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
import cv2
import imageio_ffmpeg
from scene_demo.finish_v1.vision import _parse_verdict_content

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def summarize(case):
    result = read(HERE / case / 'result.json')
    jobs = []
    checks = []
    videos = []
    for task in result['subtasks']:
        directory = Path(task['job']['run_dir'])
        finish = task['finish'] or {}
        actions = rows(directory / 'finish_actions.jsonl')
        independent = rows(directory / 'independent_finish.jsonl')
        job_checks = []
        policy_actions = rows(directory / 'action_sources.jsonl')
        policy_events = rows(directory / 'events.jsonl')
        assert len(policy_events) == task['actual_vla']
        assert [row['step'] for row in policy_events] == list(range(1, task['actual_vla'] + 1))
        for check in rows(directory / 'finish/checks.jsonl'):
            index = check['call_index']
            requests = directory / 'finish/model_requests'
            request = read(requests / f'call_{index:02d}_request.json')
            prompt_copy = requests / f'call_{index:02d}_request.txt'
            if not prompt_copy.exists():
                prompt_copy.write_text(request['prompt'] + '\n' +
                                       json.dumps({'images': request['images']}, ensure_ascii=False, indent=2) + '\n')
            response = read(requests / f'call_{index:02d}_response.json')
            final_answer_text = response['choices'][0]['message']['content']
            verdict = _parse_verdict_content(final_answer_text)
            payload = json.loads(request['prompt'].splitlines()[-1])
            measured = payload['actuator_measurements']
            images = [{'path': image['path'], 'hash_ok': hashlib.sha256(Path(image['path']).read_bytes()).hexdigest() == image['sha256']}
                      for image in request['images']]
            audit = {'capability': task['capability'], 'call_index': index,
                     'model': response.get('model'), 'final_answer': verdict,
                     'final_answer_text': final_answer_text,
                     'actual_vla_at_check': next((row['actual_vla'] for row in reversed(policy_actions)
                                                  if abs(row['timestamp'] - measured['timestamp']) <= 1e-9), None),
                     'last_action_source': next((row['source'] for row in reversed(policy_actions)
                                                 if abs(row['timestamp'] - measured['timestamp']) <= 1e-9), None),
                     'usage': response.get('usage'), 'images': images,
                     'same_timestamp': all(abs(frame['timestamp'] - measured['timestamp']) <= 1e-9 for frame in payload['sensor_frames']),
                     'encoder_matches_robot': measured['finger_qpos'] == check['robot']['finger_qpos'],
                     'at_standard_open': measured['at_standard_open'],
                     'scheduled_vla_at_check': min(150 * index, task['actual_vla']),
                     'no_phase_stage': 'phase' not in payload['current_subtask'] and 'stage' not in measured,
                     'visual_state_definition': payload.get('visual_state_definition'),
                     'request_path': str(prompt_copy)}
            assert all(image['hash_ok'] for image in images), audit
            assert audit['same_timestamp'] and audit['encoder_matches_robot'] and audit['no_phase_stage'], audit
            if task['capability'] == 'stove_on':
                assert '红色加热圈' in payload['visual_state_definition']['on']
            checks.append(audit)
            job_checks.append(audit)
        counts = dict(Counter(action['phase'] for action in actions))
        for phase, count in (finish.get('counts') or {}).items():
            assert counts.get(phase, 0) == count, (task['capability'], phase, count, counts)
        reference = job_checks[0] if job_checks else None
        feedback = finish.get('feedback_retreat') or {}
        target_open = (read(directory / 'finish/model_requests/call_01_request.json')['context']['actuator_measurements']['standard_open_finger_qpos']
                       if job_checks else [])
        jobs.append({'capability': task['capability'], 'success': task['outcome']['ok'],
                     'reason': task['outcome']['reason'], 'actual_vla': task['actual_vla'],
                     'grasp_actions': task['local_grasp_actions'], 'actual_finish_counts': counts,
                     'finish_triggered': finish.get('handoff_goal_latched') is True,
                     'native_goal_true_samples': sum(all(row['declared_predicates'].values()) for row in policy_events),
                     'last_native_goal': policy_events[-1]['declared_predicates'] if policy_events else None,
                     'release_open_actions': finish.get('release_open_actions', 0),
                     'concurrent_open_retreat': finish.get('concurrent_open_retreat'),
                     'movement_with_open_target': sum(action['phase'] == 'movement' and action.get('finger_target') == target_open for action in actions),
                     'home_streak': finish.get('home_streak', 0),
                     'independent_streak': task['independent_streak'],
                     'final_independent': independent[-1] if independent else None,
                     'finish_reason': finish.get('reason'),
                     'feedback_retreat': {key: feedback.get(key) for key in ('ok', 'reason', 'mode', 'travel_m', 'endpoint_clearance')},
                     'initial_encoder_open': reference['at_standard_open'] if reference else None,
                     'checks': [{'call_index': c['call_index'], 'final_answer': c['final_answer'], 'encoder_open': c['at_standard_open']} for c in job_checks],
                     'first_finish_command': actions[0] if actions else None,
                     'run_dir': str(directory)})
        videos.append(Path(task['job']['rollout_path']))
    destination = HERE / case / 'sequence.mp4'
    frames = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        assert capture.isOpened(), video
        assert abs(capture.get(cv2.CAP_PROP_FPS) - 20) < .01
        frames += int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    concat_list = HERE / case / 'video_inputs.txt'
    assert all("'" not in str(video) for video in videos)
    concat_list.write_text(''.join(f"file '{video}'\n" for video in videos))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n',
               '-f', 'concat', '-safe', '0', '-i', str(concat_list), '-an',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
    if not destination.exists():
        encoded = subprocess.run(command, capture_output=True)
        (HERE / case / 'video_encoding.log').write_bytes(encoded.stdout + encoded.stderr)
        assert encoded.returncode == 0
    capture = cv2.VideoCapture(str(destination))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
    capture.release()
    return {'case': case, 'state': result['plan']['state'], 'plan_success': result['plan'].get('plan_success'),
            'jobs': jobs, 'checks': checks, 'video_frames': frames,
            'new_vla': result['new_vla'], 'grasp_actions': result['local_grasp_actions']}


summary = {'id': '20261011-container-pilot', 'cases': [summarize(case) for case in ('sauce_basket', 'mug_microwave')],
           'subagents': 0, 'training': False, 'replay_actions': 0, 'standalone_home_trials': 0}
responses = [read(path) for path in HERE.glob('*/runs/**/*_response.json')]
summary['qwen_calls'] = len(responses)
summary['qwen_total_tokens'] = sum(response.get('usage', {}).get('total_tokens', 0) for response in responses)
write(HERE / 'summary.json', summary)
for case in summary['cases']:
    print(json.dumps({'case': case['case'], 'state': case['state'],
                      'jobs': [{key: job[key] for key in ('capability', 'success', 'reason', 'actual_vla', 'actual_finish_counts', 'home_streak', 'independent_streak')} for job in case['jobs']]}, ensure_ascii=False))
print(json.dumps({key: summary[key] for key in ('qwen_calls', 'qwen_total_tokens')}))
