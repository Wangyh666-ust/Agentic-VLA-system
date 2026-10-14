"""Task-local observation history, including unsuccessful intermediate checks."""
import copy
import hashlib
from pathlib import Path
from scene_demo.finish_v1.simple_checker import encoder_observation

MAX_PROCESS_CAPTURES = 4
MAX_CHECK_STATES = 10
MAX_INPUT_IMAGES = 3 * (MAX_PROCESS_CAPTURES + 1)


def public_checks(checks):
    """Keep visual judgements; never copy arbitrary model/context fields."""
    allowed = ('id', 'achieved', 'reason', 'target_matches_definition',
               'current_evidence', 'evidence_event_ids')
    return [{key: copy.deepcopy(row[key]) for key in allowed if key in row}
            for row in checks]


class ObservationTimeline:
    def __init__(self):
        self.releases = {}
        self.omitted_event_ids = []
        self.check_states = []

    def remember_release(self, event_id, context, image_paths):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('release event ID required')
        record = {'event_id': event_id,
                  'timestamp': context['actuator_measurements']['timestamp'],
                  'context': copy.deepcopy(context), 'image_paths': list(image_paths),
                  'image_sha256': [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in image_paths],
                  'classification_status': 'pending', 'raw_checks': [],
                  'effective_checks': []}
        if event_id in self.releases:
            previous = self.releases[event_id]
            if (previous['context'] != record['context']
                    or previous['image_paths'] != record['image_paths']
                    or previous['image_sha256'] != record['image_sha256']):
                raise ValueError('release event source changed')
            return
        if event_id in self.omitted_event_ids:
            return
        if len(self.releases) >= MAX_PROCESS_CAPTURES:
            self.omitted_event_ids.append(event_id)
            return
        self.releases[event_id] = record

    def classify_release(self, event_id, raw_checks, effective_checks, status='classified'):
        if event_id in self.releases:
            self.releases[event_id].update(classification_status=status,
                raw_checks=public_checks(raw_checks),
                effective_checks=public_checks(effective_checks))

    def remember_check(self, context, raw_checks, effective_checks):
        self.check_states.append({
            'timestamp': context['actuator_measurements']['timestamp'],
            'actuator_measurements': encoder_observation(context),
            'raw_checks': public_checks(raw_checks),
            'effective_checks': public_checks(effective_checks),
            'images_in_request': False,
        })
        self.check_states = self.check_states[-MAX_CHECK_STATES:]

    def request_input(self, context, image_paths):
        """Old triplets followed by the current triplet; each has its own feedback."""
        timestamp = context['actuator_measurements']['timestamp']
        combined = copy.deepcopy(context)
        paths, frames, history = [], [], []
        for record in sorted(self.releases.values(), key=lambda r: r['timestamp']):
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected
                   for p, expected in zip(record['image_paths'], record['image_sha256'])):
                raise ValueError('release image bytes changed')
            if record['timestamp'] > timestamp:
                raise ValueError('future release observation')
            row = {key: copy.deepcopy(record[key]) for key in (
                'event_id', 'timestamp', 'classification_status', 'raw_checks', 'effective_checks')}
            row['actuator_measurements'] = encoder_observation(record['context'])
            if record['timestamp'] == timestamp:
                if (record['context']['actuator_measurements']['finger_qpos']
                        != context['actuator_measurements']['finger_qpos']):
                    raise ValueError('different feedback at the same capture time')
            # A release at a polling boundary can have the same physical time
            # but different capture filenames. Preserve both sources and label
            # their equal timestamps rather than inventing a temporal gap.
            row['frame_indices'] = list(range(len(paths), len(paths) + 3))
            for source, path in zip(record['context']['sensor_frames'], record['image_paths']):
                frames.append({**source, 'frame': len(paths)})
                paths.append(path)
            history.append(row)
        current_indices = list(range(len(paths), len(paths) + 3))
        for source, path in zip(context['sensor_frames'], image_paths):
            frames.append({**source, 'frame': len(paths)})
            paths.append(str(path))
        combined['sensor_frames'] = frames
        combined['visual_timeline'] = {
            'current_frame_indices': current_indices,
            'process_history': history,
            'previous_check_states': copy.deepcopy(self.check_states),
            'history_complete': not self.omitted_event_ids,
            'omitted_release_event_ids': list(self.omitted_event_ids),
        }
        if any(row['timestamp'] > timestamp for row in self.check_states):
            raise ValueError('future intermediate check')
        return combined, paths
