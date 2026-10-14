"""Copy only the original runtime log lines belonging to this campaign."""
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
LOG = Path('/home/yhwang/fyp/xvla_native/hermes_home/logs/agent.log')


def main():
    raw_lines = LOG.read_bytes().splitlines(keepends=True)
    result = {}
    for name in ('table_tidy_stove', 'heat_yellow_mug', 'discard_two_cans'):
        agent = OUT / name / 'agent'
        output = agent / 'hermes_initial.log'
        if not output.exists():
            continue
        text = output.read_text(encoding='utf-8', errors='replace')
        match = re.search(r'session_id:\s*([A-Za-z0-9_]+)', text)
        if not match:
            result[name] = {'available': False, 'reason': 'CLI output contains no Hermes session id'}
            continue
        session = match.group(1)
        selected = b''.join(line for line in raw_lines if ('[' + session + ']').encode() in line)
        assert selected
        (agent / 'hermes_runtime_trace.log').write_bytes(selected)
        trace = selected.decode('utf-8', errors='replace')
        calls = re.findall(r'API call #(\d+): model=(\S+) provider=(\S+) in=(\d+) out=(\d+) total=(\d+)', trace)
        assert len({c[0] for c in calls}) == len(calls)
        value = {'available': True, 'session_id': session, 'api_calls': len(calls),
                 'models': sorted({c[1] for c in calls}), 'input_tokens': sum(int(c[3]) for c in calls),
                 'output_tokens': sum(int(c[4]) for c in calls), 'total_tokens': sum(int(c[5]) for c in calls),
                 'source': 'matching original API-call lines; CLI usage-file fallback remains unchanged'}
        (agent / 'hermes_logged_usage.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        result[name] = value
    (OUT / 'hermes_logged_usage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
