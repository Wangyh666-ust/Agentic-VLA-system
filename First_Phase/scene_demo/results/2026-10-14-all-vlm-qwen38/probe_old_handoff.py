"""One authorized zero-action model comparison using the exact failed request."""
import json
from pathlib import Path
from scene_demo.finish_v1.simple_checker import SimpleQwenClient
from scene_demo.finish_v1 import vision

out=Path(__file__).resolve().parent/'old_handoff_qwen38'
source=Path(__file__).resolve().parents[1]/'interactive-console/runs/20261014T030501Z_book_caddy_274c55f3/cap_01_xvla_book_caddy_b6725ed9/finish/model_requests/call_01_request.json'
record=json.loads(source.read_text())
class Probe(SimpleQwenClient):
    def parse_verdict(self, content):
        self.raw_review=vision._parse_verdict_content(content)
        return super().parse_verdict('{"goal_achieved":null,"reason":"raw relation comparison only"}')
client=Probe(max_calls=1)
client.audit_dir=out
client.request_options=record['request_options']
result=client._request(record['context'],[r['path'] for r in record['images']],record['prompt'])
summary={'source_request':str(source),'actions':0,'calls':client._calls,'model':result['model'],
         'raw_review':client.raw_review,'usage':result['usage']}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False))
