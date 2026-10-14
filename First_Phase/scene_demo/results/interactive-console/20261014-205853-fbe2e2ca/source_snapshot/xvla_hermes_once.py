"""Run the installed Hermes SDK for one tool-free multimodal planning turn.

Invoked with its installed Python/runtime, not the simulator's environment.
All actual model requests and responses are saved without credentials.
"""
import json
import sys
import time
from pathlib import Path


def main():
    source, destination, sdk = map(Path, sys.argv[1:4])
    sys.path.insert(0, str(sdk))
    from run_agent import AIAgent
    from dotenv import dotenv_values
    import os
    config = json.loads(source.read_text())
    values = dotenv_values(Path(os.environ['HERMES_HOME']) / '.env')
    key = next((values.get(k) or os.environ.get(k) for k in
                ('OPENAI_API_KEY', 'DASHSCOPE_API_KEY') if values.get(k) or os.environ.get(k)), None)
    if not key:
        raise RuntimeError('Hermes Qwen credentials unavailable')
    agent = AIAgent(model='qwen3.8-max', provider='custom',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key=key,
        api_mode='chat_completions', max_iterations=1, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        skip_background_review=True, load_soul_identity=False,
        max_tokens=5000, reasoning_config={'enabled':False},
        request_overrides={'response_format':{'type':'json_object'},
            'extra_body':{'enable_thinking':False}, 'stream':False, 'temperature':0})
    if agent.tools:
        raise RuntimeError('Planning must have no tools or executable actions')
    # Hermes' transport builds a per-request client. Instrument the installed
    # SDK resource class so the audit follows that real client as well.
    from openai.resources.chat.completions import Completions
    create = Completions.create
    calls = []
    def audited_create(resource, **kwargs):
        if calls:
            raise RuntimeError('Hermes single-request budget exhausted')
        kwargs['timeout'] = 90
        if kwargs.get('model') != 'qwen3.8-max' or kwargs.get('tools'):
            raise RuntimeError('Unexpected Hermes model/tools')
        calls.append({'started_at':time.time()})
        (destination.parent / 'hermes_request.json').write_text(
            json.dumps(kwargs,ensure_ascii=False,indent=2),encoding='utf-8')
        start = time.monotonic()
        resource._client.max_retries = 0
        response = create(resource, **kwargs)
        def save(raw):
            calls[-1]['elapsed_seconds'] = time.monotonic()-start
            (destination.parent / 'hermes_response.json').write_text(
                json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
            if raw.get('model') != 'qwen3.8-max':
                raise RuntimeError('Unexpected returned Hermes model')
            calls[-1]['usage'] = raw.get('usage')
            calls[-1]['returned_model'] = raw['model']
        if hasattr(response,'model_dump'):
            save(response.model_dump(mode='json'))
            return response
        class AuditedStream:
            def __getattr__(self,name):return getattr(response,name)
            def __enter__(self):return self
            def __exit__(self,*args):response.close()
            def __iter__(self):
                parts=[];model=None;usage=None;finish=None
                with (destination.parent/'hermes_chunks.jsonl').open('w',encoding='utf-8') as log:
                    for chunk in response:
                        raw=chunk.model_dump(mode='json')
                        log.write(json.dumps(raw,ensure_ascii=False)+'\n');log.flush()
                        model=raw.get('model') or model;usage=raw.get('usage') or usage
                        for choice in raw.get('choices',[]):
                            parts.append(choice.get('delta',{}).get('content') or '')
                            finish=choice.get('finish_reason') or finish
                        yield chunk
                save({'model':model,'usage':usage,'choices':[{'message':{'role':'assistant',
                    'content':''.join(parts)},'finish_reason':finish}],'assembled_from':'hermes_chunks.jsonl'})
        return AuditedStream()
    Completions.create = audited_create
    # Avoid SDK HTTP retries in addition to its single conversation iteration.
    agent.client.max_retries = 0
    started = time.monotonic()
    result = agent.run_conversation(user_message=config['content'],
        system_message='你是Hermes机器人任务规划器。仅根据本次三视图、需求及原生技能目录输出一个JSON对象，不调用工具、不执行动作。')
    output = {'content':result.get('final_response'), 'calls':calls,
              'sdk':str(sdk), 'elapsed_seconds':time.monotonic()-started,
              'failed':result.get('failed',False), 'error':result.get('error')}
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    if output['failed'] or len(calls)!=1 or not output['content']:
        raise RuntimeError('Hermes planning did not produce a single response')


if __name__ == '__main__':
    main()
