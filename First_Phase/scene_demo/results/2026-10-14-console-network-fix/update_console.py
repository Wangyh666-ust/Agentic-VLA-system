from pathlib import Path

base = Path(__file__).resolve().parents[2]
path = base / 'xvla_console.py'
source = path.read_text(encoding='utf-8')
start = source.index('    def preflight(self, scene, request=None):')
end = source.index('    def ensure_runtime(self):', start)
replacement = '''    def preflight(self, scene, request=None):
        """Prepare this interactive request; historical auditing is separate."""
        self.check_cancel()
        request = task_text(request)
        from scene_demo import catalog
        if scene not in catalog.SCENES:
            raise ValueError('未知场景')
        capabilities = catalog.scene_capabilities(scene, backend='xvla')
        if not capabilities:
            raise ValueError('当前场景没有可用的 X-VLA 动作')
        if os.environ.get('SCENE_PLANNER_MODEL', 'qwen3.8-max') != 'qwen3.8-max':
            raise ValueError('当前模型配置必须为 Qwen 3.8')
        self.update(stage='preflight', message='正在准备当前任务。')
        record = {'schema_version': 1, 'scene': scene, 'request': request,
                  'request_id': self.state['request_id'], 'seed': 142, 'init_state_index': 0,
                  'candidate_capability_ids': [c['capability_id'] for c in capabilities],
                  'budget_per_subgoal': 800 if scene == 'goal_table' else 900,
                  'session_step_limit': 8000, 'helpers': False,
                  'execution_requires_confirmation': True,
                  'history_audit': 'not_run_in_interactive_path',
                  'source_sha256': {}}
        snapshot = self.directory / 'source_snapshot'
        for item in sorted(HERE.glob('*.py')) + sorted((HERE / 'finish_v1').glob('*.py')):
            self.check_cancel()
            relative = item.relative_to(HERE)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = item.read_bytes()
            target.write_bytes(data)
            record['source_sha256'][str(relative)] = hashlib.sha256(data).hexdigest()
        write(self.directory / 'run_setup.json', record)

'''
path.write_text(source[:start] + replacement + source[end:], encoding='utf-8')
html = base / 'xvla_console.html'
text = html.read_text(encoding='utf-8')
text = text.replace("preflight:'核对历史'", "preflight:'准备任务'")
html.write_text(text, encoding='utf-8')
print('updated interactive setup; removed full registry subprocess')
