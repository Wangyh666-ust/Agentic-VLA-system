from pathlib import Path
p=Path('First_Phase/scene_demo/xvla_console.py');s=p.read_text(encoding='utf-8')
a=s.index('    def preview_scene(self):');b=s.index('    def load(self, scene):',a)
s=s[:a]+'''    def preview_scene(self):
        """Create and close only a render environment in its own process."""
        directory = self.directory / 'scene_preview'
        command = [sys.executable, str(HERE / 'scene_preview.py'), '--scene', self.state['scene'],
                   '--output', str(directory)]
        deadline = time.monotonic() + 180
        with (self.directory / 'preview.log').open('wb') as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as proc:
                while proc.poll() is None:
                    if self.cancel.wait(.2) or time.monotonic() > deadline:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        self.check_cancel()
                        raise RuntimeError('场景拍摄超时，请查看 preview.log。')
                result = read(directory / 'result.json')
                if proc.returncode or not result or result.get('ok') is not True:
                    raise RuntimeError('场景拍摄失败，请查看本次 preview.log。')
        return result['frames']

'''+s[b:]
a=s.index('        self.preflight(scene)\n',s.index('    def load('));b=s.index("        self.update(stage='ready'",a)
s=s[:a]+'''        self.check_cancel()
        self.update(stage='loading', message='正在载入场景并拍摄三视图；不启动模型、不执行任务。')
        preview = self.preview_scene()
        self.check_cancel()
        from scene_demo import catalog
        spec = catalog.SCENES[scene]
        self.session = {'ok': True, 'session_id': 'preview-' + secrets.token_hex(12), 'scene_id': scene,
                        'state': 'ready', 'scene_version': 0, 'capability_backend': 'xvla',
                        'capabilities': catalog.scene_capabilities(scene, backend='xvla'),
                        'storage_policy': spec.get('storage_policy', {}), 'preview_only': True}
        write(self.directory / 'preview_session.json', self.session)
'''+s[b:]
s=s.replace("'stage': 'preflight', 'request': '', 'scene': scene", "'stage': 'loading', 'request': '', 'scene': scene")
pos=s.index('        rid, sid =',s.index('    def plan(self, request):'))
s=s[:pos]+'''        if self.session.get('preview_only'):
            # Task setup starts only after the user submits a task. Recreate the
            # selected initial scene using the unchanged production initialization,
            # including settling, then take fresh planning images before grounding.
            self.update(request=request)
            self.preflight(self.state['scene'], request)
            self.check_cancel()
            self.ensure_runtime()
            self.check_cancel()
            result = self.svc.create_session(self.state['scene'], seed=142, init_state_index=0)
            if result.get('ok') is not True:
                raise RuntimeError(json.dumps(result, ensure_ascii=False))
            self.session = result.get('session', result)
            self.session = self.svc.session(self.session['session_id'])
            self.update(session_id=self.session['session_id'])
            write(self.directory / 'initial_session.json', self.session)
            self.check_cancel()
'''+s[pos:]
# A ready preview can be replaced directly without stopping a non-existent task.
s=s.replace("self.state.get('stage') in ('ready', 'draft')", "self.state.get('stage') == 'draft'")
s=s.replace("console.state.get('stage') in ('ready', 'draft')", "console.state.get('stage') == 'draft'")
p.write_text(s,encoding='utf-8')
p=Path('First_Phase/scene_demo/xvla_console.html');s=p.read_text(encoding='utf-8');s=s.replace("live&&['ready','draft'].includes(data.stage)","live&&data.stage==='draft'")
p.write_text(s,encoding='utf-8')
# Correct only the malformed derived registry metadata. Raw evidence is untouched.
p=Path('First_Phase/scene_demo/experiment_registry.json');import json;d=json.loads(p.read_text())
fixed=[]
for e in d['entries']:
    if e['id'] in ['20261014-xvla-console-software','20261014-xvla-console-scene-first'] and isinstance(e['conditions'],str):
        e['conditions']=[e['conditions']];fixed.append(e['id'])
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print({'registry_metadata_fixed':fixed})
