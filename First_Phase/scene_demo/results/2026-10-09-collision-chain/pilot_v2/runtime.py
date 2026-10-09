import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

CODE = Path('/mnt/d/FYP/First_Phase/scene_demo')
OLD_INSTALL = Path('/home/yhwang/fyp/scene_demo/hermes_home_side_pilot_20261009_v1/installs/385bcf885468a62d')
SELECTED_ENV = OLD_INSTALL/'environments/a65d5598f3a84dbb944a810e95a2f532/venv'

def attach_installed_dependencies(home: Path) -> dict:
    home = Path(home)
    assert SELECTED_ENV.is_dir(), 'Previously installed environment missing'
    source_facts = OLD_INSTALL/'facts.json'
    assert source_facts.is_file()
    destination = home/'installs/385bcf885468a62d'
    assert not destination.exists(), 'Refusing existing install-state directory'
    destination.mkdir(parents=True)
    copied_facts = destination/'facts.json'
    shutil.copyfile(source_facts,copied_facts)
    generations = destination/'environments'
    generations.symlink_to(OLD_INSTALL/'environments',target_is_directory=True)
    assert copied_facts.read_bytes()==source_facts.read_bytes()
    assert generations.resolve()==(OLD_INSTALL/'environments').resolve()
    selected = generations/'a65d5598f3a84dbb944a810e95a2f532/venv'
    assert selected.resolve()==SELECTED_ENV.resolve() and selected.is_dir()
    os.environ['HERMES_DISABLE_LAZY_INSTALLS']='1'
    return {'facts_sha256':hashlib.sha256(source_facts.read_bytes()).hexdigest(),
            'copied_facts_sha256':hashlib.sha256(copied_facts.read_bytes()).hexdigest(),
            'generations_resolved':str(generations.resolve()),
            'selected_environment_resolved':str(selected.resolve()),
            'lazy_installs_disabled':True,'symlink_path':str(generations)}

def main():
    sys.path.insert(0,str(CODE))
    import collision_grasp_pilot as collision
    pilot = collision.pilot
    original = pilot.prepare_hermes_home
    def prepare_with_installed_dependencies(home):
        config = original(home)
        reuse = attach_installed_dependencies(Path(home))
        output = Path(sys.argv[sys.argv.index('--output-dir')+1])
        assert output.is_dir(), 'Pilot output directory must already exist'
        (output/'runtime_reuse.json').write_text(json.dumps(reuse,indent=2)+'\n',encoding='utf-8')
        return config
    pilot.prepare_hermes_home = prepare_with_installed_dependencies
    os.environ['HERMES_DISABLE_LAZY_INSTALLS']='1'
    try:
        return collision.main()
    finally:
        pilot.prepare_hermes_home = original

if __name__ == '__main__':
    raise SystemExit(main())
