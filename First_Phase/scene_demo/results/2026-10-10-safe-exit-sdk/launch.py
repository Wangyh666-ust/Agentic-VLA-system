from pathlib import Path
from scene_demo import safe_exit_pilot as pilot
from scene_demo import safe_exit_pilot_io as io

io.OUTPUT = Path(__file__).resolve().parent / 'pilot'
io.INPUTS_SHA = '6ed04b29b202098e286d57270b492345b561348240e366d9f3a5f5e9886f1b6c'
if __name__ == '__main__':
    raise SystemExit(pilot.main())
