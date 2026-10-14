[CmdletBinding()]
param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
# UI process only. Models/scenes are loaded later by explicit page actions.
& wsl.exe -d Ubuntu --cd /mnt/d/FYP --exec /home/yhwang/fyp/xvla_native/sim_venv/bin/python /mnt/d/FYP/First_Phase/scene_demo/xvla_console.py --launch
if ($LASTEXITCODE -ne 0) { throw 'X-VLA console did not start. See results/interactive-console/web.log.' }
if (-not $NoBrowser) { Start-Process 'http://localhost:8083' }
