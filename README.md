# Functions Remote Launch

This directory groups feature-oriented robot functions such as `ai-visual`,
`snake`, and `staticHeap`.

## Goal

Keep each feature launchable from the local Windows machine through SSH, while
making the default path safe:

- default mode should avoid arm motion when possible
- motion should require an explicit execute/follow mode
- the local entrypoint should live under `tools/`
- the remote wrapper should resolve the real feature directory on Dofbot

## Recommended Layout

For a function named `<name>`:

- function source: `functions/<name>/`
- local SSH wrapper: `tools/run_<name>_over_ssh.ps1`
- remote launcher: `tools/run_<name>_on_dofbot.sh`
- optional remote Python entrypoint: `tools/run_<name>_on_dofbot.py`

## Launch Pattern

1. Run the local PowerShell wrapper on Windows.
2. The wrapper SSHes to Dofbot and enters the deployed project directory.
3. The remote shell wrapper resolves the real function path and sources runtime
   dependencies only when required.
4. The remote Python/script entrypoint runs the feature.

## Standard Contract

When adding a new remote-launchable function under `functions/<name>/`, keep
the following contract stable:

- local entrypoint: `tools/run_<name>_over_ssh.ps1`
- remote entrypoint: `tools/run_<name>_on_dofbot.sh`
- optional Python runner: `tools/run_<name>_on_dofbot.py`
- remote shell should resolve both `functions/<name>` and any known legacy path
- motion must require an explicit flag and should refuse to run when required
  services or modules are missing

Recommended wrapper responsibilities:

- PowerShell wrapper: assemble one SSH command only
- remote shell wrapper: resolve paths, check or start prerequisites, then
  `exec` the real runner
- Python runner: implement feature-specific safe defaults and logs

## Current Functions

### `ai-visual`

- Local wrapper: `tools/run_ai_visual_over_ssh.ps1`
- Remote wrapper: `tools/run_ai_visual_on_dofbot.sh`
- Default behavior: detection only
- Motion behavior: `--scan-map` moves the scan pose only; `--execute` performs
  pick-and-place

Example:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_ai_visual_over_ssh.ps1 --scan-map
```

### `snake`

- Local wrapper: `tools/run_snake_over_ssh.ps1`
- Remote wrapper: `tools/run_snake_on_dofbot.sh`
- Remote Python entry: `tools/run_snake_on_dofbot.py`
- Default behavior: camera + color detection only
- Follow prerequisite: `functions/snake/start_kinematics.sh`
- Color selection: `--color red|green|blue|yellow` maps to the notebook
  `choose_color` target selection
- Follow behavior: `--follow` first checks the kinematics service and starts it
  if needed, then enters the legacy snake control loop
- Detection-only remains the safe default when `--follow` is not supplied

Example:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_snake_over_ssh.ps1 --color red --frames 60
```

Follow example:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_snake_over_ssh.ps1 --follow --color yellow --frames 120
```

## Directory Resolution

Remote wrappers should probe multiple locations because the code on Dofbot may
not match the current local path exactly.

Typical candidates:

- `/home/dofbot/workspace/EngineerPractice/functions/<name>`
- `/home/dofbot/workspace/<name>`
- feature-specific legacy paths such as `/home/dofbot/hxx/snake`

## Safety Rules

- Default to non-motion startup.
- Source ROS only when a mode actually needs ROS.
- Auto-start prerequisite services only for an explicit motion mode.
- Refuse motion if the required service or module path is unavailable.
- Save one remote snapshot when practical so the run can be inspected after the
  fact.
