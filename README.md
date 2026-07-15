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

### `block-arrange`

- Local wrapper: `tools/run_block_arrange_over_ssh.ps1`
- Remote wrapper: `tools/run_block_arrange_on_dofbot.sh`
- `resync` rebuilds the stored block scene from `ai-visual`
- `move` automatically undoes prior moves when the source block is covered,
  stopping as soon as that block becomes the top of its stack
- `--no-auto-undo` restores fail-fast behavior for operators who do not want
  earlier moves changed
- Horizontal destinations must be at least 8cm from every other block; the
  default horizontal distance is also 8cm
- `undo` reverses the latest successful move
- `voice --text ...` parses one ASR transcript into an ordered `resync`/`move`
  sequence; it is plan-only unless `--execute` is supplied

Safe planning example:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_block_arrange_over_ssh.ps1 move --source blue --target red --relation right --distance-cm 8 --dry-run
```

Execute example:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_block_arrange_over_ssh.ps1 move --source blue --target red --relation above
```

Voice transcript planning example (does not move the arm):

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_block_arrange_over_ssh.ps1 voice --text "扫描当前场景，然后把红色方块放到蓝色方块上面"
```

For Chinese free-form speech, pass the ASR transcript to the same `voice`
command and add `--execute` only after the recognized text has been confirmed.
The built-in I2C speech-recognition module returns IDs for pre-registered
phrases, so it is suitable for wake words and short fixed commands, not full
multi-step transcripts. A microphone ASR service should produce text and call
this adapter.

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
