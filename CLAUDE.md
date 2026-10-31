# CueView — Agent Instructions

## Project layout
See the repository layout in `pool_shot_capture_agent_specs.md`.
All agents must create files in the exact locations specified there.

## Rules for all agents
- Python 3.11+ only. No f-strings with walrus operators (macOS 13 compat).
- Every module must have a corresponding test file in `tests/`.
- Run the VERIFY command at the bottom of your task before marking it done.
- Do not touch files outside your assigned TASK's `output_files` list.
- Import shared constants only from `src/constants.py`.

## Shared spec file
The full task specifications are in: pool_shot_capture_agent_specs.md
Read your assigned TASK block completely before writing any code.
