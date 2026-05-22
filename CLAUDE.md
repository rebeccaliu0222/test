# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`trs_automation` is a Python project in early development. The repository currently contains a single stub script (`test.py`). Architecture and conventions will grow from here.

## Running

```powershell
python test.py
```

## Conventions

- **Formatter:** Black (configured in PyCharm). Run with `black .` before committing.
- **Python version:** Managed via the `trs_automation` virtual environment in PyCharm.
- No test framework is set up yet. When one is added, prefer `pytest`.
