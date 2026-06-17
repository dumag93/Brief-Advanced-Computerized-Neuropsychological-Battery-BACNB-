# BACNB

Brief Advanced Computerized Neuropsychological Battery (BACNB) is a reduced computerized battery for experimental assessment of executive-function dynamics.

This repository package contains only the public-facing computational core:

- four individual OpenSesame/OSWeb tasks;
- the integrated Monte Carlo simulation script.

It intentionally excludes manuscript files, bibliographic PDFs, submission archives, raw participant logs, generated Monte Carlo outputs, and historical audit material.

## Contents

```text
.
|-- simulation
|   |-- README.md
|   `-- Monte_Carlo_Simulation_BACNB.py
|-- tests
|   |-- sart
|   |   |-- sart.osexp
|   |   `-- index.html
|   |-- sst
|   |   |-- sst.osexp
|   |   `-- index.html
|   |-- flanker
|   |   |-- flanker.osexp
|   |   `-- index.html
|   `-- digit_span
|       |-- digit_span.osexp
|       `-- index.html
|-- CONTRIBUTING.md
|-- requirements.txt
|-- .gitignore
`-- LICENSE.txt
```

## Tasks

- SART: sustained attention, prepotent response, omissions, commissions, anticipations, variability, fatigue dynamics, and automation index.
- SST: reactive motor inhibition, p(respond|signal), integration-based SSRT, Go RT, Go omissions, choice errors, and SSD tracking.
- Flanker: conflict monitoring, incongruent interference, accuracy, reaction time, and temporal variability.
- Digit Span: verbal working memory, forward span, backward span, manipulation cost, accuracy, and response time.

Each task folder includes an OpenSesame experiment file (`.osexp`) and an OSWeb HTML export (`index.html`).

## Language Status

The current OpenSesame/OSWeb task interfaces are in Brazilian Portuguese (PTBR). They are shared as the original working BACNB task versions. The scientific and developer communities are explicitly invited to create polished interface revisions, translations, localization files, and language-specific adaptations, as long as attribution is preserved and the repository license terms are followed.

## Why the Monte Carlo Simulation Is Included

The individual tasks define the behavioral measurement layer of BACNB. The Monte Carlo simulation is included because it documents the theoretical integration layer: it simulates the same synthetic agents across SART, SST, Flanker, and Digit Span; transforms observable task metrics into oriented functional costs; extracts emergent constructs; and maps functional profiles in a cognitive state space.

In other words, the simulation is not an extra unrelated script. It is the reproducible computational bridge between the four independent tasks and the state-space model proposed for the battery.

Run:

```bash
python simulation/Monte_Carlo_Simulation_BACNB.py --n 200000 --seed 20260610 --noise 1.0
```

For a faster local test:

```bash
python simulation/Monte_Carlo_Simulation_BACNB.py --n 5000 --seed 20260610 --noise 1.0
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scientific Status

This is an experimental and in silico methodological project. It is not a clinical diagnostic instrument, does not provide clinical norms, and should not be used for individual diagnosis without empirical validation, pilot data, reliability testing, and normative calibration.

## Open Science Intent

This project is shared to support open science, user freedom, independent verification, and public benefit. The author does not seek royalties from community use. Contributions and code changes are welcome when they preserve attribution, respect the licenses, and keep the work aligned with transparent scientific use rather than exploitation of end users.

## Author

Luís Eduardo Magro de Queiroz  
Independent researcher  
Contact: dumag93@gmail.com

## License

This repository uses a dual-license model:

- Software source code and computational algorithms are licensed under GNU GPLv3.
- Scientific documentation, explanatory text, and original task-design materials are licensed under CC BY 4.0.
- Generated OpenSesame/OSWeb exports include third-party runtime components that remain under their respective licenses.

See `LICENSE.txt` for the exact scope.
