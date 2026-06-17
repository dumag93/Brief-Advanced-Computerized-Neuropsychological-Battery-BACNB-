# Monte Carlo Simulation

This folder contains the integrated BACNB Monte Carlo simulation.

The four task folders provide the behavioral task layer. This simulation provides the computational integration layer. It generates a synthetic population, simulates the same agents across SART, SST, Flanker, and Digit Span, extracts emergent constructs from observable task metrics, and maps functional profiles in a cognitive state space.

The script is included with the task files because the state-space interpretation depends on the joint behavior of the four tasks. Without the simulation, the repository would contain isolated computerized tasks but not the reproducible model that links them into BACNB.

Run the full reference configuration:

```bash
python simulation/Monte_Carlo_Simulation_BACNB.py --n 200000 --seed 20260610 --noise 1.0
```

Run a faster smoke test:

```bash
python simulation/Monte_Carlo_Simulation_BACNB.py --n 5000 --seed 20260610 --noise 1.0
```

Outputs are generated locally and should not be committed to the repository.
