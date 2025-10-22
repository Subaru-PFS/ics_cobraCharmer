# ics-cobraCharmer

Cobra Charmer provides low-level control, simulation, and utilities for operating the Cobra fiber positioners used by
the Subaru Prime Focus Spectrograph (PFS). It includes a simple operator GUI, FPGA protocol helpers, simulators, logging
tools, and data processing utilities used within the PFS ICS (Instrument Control System).

Project home: https://github.com/Subaru-PFS/ics_cobraCharmer

## Features

- Operator GUI to exercise Cobra behaviors and run common operations.
- FPGA protocol helpers, logging and simulation tools.
- Utilities for coordinates, plotting, and analysis used in commissioning and engineering tasks.

## Requirements

- Python: 3.12+
- Network access during installation to fetch PFS libraries referenced directly from GitHub.

Core Python dependencies are declared in pyproject.toml and include PFS packages:

- ics-utils @ git+https://github.com/Subaru-PFS/ics_utils.git
- pfs-utils @ git+https://github.com/Subaru-PFS/pfs_utils.git
- pfs-instdata @ git+https://github.com/Subaru-PFS/pfs_instdata.git
- opdb @ git+https://github.com/Subaru-PFS/spt_operational_database.git

## Installation

You can install from a local clone or directly with pip. Because this repository relies on Git direct references for PFS
packages, ensure your environment can reach GitHub.

### From a local clone (recommended for development)

```
# Clone
git clone https://github.com/Subaru-PFS/ics_cobraCharmer.git
cd ics_cobraCharmer

# Create/activate a Python 3.12 environment (example using venv)
python3.12 -m venv .venv
. .venv/bin/activate

# Install in editable mode with dependencies
pip install -U pip setuptools wheel
pip install -e .
```

### Directly with pip (read-only use)

```
pip install git+https://github.com/Subaru-PFS/ics_cobraCharmer.git
```

## Repository layout

- bin/: entry-point scripts (GUI, simulators, logging helpers).
- python/ics/cobraCharmer/: package source code.
    - gui_manual.py: simple Tkinter-based GUI for manual operations.
    - fpgaSim.py: FPGA simulator entry point.
    - msimLog.py: log processing helper.
    - cobraCoach/: analysis and visualization utilities.
    - utils/: helper modules.
- docs/: Sphinx documentation.
- notebooks/: example and engineering notebooks used during development.
- ups/: EUPS packaging support used in PFS.

## Acknowledgments

This software is part of the Subaru PFS Instrument Control System. It builds on work from many contributors across the PFS collaboration.

## Troubleshooting

- Import errors for PFS packages: ensure network access to GitHub or pre-install the required PFS libraries listed
  above.
- GUI fails to connect to hardware: verify you are on the ICS network or using the provided simulators where applicable.
- OpenCV or Qt issues on macOS: ensure you use a clean virtual environment and Python 3.12, and consider installing
  system prerequisites via Homebrew.

If you encounter issues, please open an issue in the GitHub repository or contact the PFS ICS team.
