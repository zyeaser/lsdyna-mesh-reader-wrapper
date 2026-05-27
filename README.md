# lsdyna-mesh-reader-ext

A Python wrapper around [`lsdyna-mesh-reader`](https://github.com/pyvista/lsdyna-mesh-reader) that extends it with beam orientation parsing, structured NumPy export, and PyVista mesh visualization.

## What it does

Given an LS-DYNA `.k` keyword file, `mesh_data_reader_v0.py` extracts all mesh entities and writes them to an organized output directory:

| Entity | Exported files |
|--------|---------------|
| Nodes | `nodes.npy` — shape `(N, 4)` columns: `nid, x, y, z` |
| Shells | `shell_conn.npy`, `shell_pid.npy` |
| Solids | `solid_conn.npy`, `solid_pid.npy` |
| Beams | `beam_conn.npy`, `beam_pid.npy`, `beam_orient.npy` |

`*_conn` arrays are zero-padded so all elements in a section share the same row width (`eid, n1, n2, ...`).

Beam orientation vectors (`vx, vy, vz`) are parsed directly from `*ELEMENT_BEAM_ORIENTATION` cards — something the upstream library does not handle.

Optionally, PyVista renders PNG screenshots of the full mesh, shells, solids, and beams (with orientation arrows).

## Output layout

```
parsed_model/
├── binary/          # .npy arrays — fast to reload
│   ├── nodes.npy
│   ├── shell_conn.npy / shell_pid.npy
│   ├── solid_conn.npy / solid_pid.npy
│   ├── beam_conn.npy / beam_pid.npy / beam_orient.npy
├── debug/           # CSV previews (first 100 rows each)
├── metadata/
│   └── summary.json
└── plots/           # PNG screenshots (requires pyvista)
    ├── mesh_overview.png
    ├── shells.png
    ├── solids.png
    └── beams.png
```

## Usage

```bash
python mesh_data_reader_v0.py path/to/model.k
python mesh_data_reader_v0.py path/to/model.k --outdir my_output
python mesh_data_reader_v0.py path/to/model.k --no-plot
```

Or call from Python:

```python
from mesh_data_reader_v0 import export_lsdyna_mesh

export_lsdyna_mesh("model.k", outdir="parsed_model", plot=True)
```

## Viewing results

Open `view_parsed_model.ipynb` in Jupyter to interactively inspect nodes, shells, solids, beams, and rendered plots.

## Dependencies

```
lsdyna-mesh-reader
numpy
pandas
pyvista          # optional — only needed for plots
```

Install:

```bash
pip install lsdyna-mesh-reader numpy pandas pyvista
```

## Differences from upstream `lsdyna-mesh-reader`

- Custom `*ELEMENT_BEAM_ORIENTATION` parser — reads two-line beam cards including the orientation vector
- Unified NumPy export with zero-padded connectivity arrays
- Automatic PyVista screenshots with beam orientation arrows
- CLI entry point and `summary.json` metadata

---

## `convert_to_abaqus.py` — LS-DYNA → Abaqus INP converter

Reads the binary `.npy` arrays produced by `mesh_data_reader_v0.py` and writes a single Abaqus `.inp` file.

### Element mapping

| LS-DYNA type | Abaqus element |
|--------------|---------------|
| Beam         | `B31`          |
| Shell        | `S4R`          |
| Solid        | `C3D8R`        |

### What the output contains

- `*Node` block — all nodes from `nodes.npy`
- `*Element` blocks — one elset per PID (shells, solids) or per PID + orientation variant (beams)
- `*Beam Section` — per elset, with the parsed orientation vector as the local n1-axis
- `*Shell Section` — per elset, uniform thickness
- `*Solid Section` — per elset
- `*Material` — single elastic material
- Commented-out `*Step` placeholder for adding BCs and loads

### Configuration

Edit the `CONFIGURATION` block at the top of `convert_to_abaqus.py` — no other changes needed:

```python
BINARY_DIR   = 'binary'          # path to .npy files
OUTPUT_FILE  = 'model.inp'       # output path

MATERIAL_NAME = 'STEEL'
MATERIAL_E    = 200000.0         # Young's modulus (match model units)
MATERIAL_NU   = 0.3

BEAM_SECTION_TYPE        = 'PIPE'   # PIPE | CIRC | BOX | I | L
BEAM_PIPE_OUTER_RADIUS   = 0.05
BEAM_PIPE_WALL_THICKNESS = 0.005

SHELL_THICKNESS = 0.01
```

To add a beam section type other than `PIPE`, add an `elif` branch in `_beam_section_data_line()`.

### Usage

Run after `mesh_data_reader_v0.py` has populated the `binary/` directory:

```bash
python convert_to_abaqus.py
```

Output: `model.inp` in the same directory as the script.

### Beam orientation grouping

Beams sharing the same PID but different orientation vectors are written to separate elsets (`BEAM_PID{n}`, `BEAM_PID{n}_1`, …) so each elset can have its own `*Beam Section` with the correct local n1-axis direction.

> **Note:** Abaqus requires the n1-axis direction to be non-parallel to the beam axis. Verify the parsed orientation vectors are geometrically valid before submitting a job.
