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
