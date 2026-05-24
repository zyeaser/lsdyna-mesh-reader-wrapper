import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import lsdyna_mesh_reader


def read_beam_orientation_connectivity(filename):
    beams = []
    in_beam = False
    expect_vector = False
    current = None

    with open(filename, "r") as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith("$"):
                continue

            if s.startswith("*"):
                in_beam = s.upper().startswith("*ELEMENT_BEAM_ORIENTATION")
                expect_vector = False
                continue

            if in_beam:
                vals = s.split()

                if not expect_vector:
                    eid, pid, n1, n2 = map(int, vals[:4])
                    current = {"eid": eid, "pid": pid, "n1": n1, "n2": n2}
                    expect_vector = True
                else:
                    vx, vy, vz = map(float, vals[:3])
                    current["vx"] = vx
                    current["vy"] = vy
                    current["vz"] = vz
                    beams.append(current)
                    expect_vector = False

    return beams


def _build_grid(nodes, shell_conn, solid_conn):
    """Build a PyVista UnstructuredGrid from parsed node/element arrays."""
    import pyvista as pv

    points = nodes[:, 1:4].astype(float)
    nid_to_idx = {int(nid): i for i, nid in enumerate(nodes[:, 0])}

    # VTK cell type constants
    VTK_TRIANGLE   = 5
    VTK_POLYGON    = 7
    VTK_QUAD       = 9
    VTK_TETRA      = 10
    VTK_HEXAHEDRON = 12
    VTK_WEDGE      = 13
    VTK_PYRAMID    = 14

    cells_flat = []
    cell_types = []

    def _shell_type(n):
        return {3: VTK_TRIANGLE, 4: VTK_QUAD}.get(n, VTK_POLYGON)

    def _solid_type(n):
        return {4: VTK_TETRA, 5: VTK_PYRAMID, 6: VTK_WEDGE, 8: VTK_HEXAHEDRON}.get(n, VTK_POLYGON)

    for conn, type_fn in [(shell_conn, _shell_type), (solid_conn, _solid_type)]:
        if conn is None:
            continue
        for row in conn:
            nids = [int(n) for n in row[1:] if n != 0]
            idxs = [nid_to_idx[n] for n in nids if n in nid_to_idx]
            if not idxs:
                continue
            cells_flat.extend([len(idxs)] + idxs)
            cell_types.append(type_fn(len(idxs)))

    if not cell_types:
        return None

    return pv.UnstructuredGrid(
        np.array(cells_flat, dtype=np.int64),
        np.array(cell_types, dtype=np.uint8),
        points,
    )


def _plot_mesh(nodes, shell_conn, solid_conn, beam_conn, beam_orient, plot_dir):
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed — skipping plots.")
        return

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    grid = _build_grid(nodes, shell_conn, solid_conn)
    if grid is None:
        print("  Warning: no shell/solid elements found — skipping mesh plots.")

    nid_to_idx = {int(row[0]): i for i, row in enumerate(nodes)}
    xyz = nodes[:, 1:4].astype(float)

    def get_xyz(nid):
        i = nid_to_idx.get(int(nid))
        return xyz[i] if i is not None else np.zeros(3)

    # Build beam PolyData
    beam_pd = None
    if beam_conn is not None:
        try:
            pts, lines = [], []
            for row in beam_conn:
                idx = len(pts)
                pts.extend([get_xyz(row[1]), get_xyz(row[2])])
                lines.extend([2, idx, idx + 1])
            beam_pd = pv.PolyData()
            beam_pd.points = np.array(pts, dtype=float)
            beam_pd.lines = np.array(lines)
        except Exception as e:
            print(f"  Warning: beam mesh creation failed ({e})")

    def _make_plotter():
        return pv.Plotter(off_screen=True, window_size=[1600, 1200])

    def _save(pl, path):
        pl.add_axes()
        pl.camera_position = "iso"
        pl.screenshot(str(path))
        pl.close()
        print(f"  Saved: {path}")

    # Overview
    if grid is not None or beam_pd is not None:
        pl = _make_plotter()
        if grid is not None:
            pl.add_mesh(grid, color="white", show_edges=True, smooth_shading=True,
                        edge_color="steelblue", line_width=0.5, opacity=0.9)
        if beam_pd is not None:
            pl.add_mesh(beam_pd, color="crimson", line_width=2.0)
        _save(pl, plot_dir / "mesh_overview.png")

    # Shells only
    if grid is not None:
        shell_idx = np.where(np.isin(grid.celltypes, [5, 7, 9]))[0]
        if len(shell_idx) > 0:
            pl = _make_plotter()
            pl.add_mesh(grid.extract_cells(shell_idx), color="white", show_edges=True,
                        smooth_shading=True, edge_color="steelblue", line_width=0.5)
            _save(pl, plot_dir / "shells.png")

        # Solids only
        solid_idx = np.where(np.isin(grid.celltypes, [10, 12, 13, 14]))[0]
        if len(solid_idx) > 0:
            pl = _make_plotter()
            pl.add_mesh(grid.extract_cells(solid_idx), color="white", show_edges=True,
                        smooth_shading=True, edge_color="darkorange", line_width=0.5)
            _save(pl, plot_dir / "solids.png")

    # Beams
    if beam_pd is not None:
        try:
            pl = _make_plotter()
            pl.add_mesh(beam_pd, color="crimson", line_width=2.0)

            if beam_orient is not None and grid is not None:
                try:
                    b = grid.bounds
                    scale = max(b[1]-b[0], b[3]-b[2], b[5]-b[4]) * 0.04
                    centers = np.array([
                        (get_xyz(row[1]) + get_xyz(row[2])) / 2 for row in beam_conn
                    ])
                    pl.add_arrows(centers, beam_orient.astype(float), mag=scale, color="navy")
                except Exception as e:
                    print(f"  Warning: beam orientation arrows skipped ({e})")

            _save(pl, plot_dir / "beams.png")
        except Exception as e:
            print(f"  Warning: beam plot failed ({e})")


def export_lsdyna_mesh(filename, outdir="parsed_model", plot=True):
    filename = Path(filename)
    outdir = Path(outdir)

    if outdir.exists():
        shutil.rmtree(outdir)

    binary_dir = outdir / "binary"
    debug_dir = outdir / "debug"
    metadata_dir = outdir / "metadata"

    binary_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    deck = lsdyna_mesh_reader.Deck(str(filename))

    # -------------------------
    # NODES
    # -------------------------
    node_section = deck.node_sections[0]

    nodes = np.column_stack([
        node_section.nid,
        node_section.coordinates
    ])

    np.save(binary_dir / "nodes.npy", nodes)

    pd.DataFrame(
        nodes[:100],
        columns=["nid", "x", "y", "z"]
    ).to_csv(debug_dir / "nodes_preview.csv", index=False)

    # -------------------------
    # SHELL ELEMENTS
    # -------------------------
    n_shells = 0
    shell_conn = None

    if len(deck.element_shell_sections) > 0:
        shell_rows = []
        shell_pids = []

        for sec in deck.element_shell_sections:
            for i in range(len(sec.eid)):
                eid = sec.eid[i]
                pid = sec.pid[i]
                conn = sec.node_ids[
                    sec.node_id_offsets[i]:sec.node_id_offsets[i + 1]
                ]

                row = [eid] + list(conn)
                shell_rows.append(row)
                shell_pids.append(pid)

        max_shell_nodes = max(len(r) - 1 for r in shell_rows)
        shell_conn = np.zeros((len(shell_rows), 1 + max_shell_nodes), dtype=np.int64)

        for i, r in enumerate(shell_rows):
            shell_conn[i, :len(r)] = r

        shell_pid = np.array(shell_pids, dtype=np.int64)

        np.save(binary_dir / "shell_conn.npy", shell_conn)
        np.save(binary_dir / "shell_pid.npy", shell_pid)

        cols = ["eid"] + [f"n{i}" for i in range(1, max_shell_nodes + 1)]
        pd.DataFrame(shell_conn[:100], columns=cols).to_csv(
            debug_dir / "shell_preview.csv", index=False
        )

        n_shells = len(shell_conn)

    # -------------------------
    # SOLID ELEMENTS
    # -------------------------
    n_solids = 0
    solid_conn = None

    if len(deck.element_solid_sections) > 0:
        solid_rows = []
        solid_pids = []

        for sec in deck.element_solid_sections:
            for i in range(len(sec.eid)):
                eid = sec.eid[i]
                pid = sec.pid[i]
                conn = sec.node_ids[
                    sec.node_id_offsets[i]:sec.node_id_offsets[i + 1]
                ]

                row = [eid] + list(conn)
                solid_rows.append(row)
                solid_pids.append(pid)

        max_solid_nodes = max(len(r) - 1 for r in solid_rows)
        solid_conn = np.zeros((len(solid_rows), 1 + max_solid_nodes), dtype=np.int64)

        for i, r in enumerate(solid_rows):
            solid_conn[i, :len(r)] = r

        solid_pid = np.array(solid_pids, dtype=np.int64)

        np.save(binary_dir / "solid_conn.npy", solid_conn)
        np.save(binary_dir / "solid_pid.npy", solid_pid)

        cols = ["eid"] + [f"n{i}" for i in range(1, max_solid_nodes + 1)]
        pd.DataFrame(solid_conn[:100], columns=cols).to_csv(
            debug_dir / "solid_preview.csv", index=False
        )

        n_solids = len(solid_conn)

    # -------------------------
    # BEAM ELEMENTS
    # custom parser
    # -------------------------
    beams = read_beam_orientation_connectivity(filename)

    n_beams = len(beams)
    beam_conn = None
    beam_orient = None

    if n_beams > 0:
        beam_conn = np.array(
            [[b["eid"], b["n1"], b["n2"]] for b in beams],
            dtype=np.int64
        )

        beam_pid = np.array(
            [b["pid"] for b in beams],
            dtype=np.int64
        )

        beam_orient = np.array(
            [[b["vx"], b["vy"], b["vz"]] for b in beams],
            dtype=np.float64
        )

        np.save(binary_dir / "beam_conn.npy", beam_conn)
        np.save(binary_dir / "beam_pid.npy", beam_pid)
        np.save(binary_dir / "beam_orient.npy", beam_orient)

        pd.DataFrame(
            np.column_stack([beam_conn, beam_pid, beam_orient])[:100],
            columns=["eid", "n1", "n2", "pid", "vx", "vy", "vz"]
        ).to_csv(debug_dir / "beam_preview.csv", index=False)

    # -------------------------
    # PLOTS
    # -------------------------
    if plot:
        _plot_mesh(
            nodes=nodes,
            shell_conn=shell_conn,
            solid_conn=solid_conn,
            beam_conn=beam_conn,
            beam_orient=beam_orient,
            plot_dir=outdir / "plots",
        )

    # -------------------------
    # SUMMARY
    # -------------------------
    summary = {
        "source_file": str(filename),
        "n_nodes": int(len(nodes)),
        "n_shells": int(n_shells),
        "n_solids": int(n_solids),
        "n_beams": int(n_beams),
        "node_columns": ["nid", "x", "y", "z"],
        "shell_conn_columns": "eid,n1,n2,n3,n4,...",
        "solid_conn_columns": "eid,n1,n2,n3,n4,n5,n6,n7,n8,...",
        "beam_conn_columns": ["eid", "n1", "n2"],
        "beam_orient_columns": ["vx", "vy", "vz"],
    }

    with open(metadata_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Done.")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse an LS-DYNA .k mesh file and export to CSV/JSON.")
    parser.add_argument("input", help="Path to the LS-DYNA .k input file")
    parser.add_argument("--outdir", default="parsed_model", help="Output directory (default: parsed_model)")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plot PNGs")
    args = parser.parse_args()

    export_lsdyna_mesh(args.input, outdir=args.outdir, plot=not args.no_plot)