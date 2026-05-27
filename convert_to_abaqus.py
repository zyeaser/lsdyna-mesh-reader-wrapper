#!/usr/bin/env python3
"""
Convert parsed LS-DYNA binary data to Abaqus INP format.

Element mapping:
  Beam  -> B31
  Shell -> S4R
  Solid -> C3D8R

Beam cross-section is PIPE by default.
See the CONFIGURATION block below to change section type or dimensions.
"""

import numpy as np
import os
from collections import defaultdict

# ===========================================================================
# CONFIGURATION — edit this block only
# ===========================================================================

# Paths
BINARY_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'binary')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.inp')

# Material (single material for now — extend as needed)
MATERIAL_NAME = 'STEEL'
MATERIAL_E    = 200000.0   # Young's modulus  (match model units)
MATERIAL_NU   = 0.3        # Poisson's ratio

# ---------------------------------------------------------------------------
# Beam cross-section
# ---------------------------------------------------------------------------
# Change BEAM_SECTION_TYPE to any valid Abaqus beam section keyword, e.g.:
#   'PIPE'  -> circular hollow  — needs (outer_radius, wall_thickness)
#   'CIRC'  -> solid circle     — needs (radius,)
#   'BOX'   -> rectangular hollow — needs (a, b, t1, t2, t3, t4)
#   'I'     -> I-section        — needs (l, h, b1, b2, t1, t2, t3)
#   'L'     -> angle            — needs (a, b, t1, t2)
# Only PIPE is wired up below; add elif branches for other types.
BEAM_SECTION_TYPE = 'PIPE'

# PIPE parameters  (ignored when section type is not PIPE)
BEAM_PIPE_OUTER_RADIUS   = 0.05    # outer radius    (model units)
BEAM_PIPE_WALL_THICKNESS = 0.005   # wall thickness  (model units)

# Shell thickness
SHELL_THICKNESS = 0.01   # (model units)

# ===========================================================================
# END CONFIGURATION
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v):
    """Compact float string — no trailing zeros."""
    return f'{float(v):.6g}'


def _beam_section_data_line(section_type):
    """Return the geometry data line(s) for the chosen beam section type."""
    st = section_type.upper()
    if st == 'PIPE':
        return f' {_fmt(BEAM_PIPE_OUTER_RADIUS)}, {_fmt(BEAM_PIPE_WALL_THICKNESS)}'
    # --- add more section types here ---
    # elif st == 'CIRC':
    #     return f' {_fmt(BEAM_CIRC_RADIUS)}'
    # elif st == 'BOX':
    #     return f' {_fmt(a)}, {_fmt(b)}, {_fmt(t1)}, {_fmt(t2)}, {_fmt(t3)}, {_fmt(t4)}'
    else:
        raise ValueError(
            f'Section type "{section_type}" is not configured. '
            'Add a branch in _beam_section_data_line().'
        )


def _safe_name(name, maxlen=80):
    """Sanitize an Abaqus keyword argument (no spaces, max 80 chars)."""
    return name.replace(' ', '_').replace('.', 'p').replace('-', 'N')[:maxlen]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(binary_dir):
    def p(name):
        return os.path.join(binary_dir, name)

    nodes       = np.load(p('nodes.npy'))        # (N, 4)  nid x y z
    beam_conn   = np.load(p('beam_conn.npy'))     # (B, 3)  eid n1 n2
    beam_pid    = np.load(p('beam_pid.npy'))      # (B,)    pid
    beam_orient = np.load(p('beam_orient.npy'))   # (B, 3)  vx vy vz
    shell_conn  = np.load(p('shell_conn.npy'))    # (S, 5)  eid n1 n2 n3 n4
    shell_pid   = np.load(p('shell_pid.npy'))     # (S,)    pid
    solid_conn  = np.load(p('solid_conn.npy'))    # (V, 9)  eid n1..n8
    solid_pid   = np.load(p('solid_pid.npy'))     # (V,)    pid

    return (nodes,
            beam_conn, beam_pid.ravel().astype(int), beam_orient,
            shell_conn, shell_pid.ravel().astype(int),
            solid_conn, solid_pid.ravel().astype(int))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_beams(beam_pid, beam_orient):
    """
    Group beam row indices by (pid, rounded_orientation_tuple).

    Returns: dict[(pid, ov_tuple)] -> list[row_index]
    """
    groups = defaultdict(list)
    for i, (pid, ov) in enumerate(zip(beam_pid, beam_orient)):
        key = (int(pid), tuple(float(round(v, 6)) for v in ov))
        groups[key].append(i)
    return groups


def group_by_pid(pid_array):
    """Group row indices by pid. Returns dict[pid] -> list[row_index]."""
    groups = defaultdict(list)
    for i, pid in enumerate(pid_array):
        groups[int(pid)].append(i)
    return groups


# ---------------------------------------------------------------------------
# INP writer
# ---------------------------------------------------------------------------

def write_inp(output_file,
              nodes,
              beam_conn, beam_pid, beam_orient,
              shell_conn, shell_pid,
              solid_conn, solid_pid):

    beam_groups  = group_beams(beam_pid, beam_orient)
    shell_groups = group_by_pid(shell_pid)
    solid_groups = group_by_pid(solid_pid)

    # Build beam elset names  [(elset_name, pid, ov_tuple), ...]
    # Name: BEAM_PID{pid}_{counter}  (counter distinguishes orientation variants)
    pid_counter = defaultdict(int)
    beam_elsets = []
    for (pid, ov_tuple) in sorted(beam_groups.keys()):
        cnt = pid_counter[pid]
        pid_counter[pid] += 1
        if cnt == 0:
            name = _safe_name(f'BEAM_PID{pid}')
        else:
            name = _safe_name(f'BEAM_PID{pid}_{cnt}')
        beam_elsets.append((name, pid, ov_tuple))

    shell_elsets = [(_safe_name(f'SHELL_PID{pid}'), pid)
                    for pid in sorted(shell_groups.keys())]
    solid_elsets = [(_safe_name(f'SOLID_PID{pid}'), pid)
                    for pid in sorted(solid_groups.keys())]

    with open(output_file, 'w') as f:
        # --- Heading ---
        f.write('*Heading\n')
        f.write('** Abaqus model — converted from LS-DYNA\n')
        f.write('** Beams: B31  |  Shells: S4R  |  Solids: C3D8R\n')
        f.write('**\n')

        # --- Nodes ---
        f.write('*Node\n')
        for row in nodes:
            nid = int(row[0])
            x, y, z = row[1], row[2], row[3]
            f.write(f' {nid}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}\n')

        # --- Beam elements ---
        f.write('**\n** BEAM ELEMENTS (B31)\n**\n')
        for (name, pid, ov_tuple), (key, indices) in zip(
                beam_elsets, sorted(beam_groups.items())):
            f.write(f'*Element, type=B31, elset={name}\n')
            for i in indices:
                row = beam_conn[i]
                eid, n1, n2 = int(row[0]), int(row[1]), int(row[2])
                f.write(f' {eid}, {n1}, {n2}\n')

        # --- Shell elements ---
        f.write('**\n** SHELL ELEMENTS (S4R)\n**\n')
        for (name, pid) in shell_elsets:
            f.write(f'*Element, type=S4R, elset={name}\n')
            for i in shell_groups[pid]:
                row = shell_conn[i]
                eid = int(row[0])
                ns  = ', '.join(str(int(n)) for n in row[1:])
                f.write(f' {eid}, {ns}\n')

        # --- Solid elements ---
        f.write('**\n** SOLID ELEMENTS (C3D8R)\n**\n')
        for (name, pid) in solid_elsets:
            f.write(f'*Element, type=C3D8R, elset={name}\n')
            for i in solid_groups[pid]:
                row = solid_conn[i]
                eid = int(row[0])
                ns  = ', '.join(str(int(n)) for n in row[1:])
                f.write(f' {eid}, {ns}\n')

        # --- Beam sections ---
        f.write('**\n** BEAM SECTIONS\n**\n')
        for (name, pid, ov_tuple) in beam_elsets:
            vx, vy, vz = ov_tuple
            f.write(
                f'*Beam Section, elset={name}, '
                f'material={MATERIAL_NAME}, section={BEAM_SECTION_TYPE}\n'
            )
            f.write(_beam_section_data_line(BEAM_SECTION_TYPE) + '\n')
            # n1 direction — approximate local 1-axis of cross-section
            # NOTE: must NOT be parallel to the beam axis; Abaqus will error if it is.
            f.write(f' {_fmt(vx)}, {_fmt(vy)}, {_fmt(vz)}\n')

        # --- Shell sections ---
        f.write('**\n** SHELL SECTIONS\n**\n')
        for (name, pid) in shell_elsets:
            f.write(f'*Shell Section, elset={name}, material={MATERIAL_NAME}\n')
            f.write(f' {_fmt(SHELL_THICKNESS)}, 5\n')  # thickness, Simpson pts

        # --- Solid sections ---
        f.write('**\n** SOLID SECTIONS\n**\n')
        for (name, pid) in solid_elsets:
            f.write(f'*Solid Section, elset={name}, material={MATERIAL_NAME}\n')
            f.write('\n')

        # --- Material ---
        f.write('**\n** MATERIAL\n**\n')
        f.write(f'*Material, name={MATERIAL_NAME}\n')
        f.write('*Elastic\n')
        f.write(f' {_fmt(MATERIAL_E)}, {_fmt(MATERIAL_NU)}\n')

        # --- Step placeholder ---
        f.write('**\n** STEP (placeholder — add BCs and loads here)\n**\n')
        f.write('** *Step, name=Step-1\n')
        f.write('** *Static\n')
        f.write('**  1., 1., 1e-05, 1.\n')
        f.write('** *End Step\n')

    print(f'Written: {output_file}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading binary data ...')
    data = load_data(BINARY_DIR)
    nodes, beam_conn, beam_pid, beam_orient, shell_conn, shell_pid, solid_conn, solid_pid = data

    print(f'  Nodes : {len(nodes):>7,}')
    print(f'  Beams : {len(beam_conn):>7,}')
    print(f'  Shells: {len(shell_conn):>7,}')
    print(f'  Solids: {len(solid_conn):>7,}')

    n_beam_pids  = len(np.unique(beam_pid))
    n_shell_pids = len(np.unique(shell_pid))
    n_solid_pids = len(np.unique(solid_pid))
    print(f'  Beam PIDs : {n_beam_pids}')
    print(f'  Shell PIDs: {n_shell_pids}')
    print(f'  Solid PIDs: {n_solid_pids}')

    print('\nWriting INP file ...')
    write_inp(OUTPUT_FILE, *data)
    print('Done.')


if __name__ == '__main__':
    main()
