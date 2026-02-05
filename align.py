import argparse
import gc
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def load_ply(path: Path):
    """Loads PLY data safely."""
    if not path.exists():
        print(f"❌ Error: Input file does not exist: {path}")
        sys.exit(1)

    try:
        return PlyData.read(str(path))
    except Exception as e:
        print(f"❌ Error loading PLY file: {e}")
        sys.exit(1)


def save_ply(data: dict, output_path: Path):
    """Creates a PlyElement and writes to disk."""
    # Create the element
    el = PlyElement.describe(data, "vertex")

    # Write to file
    try:
        PlyData([el], text=False).write(str(output_path))
        print(f"💾 Saved modified PLY to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving PLY: {e}")


def get_quaternions(vertex_data):
    """Extracts quaternions in (x, y, z, w) format for Scipy."""
    # Try standard 3DGS naming first (rot_0 is usually W/Real part)
    # Scipy expects (x, y, z, w)
    try:
        # Assuming rot_1, rot_2, rot_3 are x,y,z and rot_0 is w
        q = np.stack(
            [
                vertex_data["rot_1"],
                vertex_data["rot_2"],
                vertex_data["rot_3"],
                vertex_data["rot_0"],
            ],
            axis=-1,
        )
    except (KeyError, ValueError):
        try:
            # Fallback for other conventions
            q = np.stack(
                [
                    vertex_data["q1"],
                    vertex_data["q2"],
                    vertex_data["q3"],
                    vertex_data["q0"],
                ],
                axis=-1,
            )
        except KeyError:
            print("⚠️ Warning: No rotation fields found. Skipping rotation logic.")
            return None
    return q


def process_splat(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"--- 🚀 Processing Splat: {input_path.name} ---")

    # 1. Load
    plydata = load_ply(input_path)
    vertex = plydata["vertex"]

    # 2. Extract Positions
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    # Center the model
    center = np.median(positions, axis=0)
    centered_positions = positions - center

    # 3. Create Transformation Matrix
    # Order: Scale -> Rotate
    r = R.from_euler("xy", [args.rotate_x, args.rotate_y], degrees=True)
    rot_matrix = r.as_matrix().astype(np.float32)

    # Apply Scale
    if args.scale != 1.0:
        centered_positions *= args.scale

    # Apply Rotation to Positions
    # (N, 3) @ (3, 3).T
    rotated_positions = centered_positions @ rot_matrix.T

    # 4. Rotate Quaternions
    q_raw = get_quaternions(vertex)

    if q_raw is not None:
        orig_rotations = R.from_quat(q_raw)
        global_rot = R.from_matrix(rot_matrix)

        # Apply global rotation to local rotations
        new_rotations = global_rot * orig_rotations

        # Convert back to quat (scipy returns x, y, z, w)
        # We enforce normalization to prevent artifacting
        new_q = new_rotations.as_quat()
        # Normalize (scipy usually does, but double check for float precision)
        norms = np.linalg.norm(new_q, axis=1, keepdims=True)
        new_q = new_q / norms
    else:
        new_q = None

    # 5. Prepare Output Data
    # Copy original data structure to preserve color, opacity, SH, etc.
    new_vertices = vertex.data.copy()

    # Update Positions
    new_vertices["x"] = rotated_positions[:, 0]
    new_vertices["y"] = rotated_positions[:, 1]
    new_vertices["z"] = rotated_positions[:, 2]

    # Update Rotations if they existed
    if new_q is not None:
        # Mapping back to: rot_0 (w), rot_1 (x), rot_2 (y), rot_3 (z)
        new_vertices["rot_0"] = new_q[:, 3]  # W
        new_vertices["rot_1"] = new_q[:, 0]  # X
        new_vertices["rot_2"] = new_q[:, 1]  # Y
        new_vertices["rot_3"] = new_q[:, 2]  # Z

    # Free memory before saving
    del plydata, vertex, positions, rotated_positions, q_raw
    gc.collect()

    # Save
    save_ply(new_vertices, output_path)

    # 6. Compression
    if args.compress:
        run_compression(output_path, args)


def run_compression(input_ply: Path, args):
    print("\n--- 📦 Running 3dgsconverter ---")

    # Check if tool exists
    if shutil.which("3dgsconverter") is None:
        print("❌ Error: '3dgsconverter' not found in PATH.")
        print(
            "   Install it via: pip install 3dgsconverter (or check your environment)"
        )
        return

    spz_output = input_ply.with_suffix(".spz")

    cmd = [
        "3dgsconverter",
        "-i",
        str(input_ply),
        "-o",
        str(spz_output),
        "--target_format",
        "spz",
        "--compression_level",
        str(args.level),
        "--sh_level",
        str(args.sh),
        "--min_opacity",
        str(args.opacity),
        "--force",
    ]

    if args.sor > 0:
        cmd += ["--sor_intensity", str(args.sor)]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Final compressed file ready: {spz_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Compression failed with error code {e.returncode}.")
    except Exception as e:
        print(f"❌ Unexpected error during compression: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Gaussian Splats: Center, Rotate, Scale, and Compress.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # File IO
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("input", help="Path to input .ply file")
    io_group.add_argument("output", help="Path to output .ply file")

    # Transformation
    trans_group = parser.add_argument_group("Transformation")
    trans_group.add_argument(
        "--rotate_x", type=float, default=90.0, help="Rotation around X axis (degrees)"
    )
    trans_group.add_argument(
        "--rotate_y", type=float, default=0.0, help="Rotation around Y axis (degrees)"
    )
    trans_group.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor (e.g. 0.5 for half size)"
    )

    # Compression
    comp_group = parser.add_argument_group("Compression (3dgsconverter)")
    comp_group.add_argument(
        "--compress", action="store_true", help="Run compression after processing"
    )
    comp_group.add_argument(
        "--level", type=int, default=7, help="Gzip compression level (1-9)"
    )
    comp_group.add_argument("--sh", type=int, default=3, help="SH degree to keep (0-3)")
    comp_group.add_argument(
        "--opacity", type=int, default=0, help="Minimum opacity culling (0-255)"
    )
    comp_group.add_argument(
        "--sor",
        type=int,
        default=0,
        help="Statistical Outlier Removal intensity (0 to disable)",
    )

    args = parser.parse_args()

    process_splat(args)
