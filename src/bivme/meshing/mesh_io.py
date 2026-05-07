from pathlib import Path

import numpy as np
import os
import pyvista as pv

def write_vtk_surface(filename: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Write a triangle mesh as legacy VTK PolyData (ASCII).
    `vertices`: (N,3) float
    `faces`: (M,3) int, 0-based indexing
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (N,3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (M,3) of triangle indices")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("faces contain out-of-range vertex indices")

    n_pts = vertices.shape[0]
    n_tri = faces.shape[0]

    with open(filename, "w", newline="\n") as f:
        # Header
        f.write("# vtk DataFile Version 4.2\n")
        f.write("Triangle mesh\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n_pts} float\n")
        for v in vertices:
            # Match your OBJ precision style; tweak if you want more decimals
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Triangles as POLYGONS: line format is "3 i j k"
        # Second integer is the total number of integers following: n_tri * (3+1)
        f.write(f"POLYGONS {n_tri} {n_tri * 4}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

def write_colored_vtk_surface(filename: str, vertices: np.ndarray, faces: np.ndarray, colormat: np.ndarray) -> None:
    """
    Write a VTK surface mesh.

    Parameters
    ----------
    filename : The name of the output VTK file.
    vertices : An array of shape (N, 3) representing the vertex coordinates.
    faces : An array of shape (M, 3) representing the triangular faces.

    Returns
    -------
    None
    """

    if np.__version__ >= '1.20.0': # for compatibility with later versions of numpy
        np.bool = np.bool_

    mesh = pv.PolyData(vertices, np.c_[np.ones(len(faces)) * 3, faces].astype(int))
    mesh["colors"] = colormat
    mesh.save(filename, binary=False)

def export_to_obj(file_name: os.PathLike, vertices: np.ndarray, faces: np.ndarray) -> None:
    file_name = str(file_name)
    if not os.path.basename(file_name).lower().endswith(".obj"):
        raise ValueError("filename should include .obj extension")

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)

    with open(file_name, 'w', newline="\n") as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        # OBJ is 1-based indexing
        for p in faces:
            f.write("f %d %d %d\n" % (p[0] + 1, p[1] + 1, p[2] + 1))

def export_mesh(output_format, output_dir, filename, vertices, faces, logger):
    writers = {
        ".vtk": ("vtk-meshes", write_vtk_surface),
        ".obj": ("obj-meshes", export_to_obj),
    }

    if output_format in writers:
        folder_name, writer_func = writers[output_format]
        output_folder_fmt = Path(output_dir, folder_name)
        output_folder_fmt.mkdir(exist_ok=True)
        mesh_path = output_folder_fmt / filename

        writer_func(str(mesh_path), vertices, faces)
        logger.success(f"{filename} successfully saved to {output_folder_fmt}")
    else:
        logger.error("argument format must be .obj or .vtk")