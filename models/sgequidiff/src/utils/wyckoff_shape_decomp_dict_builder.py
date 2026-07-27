import json
import cloudpickle
import os
import math

from scipy.spatial import ConvexHull
import meshpy.tet as mptet
import meshpy.triangle as mptri
import numpy as np

from utils.io_utils import ASU_DICT_PATH, SHAPE_DECOMP_DICT_PATH
from aliases import *


# Helper functions
def to_array(region: List[List[str]]):
    return np.array([[eval(p) for p in pt] for pt in region])


def to_affine_transform(simplicial_complex: np.array):
    """
    Args:
        simplicial_complex: shape (n_dim+1, 3) array
    Returns:
        offset: shape (,3) array
        basis_maps: shape (n_dim, 3) array
    """
    assert simplicial_complex.shape[0] > 1
    offset = simplicial_complex[0]
    basis_maps = []
    for vec in simplicial_complex[1:]:
        basis_maps.append(vec - offset)
    return offset, np.stack(basis_maps)


def get_mesh_volume(mesh, dim: int) -> float:
    mesh_pts = np.array(mesh.points)
    total_volume = 0.0
    for e in mesh.elements:
        simplicial_complex = mesh_pts[np.array(e)]  # (dim+1, dim)
        b, A = to_affine_transform(simplicial_complex)
        # Nice way to compute the volume of a simplicial complex
        volume_correction = np.linalg.det(A @ A.T) ** 0.5
        volume = volume_correction / math.factorial(dim)
        total_volume += volume
    return total_volume


def project_to_2d(region: np.array):
    """
    Args:
        region: shape (n_facet_vertices, 3) array of floats
    Returns:
        mesh:
        pullback_map:
        volume_correction:
    """
    # Given a set of points on a plane, project them to 2d

    offset = region[0]  # (3,)
    # normalize the plane to include the origin
    normalized_pts = region - offset  # (n_facet_vertices, 3)
    # Get basis vectors for the plane
    basis_vecs = normalized_pts[1:3]  # (2, 3)
    volume_correction = (
        np.linalg.det(basis_vecs @ basis_vecs.T) ** 0.5
    )  # area spanned by 'basis_vecs'

    # Project the points to 2d
    pinv = np.linalg.pinv(basis_vecs)  # (3, 2)
    pts_2d = normalized_pts @ pinv

    # Build the mesh in 2d using the projected points
    hull = ConvexHull(pts_2d)
    mesh_info = mptri.MeshInfo()
    mesh_info.set_points(hull.points)
    mesh_info.set_facets(hull.simplices)
    mesh = mptri.build(mesh_info)

    # Build the function that will map us back to 3d coords
    pullback_map = lambda x: x @ basis_vecs + offset

    return mesh, pullback_map, volume_correction


### End helper functions


def build_wyckoff_shape_decomposition_dict():
    if os.path.exists(SHAPE_DECOMP_DICT_PATH.as_posix()):
        return

    # Create the decomposition dictionary. To start, open the ASU dict
    with open(ASU_DICT_PATH.as_posix(), "rb") as f:
        asu_dict = json.load(f)

    shape_decomposition_dict = {}
    for sg, sg_dict in asu_dict.items():
        sg_shape_decomposition_dict = {}
        for wyckoff_letter in sg_dict["ordered_wyckoff_letters"]:
            geometry = sg_dict[wyckoff_letter]

            # Get the dimension of the wyckoff site.
            site_dim = int(geometry["dim"])

            if site_dim == 0:
                wyckoff_shapes_info = {"dim": 0, "volumes": None}

            elif site_dim == 1:
                wyckoff_shapes_info = {"dim": 1, "volumes": []}
                for interval in geometry["vertices"]:
                    arr_interval = to_array(interval)
                    b, A = to_affine_transform(arr_interval)
                    volume = (A @ A.T) ** 0.5
                    wyckoff_shapes_info["volumes"].append(float(volume.squeeze()))

            elif site_dim == 2:
                wyckoff_shapes_info = {"dim": 2, "volumes": []}

                wyckoff_position_triangles = []
                # len(n_facets) list of (n_triangles, n_vertices=3, ndim=3) arrays
                wyckoff_position_triangle_areas = []
                # len(n_facets) list of (n_triangles,) arrays
                max_delaunay_triangles_per_facet = 0
                for facet in geometry["vertices"]:
                    # facet: shape (n_facet_vertices, 3) nested list of strings
                    arr_region = to_array(facet)  # shape (n_facet_vertices, 3) array of floats

                    # Project to 2d, do computation, and map back to 3d
                    mesh_2d, pullback_map, volume_correction = project_to_2d(arr_region)
                    mesh_vol = get_mesh_volume(mesh_2d, site_dim)

                    # -- For uniform sampling inside 2d Wyckoffs later, save
                    # triangles and their areas
                    facet_triangle_vertices, facet_triangle_areas = [], []
                    for e in mesh_2d.elements:  # Loop over triangles
                        triangle_vertices = np.array(mesh_2d.points)[np.array(e)]
                        # (3, 2)
                        b, A = to_affine_transform(triangle_vertices)
                        triangle_area = (np.linalg.det(A @ A.T) ** 0.5) / 2.0
                        facet_triangle_vertices.append(pullback_map(triangle_vertices))
                        facet_triangle_areas.append(triangle_area)
                    facet_triangle_vertices = np.stack(facet_triangle_vertices, axis=0)
                    # (n_triangles_in_facet, 3, 3)
                    facet_triangle_areas = np.stack(facet_triangle_areas, axis=0)
                    # (n_triangles_in_facet,)
                    max_delaunay_triangles_per_facet = max(
                        max_delaunay_triangles_per_facet, facet_triangle_areas.shape[0]
                    )
                    wyckoff_position_triangles.append(facet_triangle_vertices)
                    wyckoff_position_triangle_areas.append(facet_triangle_areas)
                    # --

                    wyckoff_shapes_info["volumes"].append(float(mesh_vol * volume_correction))
                wyckoff_shapes_info["facet_triangles"] = wyckoff_position_triangles
                wyckoff_shapes_info["facet_triangle_areas"] = wyckoff_position_triangle_areas
                wyckoff_shapes_info["max_triangles_per_facet"] = max_delaunay_triangles_per_facet

            elif site_dim == 3:
                wyckoff_shapes_info = {"dim": 3, "volumes": []}
                # All the 3d wyckoff positions only have a single region, so no loop is necessary
                arr_region = to_array(geometry["vertices"])
                hull = ConvexHull(arr_region)
                mesh_info = mptet.MeshInfo()
                mesh_info.set_points(hull.points)
                mesh_info.set_facets(hull.simplices)
                mesh = mptet.build(mesh_info)
                mesh_vol = get_mesh_volume(mesh, site_dim)
                wyckoff_shapes_info["volumes"].append(float(mesh_vol))
            else:
                raise ValueError("Site dimension must be in [0, 1, 2, 3]")
            # Update the spacegroup dictionary
            sg_shape_decomposition_dict[wyckoff_letter] = wyckoff_shapes_info

        # Finally, update the full dictionary with the spacegroup dictionary
        shape_decomposition_dict[sg] = sg_shape_decomposition_dict

    with open(SHAPE_DECOMP_DICT_PATH.as_posix(), "wb") as f:
        cloudpickle.dump(shape_decomposition_dict, f)
