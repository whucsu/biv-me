from scipy.io import loadmat
from scipy.spatial import cKDTree
from collections import OrderedDict

# local imports
from .GPDataSet import *
from .surface_enum import Surface
from .surface_enum import SURFACE_CONTOUR_MAP
from .build_model_tools import *


##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz
class BiventricularModel:
    """This class creates a surface from the control mesh, based on
    Catmull-Clark subdivision surface method. Surfaces have the following properties:

    Attributes:
       num_nodes = 388                       Number of control nodes.
       NUM_ELEMENTS = 187                    Number of elements.
       num_surface_nodes = 5810               Number of nodes after subdivision
                                            (surface points).
       control_mesh                         Array of x,y,z coordinates of
                                            control mesh (388x3).
       et_vertex_xi                         local xi position (xi1,xi2,xi3)
                                            for each vertex (5810x3).

       et_pos                               Array of x,y,z coordinates for each
                                            surface nodes (5810x3).
       et_vertex_element_num                Element num for each surface
                                            nodes (5810x1).
       et_indices                           Elements connectivities (n1,n2,n3)
                                            for each face (11760x3).
       et_indices_control_mesh              Element connectivities (n1, n2, n3) for each face of the coarse mesh (708x3)
       basis_matrix                         Matrix (5810x388) containing basis
                                            functions used to evaluate surface
                                            at surface point locations
       matrix                               Subdivision matrix (388x5810).


       gtstsg_x, gtstsg_y, gtstsg_z         Regularization/Smoothing matrices
                                            (388x388) along
                                            Xi1 (circumferential),
                                            Xi2 (longitudinal) and
                                            Xi3 (transmural) directions


       APEX_INDEX                           Vertex index of the apex

       et_vertex_start_end                  Surface index limits for vertices
                                            et_pos. Surfaces are sorted in
                                            the following order:
                                            LV_ENDOCARDIAL, RV septum, RV free wall,
                                            epicardium, mitral valve, aorta,
                                            tricuspid, pulmonary valve,
                                            RV insert.
                                            Valve centroids are always the last
                                            vertex of the corresponding surface

       surface_start_end                    Surface index limits for embedded
                                            triangles et_indices.
                                            Surfaces are sorted in the following order:
                                            LV_ENDOCARDIAL, RV septum, RV free wall,
                                            epicardium, mitral valve, aorta,
                                            tricuspid, pulmonary valve, RV insert.

       mbder_dx, mbder_dy, mbder_dz         Matrices (5049x338) containing
                                            weights used to calculate gradients
                                            of the displacement field at Gauss
                                            point locations.

       Jac11, jac_12, jac_13                  Matrices (11968x388) containing
                                            weights used to calculate Jacobians
                                            at Gauss point location (11968x338).
                                            Each matrix element is a linear
                                            combination of the 388 control points.
                                            J11 contains the weight used to
                                            compute the derivatives along Xi1,
                                            J12 along Xi2 and J13 along Xi3.
                                            Jacobian determinant is
                                            calculated/checked on 11968 locations.
       fraction                 gives the level of the patch
                                (level 0 = 1,level 1 = 0.5,level 2 = 0.25)
       b_spline                  gives the 32 control points which need to be weighted
                                (for each vertex)
       patch_coordinates        patch coordinates
       boundary                 boundary
       phantom_points           Some elements only have an epi surface.
                                The phantomt points are 'fake' points on
                                the endo surface.


    """

    NUM_NODES = 388
    """Class constant, Number of control nodes (388)."""
    NUM_ELEMENTS = 187
    """Class constant, number of elements (187)."""
    NUM_SURFACE_NODES = 5810
    """class constant, number of nodes after subdivision (5810)."""
    APEX_INDEX = 5485  # # 50 endo #5485 #epi
    """class constant, vertex index defined as the apex point."""
    NUM_GAUSSIAN_POINTS = 5049
    """Number of gaussian points"""
    NUM_NODES_THRU_WALL = 160
    """Number of points defining the thru wall"""
    NUM_SUBDIVIDED_FACES = 11760
    """Number of faces after subdivision"""
    NUM_COARSE_FACES = 708
    """Number of faces before subdivision"""
    NUM_LOCAL_POINTS = 12509
    """Number of local points - used for patch estimation"""

    control_mesh_vertex_start_end = np.array(
        [
            [0, 104],  # LV_ENDOCARDIAL
            [105, 210],  # RV_ENDOCARDIAL
            [211, 353],  # EPICARDIAL
        ]
    )

    et_vertex_start_end = np.array(
        [
            [0, 1499],
            [1500, 2164],
            [2165, 3223],
            [3224, 5581],
            [5582, 5630],
            [5631, 5655],
            [5656, 5696],
            [5697, 5729],
            [5730, 5809],
        ]
    )
    """Class constant, surface index limits for vertices `et_pos`. 
    Surfaces are defined in the following order:
    
        LV_ENDOCARDIAL = 0 
        RV_SEPTUM = 1
        RV_FREEWALL = 2
        EPICARDIAL =3
        MITRAL_VALVE =4
        AORTA_VALVE = 5
        TRICUSPID_VALVE = 6
        PULMONARY_VALVE = 7
        RV_INSERT = 8
    
    For a valve surface the centroids are defined last point of the 
    corresponding surface. To get surface end and start vertex index use 
    get_surface_vertex_start_end_index(surface_name)
    
    Example
    --------
        lv_endo_start_idx= et_vertex_start_end[0][0]
        lv_endo_end_idx= et_vertex_start_end[0][1]
        lv_aorta_end_idx= et_vertex_start_end[5][1]-1
        lv_aorta_centroid_idx= et_vertex_start_end[5][1]
        lv_endo_start_idx,lv_endo_end_idx = 
        mesh.get_surface_vertex_start_end_index(surface_name)
       
    surface_name as defined in `Surface` class
    """
    surface_start_end = np.array(
        [
            [0, 3071],
            [3072, 4479],
            [4480, 6751],
            [6752, 11615],
            [11616, 11663],
            [11664, 11687],
            [11688, 11727],
            [11728, 11759],
        ]
    )
    """Class constant,  surface index limits for embedded triangles `et_indices`.
    Surfaces are defined in the following order  
      
            LV_ENDOCARDIAL = 0 
            RV_SEPTUM = 1
            RV_FREEWALL = 2
            EPICARDIAL =3
            MITRAL_VALVE =4
            AORTA_VALVE = 5
            TRICUSPID_VALVE = 6
            PULMONARY_VALVE = 7
            RV_INSERT = 8
    
    To get surface end and start vertex index use 
    get_surface_start_end_index(surface_name)
    
    Example
    --------
        lv_endo_start_idx= surface_start_end[0][0]
        lv_endo_end_idx= surface_start_end[0][1]
        lv_aorta_start_idx= surface_start_end[5][0]
        lv_aorta_end_idx= surface_start_end[5][1]
        
        lv_endo_start_idx,lv_endo_end_idx = 
        mesh.get_surface_start_end_index(surface_name)
    """

    control_mesh_start_end = np.array(
        [
            [0, 191],  # LV_ENDOCARDIAL = 0
            [192, 421],  # RV_ENDOCARDIAL = 1
            [422, 707],  # EPICARDIAL = 2
        ]
    )
    """Class constant,  control mesh index limits for embedded triangles `et_indices_control_mesh`.
    Surfaces are defined in the following order

            LV_ENDOCARDIAL = 0
            RV_ENDOCARDIAL = 1
            EPICARDIAL = 2

    To get control mesh end and start vertex index use
    get_control_mesh_start_end_index(surface_name)
    """

    _CONST_ARRAYS = {
        "matrix", "et_indices", "material",
        "et_indices_control_mesh", "et_indices_thru_wall", "et_indices_epi_lvrv",
        "gtstsg_x", "gtstsg_y", "gtstsg_z",
        "et_vertex_element_num",
        "mbder_dx", "mbder_dy", "mbder_dz",
        "jac_11", "jac_12", "jac_13",
        "basis_matrix",
        # build_mode-only:
        "et_vertex_xi", "b_spline", "boundary", "control_et_indices",
        "phantom_points", "patch_coordinates", "fraction", "local_matrix",
    }

    # Attributes that typically CHANGE during fitting: copy (or recompute)
    _MUTABLE_ARRAYS = {"control_mesh", "et_pos"}
    _INPUT_MUTABLES = {"label", "build_mode", "collision_detection", "reference_collision"}

    def __init__(self, control_mesh_dir: os.PathLike,
                 label: str = "default",
                 build_mode: bool = False,
                 collision_detection: bool = False):
        """Return a Surface object whose control mesh should be
        fitted to the dataset *DataSet*.

        control_mesh is always the same - this is the RVLV template. If
        the template is changed, one needs to regenerate all the matrices.
        The build_mode allows to load the data needed to interpolate a
        surface field. For fitting purposes set build_mode to False
        """
        self.label = label

        # False by default, true to evaluate surface points at xi local coordinates
        self.build_mode = build_mode

        # `numNodes`X3 array[float] of x,y,z coordinates of control mesh.
        assert control_mesh_dir.exists(), f"Cannot find {control_mesh_dir}!"
        model_file = control_mesh_dir / "model.txt"
        assert model_file.exists(), f"Missing {model_file}!"
        self.control_mesh = np.loadtxt(model_file)

        # Subdivision matrix (`numNodes`x`numSurfaceNodes`).
        subdivision_matrix_file = control_mesh_dir / "subdivision_matrix_sparse.mat"
        assert subdivision_matrix_file.exists(), f"Missing {subdivision_matrix_file}!"
        self.matrix = loadmat(subdivision_matrix_file)['S'].toarray()

        # `numSurfaceNodes`x3 array[float] of x,y,z coordinates for each surface nodes.
        self.et_pos = np.dot(self.matrix, self.control_mesh)

        # 11760x3 array[int] of elements connectivity (n1,n2,n3) for each face.
        et_index_file = control_mesh_dir / "ETIndicesSorted.txt"
        assert et_index_file.exists(), f"Missing {et_index_file}!"
        self.et_indices = np.loadtxt(et_index_file, dtype=int) - 1

        material_file = control_mesh_dir / 'ETIndicesMaterials.txt'
        assert material_file.exists(), f"Missing {et_index_file}"
        self.material = np.loadtxt(material_file, dtype='str')

        # Collision detection on or off
        self.collision_detection = collision_detection

        # Through-wall mesh indices for myocardial mass calculations
        et_index_thru_wall_file = control_mesh_dir / "thru_wall_et_indices.txt"
        assert et_index_thru_wall_file.exists(), f"Missing {et_index_thru_wall_file} for myocardial mass calculation"
        self.et_indices_thru_wall = np.loadtxt(et_index_thru_wall_file, dtype=int) - 1

        # 11760x3 array[int] of elements connectivity (n1,n2,n3) for each face of the coarse_mesh.
        et_index_file = control_mesh_dir / "ETIndices_control_mesh.txt"
        assert et_index_file.exists(), f"Missing {et_index_file}!"
        self.et_indices_control_mesh = np.loadtxt(et_index_file, dtype=int) - 1

        # RB addition for MyoMass calc
        et_index_epi_lvrv_file = control_mesh_dir / "ETIndicesEpiRVLV.txt"
        assert et_index_epi_lvrv_file.exists(), f"Missing {et_index_epi_lvrv_file} for myocardial mass calculation"
        self.et_indices_epi_lvrv = np.loadtxt(et_index_epi_lvrv_file, dtype=int) - 1

        # `numNodes`x`numNodes` Regularization/Smoothing matrix along Xi1 (circumferential direction)
        gtstsg_x_file = control_mesh_dir / "GTSTG_x_sparse.mat"
        assert gtstsg_x_file.exists(), f"Missing {gtstsg_x_file}"
        self.gtstsg_x = loadmat(gtstsg_x_file)['S'].toarray()

        # `numNodes`x`numNodes` Regularization/Smoothing matrix along Xi2 (longitudinal) direction
        gtstsg_y_file = control_mesh_dir / "GTSTG_y_sparse.mat"
        assert gtstsg_y_file.exists(), f"Missing {gtstsg_y_file}"
        self.gtstsg_y = loadmat(gtstsg_y_file)['S'].toarray()

        # `numNodes`x`numNodes` Regularization/Smoothing matrix along Xi3 (transmural) direction
        gtstsg_z_file = control_mesh_dir / "GTSTG_z_sparse.mat"
        assert gtstsg_z_file.exists(), f"Missing {gtstsg_z_file}"
        self.gtstsg_z = loadmat(gtstsg_z_file)['S'].toarray()

        # `numSurfaceNodes`x1 array[int] Element num for each surface nodes. Used for surface evaluation
        et_vertex_element_num_file = control_mesh_dir / "etVertexElementNum.txt"
        assert et_vertex_element_num_file.exists(), f"Missing {et_vertex_element_num_file}"
        self.et_vertex_element_num = np.loadtxt(et_vertex_element_num_file, dtype=int, usecols=0) - 1

        # `numSurfaceNodes`x`numNodes` Weighted matrix to calculate gradients of the displacement field at Gauss points
        mbder_x_file = control_mesh_dir / "mBder_x_sparse.mat"
        assert mbder_x_file.exists(), f"Missing {mbder_x_file}"
        self.mbder_dx = loadmat(mbder_x_file)['S'].toarray()

        mbder_y_file = control_mesh_dir / "mBder_y_sparse.mat"
        assert mbder_y_file.exists(), f"Missing {mbder_y_file}"
        self.mbder_dy = loadmat(mbder_y_file)['S'].toarray()

        mbder_z_file = control_mesh_dir / "mBder_z_sparse.mat"
        assert mbder_z_file.exists(), f"Missing {mbder_z_file}"
        self.mbder_dz = loadmat(mbder_z_file)['S'].toarray()

        # 11968 x `numNodes` matrix containing weights used to calculate Jacobians along Xi1 at Gauss point locations
        # Each matrix element is a linear combination of the 388 control points.
        jac_11_file = control_mesh_dir / "J11_sparse.mat"
        assert jac_11_file.exists(), f"Missing {jac_11_file}"
        self.jac_11 = loadmat(jac_11_file)['S'].toarray()

        # 11968 x `numNodes` matrix containing weights used to calculate Jacobians along Xi2 at Gauss point locations
        # Each matrix element is a linear combination of the 388 control points.
        jac_12_file = control_mesh_dir / "J12_sparse.mat"
        assert jac_12_file.exists(), f"Missing {jac_12_file}"
        self.jac_12 = loadmat(jac_12_file)['S'].toarray()

        # 11968 x `numNodes` matrix containing weights used to calculate Jacobians along Xi3 at Gauss point locations
        # Each matrix element is a linear combination of the 388 control points.
        jac_13_file = control_mesh_dir / "J13_sparse.mat"
        assert jac_13_file.exists(), f"Missing {jac_13_file}"
        self.jac_13 = loadmat(jac_13_file)['S'].toarray()

        # `numSurfaceNodes`x`numNodes` array[float] basis functions used to evaluate surface at surface points
        basic_matrix_file = control_mesh_dir / "basis_matrix_sparse.mat"
        assert basic_matrix_file.exists(), f"Missing {basic_matrix_file}"
        self.basis_matrix = loadmat(basic_matrix_file)['S'].toarray()

        # Build mode must be true from here on
        if not self.build_mode:
            return

        # `numSurfaceNodes`x3 array[float] of local xi position (xi1,xi2,xi3) for each vertex.
        et_vertex_xi_file = control_mesh_dir / "etVertexXi.txt"
        assert et_vertex_xi_file.exists(), f"Missing {et_vertex_xi_file}"
        self.et_vertex_xi = np.loadtxt(et_vertex_xi_file)

        # numSurfaceNodesX32 array[int] of 32 control points which need to be weighted (for each vertex)
        b_spline_file = control_mesh_dir / "control_points_patches.txt"
        assert b_spline_file.exists(), f"Missing {b_spline_file}"
        self.b_spline = np.loadtxt(b_spline_file, dtype=int) - 1

        # boundary
        boundary_file = control_mesh_dir / "boundary.txt"
        assert boundary_file.exists(), f"Missing {boundary_file}"
        self.boundary = np.loadtxt(boundary_file, dtype=int)

        # (K,8) matrix of control mesh connectivity
        control_ef_file = control_mesh_dir / "control_mesh_connectivity.txt"
        assert control_ef_file.exists(), f"Missing {control_ef_file}"
        self.control_et_indices = np.loadtxt(control_ef_file, dtype=int) - 1

        # Some surface nodes are not needed for the definition of the biventricular 2D surface therefore they are not
        # include in the surface node matrix. However they are necessary for the 3D interpolation (septum area).
        # These elements are called the phantom points and the corresponding information as the subdivision level,
        # localpatch coordinates etc. are stored in phantom points array.
        phantom_points_file = control_mesh_dir / "phantom_points.txt"
        assert phantom_points_file.exists(), f"Missing {phantom_points_file}"
        self.phantom_points = np.loadtxt(phantom_points_file, dtype=float)
        self.phantom_points[:, :17] = self.phantom_points[:, :17].astype(int) - 1

        # local patch coordinates
        patch_coordinates_file = control_mesh_dir / "patch_coordinates.txt"
        assert patch_coordinates_file.exists(), f"Missing {patch_coordinates_file}"
        self.patch_coordinates = np.loadtxt(patch_coordinates_file)

        # According to CC subdivision surface, to evaluate a point on a surface the original control mesh needs to be
        # subdivided in 'child' patches. The coordinates of the child patches are then used to map the local
        # coordinates with respect to control mesh in to the local coordinates with respect to child patch.
        # The patch coordinates and subdivision level of each surface node are pre-computed and here imported as
        # patch_coordinates and fraction. See:
        # Atlas-based Analysis of Biventricular Heart Shape and Motion in Congenital Heart Disease. C. Mauger (p34-37)

        # `numSurfaceNodes`x1 vector[int] subdivision level of the  patch (level 0 = 1,level 1 = 0.5,level 2 = 0.25).
        # See patch_coordinates for details`
        fraction_file = control_mesh_dir / "fraction.txt"
        assert fraction_file.exists(), f"Missing {fraction_file}"
        self.fraction = np.loadtxt(fraction_file)

        local_matrix_file = control_mesh_dir / "local_matrix_sparse.mat"
        assert local_matrix_file.exists(), f"Missing {local_matrix_file}"
        self.local_matrix = loadmat(local_matrix_file)['S'].toarray()

    def get_nodes(self) -> np.ndarray:
        """
        Returns
        --------
        `NUM_SURFACE_NODES`x3 array of vertices coordinates
        """
        return self.et_pos

    def get_control_mesh_nodes(self) -> np.ndarray:
        """
        Returns
        -------
        `NUM_NODES`x3 array of coordinates of control points
        """
        return self.control_mesh

    def get_surface_vertex_start_end_index(self, surface_name: Surface) -> np.ndarray:
        """Return first and last vertex index for a given surface to use
        with `et_pos` array

        Parameters
        -----------

        `surface_name`  Surface name as defined in 'Surface' enumeration

        `Returns`
        ---------
        2x1 array with first and last vertices index belonging to
            surface_name
        """

        if surface_name == Surface.LV_ENDOCARDIAL:
            return self.et_vertex_start_end[0, :]

        if surface_name == Surface.RV_SEPTUM:
            return self.et_vertex_start_end[1, :]

        if surface_name == Surface.RV_FREEWALL:
            return self.et_vertex_start_end[2, :]

        if surface_name == Surface.EPICARDIAL:
            return self.et_vertex_start_end[3, :]

        if surface_name == Surface.MITRAL_VALVE:
            return self.et_vertex_start_end[4, :]

        if surface_name == Surface.AORTA_VALVE:
            return self.et_vertex_start_end[5, :]

        if surface_name == Surface.TRICUSPID_VALVE:
            return self.et_vertex_start_end[6, :]

        if surface_name == Surface.PULMONARY_VALVE:
            return self.et_vertex_start_end[7, :]

        if surface_name == Surface.RV_INSERT:
            return self.et_vertex_start_end[8, :]

        if surface_name == Surface.APEX:
            return [self.APEX_INDEX] * 2

        return None

    def get_surface_faces(self, surface: Surface) -> np.ndarray:
        """Get the faces definition for a surface triangulation"""

        surface_index = self.get_surface_start_end_index(surface)
        return self.et_indices[surface_index[0]: surface_index[1] + 1, :]

    def get_surface_start_end_index(self, surface_name: Surface) -> np.ndarray:
        """Return first and last element index for a given surface, tu use
        with `et_indices` array
        Parameters
        ----------
        `surface_name` surface name as defined by `Surface` enum

        Returns
        -------
        2x1 array containing first and last vertices index belonging to
           `surface_name`
        """
        if surface_name == Surface.LV_ENDOCARDIAL:
            return self.surface_start_end[0, :]

        if surface_name == Surface.RV_SEPTUM:
            return self.surface_start_end[1, :]

        if surface_name == Surface.RV_FREEWALL:
            return self.surface_start_end[2, :]

        if surface_name == Surface.EPICARDIAL:
            return self.surface_start_end[3, :]

        if surface_name == Surface.MITRAL_VALVE:
            return self.surface_start_end[4, :]

        if surface_name == Surface.AORTA_VALVE:
            return self.surface_start_end[5, :]

        if surface_name == Surface.TRICUSPID_VALVE:
            return self.surface_start_end[6, :]

        if surface_name == Surface.PULMONARY_VALVE:
            return self.surface_start_end[7, :]

        return None

    def get_control_mesh_vertex_start_end_index(self, surface_name: ControlMesh) -> np.ndarray:
        """Return first and last vertex index for a given surface to use
        with `et_pos` array

        Parameters
        -----------

        `surface_name`  Surface name as defined in 'Surface' enumeration

        `Returns`
        ---------
        2x1 array with first and last vertices index belonging to
            surface_name
        """
        if surface_name == ControlMesh.LV_ENDOCARDIAL:
            return self.control_mesh_vertex_start_end[0, :]

        if surface_name == ControlMesh.RV_ENDOCARDIAL:
            return self.control_mesh_vertex_start_end[1, :]

        if surface_name == Surface.EPICARDIAL:
            return self.control_mesh_vertex_start_end[2, :]

        return None

    def get_control_mesh_faces(self, surface: ControlMesh) -> np.ndarray:
        """Get the faces definition for a surface triangulation"""
        surface_index = self.get_control_mesh_start_end_index(surface)
        return self.et_indices_control_mesh[surface_index[0]: surface_index[1] + 1, :]

    def get_control_mesh_start_end_index(self, surface_name: ControlMesh) -> np.ndarray:
        """Return first and last element index for a given surface, tu use
        with `et_indices_control_mesh` array
        Parameters
        ----------
        `surface_name` surface name as defined by `ControlMesh` enum

        Returns
        -------
        2x1 array containing first and last vertices index belonging to
           `surface_name`
        """
        if surface_name == ControlMesh.LV_ENDOCARDIAL:
            return self.surface_start_end[0, :]

        if surface_name == ControlMesh.RV_ENDOCARDIAL:
            return self.surface_start_end[1, :]

        if surface_name == ControlMesh.EPICARDIAL:
            return self.surface_start_end[2, :]

        return None

    def is_diffeomorphic(self, updated_control_mesh: np.ndarray, min_jacobian: float) -> bool:
        """This function checks the Jacobian value at Gauss point location
        (I am using 3x3x3 per element).

        Notes
        ------
        Returns 0 if one of the determinants is below a given
        threshold and 1 otherwise.
        It is recommended to use min_jacobian = 0.1 to make sure that there
        is no intersection/folding; a value of 0 can be used, but it might
        still give a positive jacobian
        if there are small intersections due to numerical approximation.

        Parameters
        -----------
        `new_control_mesh` control mesh we want to check

        Returns
        -------
            boolean value
        """
        # # Precompute inner products for all jac_* with each mesh component (vectorized for much faster processing)
        # j11 = self.jac_11 @ updated_control_mesh[:, 0]
        # j12 = self.jac_12 @ updated_control_mesh[:, 0]
        # j13 = self.jac_13 @ updated_control_mesh[:, 0]
        #
        # j21 = self.jac_11 @ updated_control_mesh[:, 1]
        # j22 = self.jac_12 @ updated_control_mesh[:, 1]
        # j23 = self.jac_13 @ updated_control_mesh[:, 1]
        #
        # j31 = self.jac_11 @ updated_control_mesh[:, 2]
        # j32 = self.jac_12 @ updated_control_mesh[:, 2]
        # j33 = self.jac_13 @ updated_control_mesh[:, 2]
        #
        # # Stack into a (N, 3, 3) array of Jacobians
        # jacobians = np.stack([
        #     np.stack([j11, j12, j13], axis=-1),
        #     np.stack([j21, j22, j23], axis=-1),
        #     np.stack([j31, j32, j33], axis=-1),
        # ], axis=-2)  # shape: (N, 3, 3)
        #
        # # Compute all determinants at once
        # determinants = np.linalg.det(jacobians)

        # Ensure openBLAS-friendly
        C = np.ascontiguousarray(updated_control_mesh, dtype=np.float64)

        # 3 matmuls instead of 9 matvecs
        # Each jni is (N, 3) and contains a column of the Jacobian per row:
        jn1 = self.jac_11 @ C
        jn2 = self.jac_12 @ C
        jn3 = self.jac_13 @ C

        # det(J) = (T1 x T2) · T3  (row-wise)
        # Einstein summation for faster processing
        cross = np.cross(jn1, jn2, axis=1)  # (N, 3)
        determinants = np.einsum('ij,ij->i', cross, jn3)  # (N,)

        return np.all(determinants >= min_jacobian)

    def update_pose_and_scale(self, dataset: GPDataSet) -> None:
        """A method that scale and translate the model to rigidly align
        with the guide points.

        Notes
        ------
        Parameters
        ------------
        `dataset` GPDataSet object with guide points
        Returns
        --------

        `scale_factor` scale factor between template and data points.
        """
        scale_factor = self.get_scaling(dataset)
        self.update_control_mesh(self.control_mesh * scale_factor)

        # The rotation is defined about the origin so we need to translate the model to the origin
        self.update_control_mesh(self.control_mesh - self.et_pos.mean(axis=0))
        rotation = self.get_rotation(dataset)
        self.update_control_mesh(np.array([np.dot(rotation, node) for node in self.control_mesh]))

        # Translate the model back to origin of the DataSet coordinate system
        translation = self.get_translation(dataset)

        self.update_control_mesh(self.control_mesh + translation)

    def get_scaling(self, dataset: GPDataSet) -> float:
        """Calculates a scaling factor for the model
        to match the guide points defined in dataset

        Parameters
        -----------
        `data_set` GPDataSet object

        Returns
        --------
        `scaleFactor` float
        """
        model_shape_index = [
            self.APEX_INDEX,
            self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],
            self.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1],
        ]
        model_shape = np.array(self.et_pos[model_shape_index, :])
        reference_shape = np.array([dataset.apex, dataset.mitral_centroid, dataset.tricuspid_centroid])
        mean_m = model_shape.mean(axis=0)
        mean_r = reference_shape.mean(axis=0)
        model_shape = model_shape - mean_m
        reference_shape = reference_shape - mean_r
        ss_model = (model_shape ** 2).sum()
        ss_reference = (reference_shape ** 2).sum()

        # centered Forbidius norm
        norm_model = np.sqrt(ss_model)
        reference_norm = np.sqrt(ss_reference)

        scale_factor = reference_norm / norm_model

        return scale_factor

    def get_translation(self, dataset: GPDataSet) -> np.ndarray:
        """Calculates a translation for (x, y, z)
        axis that aligns the model RV center with dataset RV center
        Parameters
        -----------
        `data_set` GPDataSet object

        Returns
        --------
          `translation` 3X1 array[float] with x, y and z translation
        """
        dataset_coordinates = [dataset.apex, dataset.mitral_centroid, dataset.tricuspid_centroid]
        model_point_indices = [self.APEX_INDEX,
                               self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],
                               self.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1]]
        model_coordinates = self.et_pos[model_point_indices, :]
        translation = np.mean(dataset_coordinates, axis=0) - np.mean(model_coordinates, axis=0)

        return translation

    def get_rotation(self, data_set: GPDataSet) -> np.ndarray:
        """Computes the rotation between model and data set,
        the rotation is given
        by considering the x-axis direction defined by the mitral valve centroid
        and apex the origin of the coordinates system is the mid point between
        the apex and mitral centroid

        Parameters
        ----------
        `data_set` GPDataSet object
        Returns
        --------
        `rotation` 3x3 rotation matrix
        """
        base = data_set.mitral_centroid
        base_model = self.et_pos[self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1], :]

        # computes data_set coordinates system
        x_axis = data_set.apex - base
        x_axis = x_axis / np.linalg.norm(x_axis)

        apex_position_model = self.et_pos[self.APEX_INDEX, :]

        x_axis_model = apex_position_model - base_model
        x_axis_model = x_axis_model / np.linalg.norm(x_axis_model)  # normalize

        # compute origin defined at 1/3 of the height of the model on the Ox
        # axis
        temp_original = 0.5 * (data_set.apex + base)
        temp_original_model = 0.5 * (apex_position_model + base_model)

        max_d = np.linalg.norm(0.5 * (data_set.apex - base))
        min_d = -np.linalg.norm(0.5 * (data_set.apex - base))

        max_d_model = np.linalg.norm(0.5 * (apex_position_model - base_model))
        min_d_model = -np.linalg.norm(0.5 * (apex_position_model - base_model))

        point_proj = data_set.points_coordinates[(data_set.contour_type == ContourType.SAX_LV_ENDOCARDIAL), :]

        point_proj = np.vstack(
            (point_proj, data_set.points_coordinates[(data_set.contour_type == ContourType.LAX_LV_ENDOCARDIAL), :])
        )

        assert len(point_proj) > 0, f"No LV contours found in get_rotation"

        temp_d = [np.dot(x_axis, p) for p in (point_proj - temp_original)]
        max_d = max(np.max(temp_d), max_d)
        min_d = min(np.min(temp_d), min_d)

        point_proj_model = self.et_pos[
                           self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[0]:
                           self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[1] + 1,
                           :, ]

        temp_d_model = [
            np.dot(x_axis_model, point_model)
            for point_model in (point_proj_model - temp_original_model)
        ]
        max_d_model = max(np.max(temp_d_model), max_d_model)
        min_d_model = min(np.min(temp_d_model), min_d_model)

        centroid = temp_original + min_d * x_axis + ((max_d - min_d) / 3.0) * x_axis
        centroid_model = (temp_original_model
                          + min_d_model * x_axis_model
                          + ((max_d_model - min_d_model) / 3.0) * x_axis_model
                          )

        # Compute Oy axis
        valid_index = ((data_set.contour_type == ContourType.SAX_RV_FREEWALL) +
                       (data_set.contour_type == ContourType.SAX_RV_SEPTUM) +
                       (data_set.contour_type == ContourType.LAX_RV_FREEWALL) +
                       (data_set.contour_type == ContourType.LAX_RV_SEPTUM))

        rv_endo_points = data_set.points_coordinates[valid_index, :]
        septal_start = self.get_surface_vertex_start_end_index(Surface.RV_SEPTUM)[0]
        rv_fw_start = self.get_surface_vertex_start_end_index(Surface.RV_FREEWALL)[1]
        rv_endo_points_model = self.et_pos[septal_start:rv_fw_start + 1, :]

        rv_centroid = rv_endo_points.mean(axis=0)
        rv_centroid_model = rv_endo_points_model.mean(axis=0)

        scale = np.dot(x_axis, rv_centroid) - np.dot(x_axis, centroid) / np.dot(x_axis, x_axis)
        scale_model = (np.dot(x_axis_model, rv_centroid_model) -
                       np.dot(x_axis_model, centroid_model) /
                       np.dot(x_axis_model, x_axis_model))
        rv_proj = centroid + scale * x_axis
        rv_proj_model = centroid_model + scale_model * x_axis_model

        y_axis = rv_centroid - rv_proj
        y_axis_model = rv_centroid_model - rv_proj_model

        y_axis /= np.linalg.norm(y_axis)
        y_axis_model /= np.linalg.norm(y_axis_model)

        z_axis = np.cross(x_axis, y_axis)
        z_axis_model = np.cross(x_axis_model, y_axis_model)

        # normalization
        z_axis /= np.linalg.norm(z_axis)
        z_axis_model /= np.linalg.norm(z_axis_model)

        # Find translation and rotation between the two coordinates systems
        # The easiest way to solve it (in my opinion) is by using a
        # Singular Value Decomposition as reported by Markley (1988):
        #    1. Obtain a matrix B as follows:
        #        B=∑ni=1aiwiviTB=∑i=wiviT
        #    2. Find the SVD of BB
        #        B=USVT
        #    3. The rotation matrix is:
        #        R=UMVT, where M=diag([11det(U)det(V)])

        # Step 1
        b = (np.outer(x_axis, x_axis_model) + np.outer(y_axis, y_axis_model) + np.outer(z_axis, z_axis_model))

        # Step 2
        [u, _, v_t] = np.linalg.svd(b)

        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(u) * np.linalg.det(v_t)]])
        rotation = np.dot(u, np.dot(m, v_t))

        return rotation

    def update_control_mesh(self, new_control_mesh: np.ndarray) -> None:
        """Update control mesh
        Parameters
        ----------
        new_control_mesh: (388,3) array of new control node positions
        """
        self.control_mesh = new_control_mesh
        self.et_pos = self.matrix @ self.control_mesh

    def get_intersection_with_plane(self, po: np.ndarray, no: np.ndarray, surface_to_use: Surface = None) -> np.ndarray:
        """Calculate intersection points between a plane with the
        biventricular model (LV_ENDOCARDIAL only)

        Parameters
        ----------
        `po` (3,1) array[float] a point of the plane
        `no` (3,1) array[float normal to the plane

        Returns
        -------

        `f_idx` (N,3) array[float] are indices of the surface nodes indicating
        intersecting the plane"""

        # Adjust po & no into a column vector
        no = no / np.linalg.norm(no)

        f_idx = []

        if surface_to_use is None:
            surface_to_use = [Surface.LV_ENDOCARDIAL]
        for surface in surface_to_use:  # We just want intersection LV_ENDOCARDIAL,
            # RVS. RVFW, epi
            # Get the faces
            faces = self.get_surface_faces(surface)

            # --- find triangles that intersect with plane: (po,no)
            # calculate sign distance of each vertices

            # set the origin of the model at po
            centered_vertex = self.et_pos - [list(po)] * len(self.et_pos)
            # projection on the normal
            dist = np.dot(no, centered_vertex.T)

            signed_distance = np.sign(dist[faces])
            # search for triangles having the vertex on the both sides of the
            # slice plane => intersecting with the slice plane
            valid_index = [
                np.any(signed_distance[i] > 0) and np.any(signed_distance[i] < 0)
                for i in range(len(signed_distance))
            ]
            intersecting_face_idx = np.where(valid_index)[0]

            if len(intersecting_face_idx) < 0:
                return np.empty((0, 3))

            # Find the intersection lines - find segments for each intersected
            # triangles that intersects the plane
            # see http://softsurfer.com/Archive/algorithm_0104/algorithm_0104B.htm

            # pivot points

            i_pos = [x for x in intersecting_face_idx if np.sum(signed_distance[x] > 0) == 1]  # all
            # triangles with one vertex on the positive part
            i_neg = [x for x in intersecting_face_idx if np.sum(signed_distance[x] < 0) == 1]  # all
            # triangles with one vertex on the negative part
            p1 = []
            u = []

            for face_index in i_pos:  # triangles where only one
                # point
                # on positive side
                # pivot points
                pivot_point_mask = signed_distance[face_index, :] > 0
                res = centered_vertex[faces[face_index, pivot_point_mask], :][0]
                p1.append(list(res))
                # u vectors
                u = u + list(
                    np.subtract(centered_vertex[faces[face_index, np.invert(pivot_point_mask)], :], [list(res)] * 2, ))

            for face_index in i_neg:  # triangles where only one
                # point on negative side
                # pivot points
                pivot_point_mask = signed_distance[face_index, :] < 0
                res = centered_vertex[faces[face_index, pivot_point_mask], :][
                    0]  # select the vertex on the negative side
                p1.append(res)
                # u vectors
                u = u + list(
                    np.subtract(centered_vertex[faces[face_index, np.invert(pivot_point_mask)], :], [list(res)] * 2, ))

            # calculate the intersection point on each triangle side
            u = np.asarray(u).T
            p1 = np.asarray(p1).T
            if len(p1) == 0:
                continue
            mat = np.zeros((3, 2 * p1.shape[1]))
            mat[0:3, 0::2] = p1
            mat[0:3, 1::2] = p1
            p1 = mat

            si = -np.dot(no.T, p1) / (np.dot(no.T, u))
            factor_u = np.array([list(si)] * 3)
            pts = np.add(p1, np.multiply(factor_u, u)).T
            # add vertices that are on the surface
            pon = centered_vertex[faces[signed_distance == 0], :]
            pts = np.vstack((pts, pon))
            # #change points to the original position
            f_idx = f_idx + list(pts + [list(po)] * len(pts))

        return f_idx

    def get_intersection_with_dicom_image(self, slice: Slice, surface: Surface = None) -> np.ndarray:
        """Get the intersection contour points between the biventricular
        model with a DICOM image

        Example
        -------
            obj.get_intersection_with_dicom_image(slice, Surface.RV_SEPTUM)

        Parameters
        ----------

        `slice` Slice obj with the dicom information

        `surface` Surface enum, model surface to be intersected

        Returns
        -------

        `P` (n,3) array[float] intersecting points
        """
        image_position = np.asarray(slice.position, dtype=float)
        image_orientation = np.asarray(slice.orientation, dtype=float)

        # get image position and the image vectors
        v1 = np.asarray(image_orientation[0:3], dtype=float)
        v2 = np.asarray(image_orientation[3:6], dtype=float)
        v3 = np.cross(v1, v2)

        # get intersection points
        p = self.get_intersection_with_plane(image_position, v3, surface_to_use=surface)

        return p

    def compute_data_xi(self, weight: float, data: GPDataSet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Projects the N guide points to the closest point of the model
        surface.

        If 2 data points are projected  onto the same surface point,
        the closest one is kept. Surface type is matched with the Contour
        type using 'SURFACE_CONTOUR_MAP' variable (surface_enum)

        Parameters
        -----------
        `weight` float with weights given to the data points
        `data` GPDataSet object with guide points

        Returns
        --------
        `data_points_index` (`N`,1) array[int] with index of the closest
        control point to the each node
        `w` (`N`,`N`) matrix[float] of  weights of the data points
        `distance_d_prior` (`N`,1) matrix[float] distances to the closest points
        `psi_matrix` basis function matrix (`N`,`NUM_NODES`)

        """
        data_points = np.asarray(data.points_coordinates)
        data_contour_type = np.asarray(data.contour_type)
        data_weights = np.array(data.weights)
        psi_matrix = []
        w = []
        distance_d_prior = []
        index = []
        data_points_index = []
        index_unique = []  # add by LDT 3/11/2021

        basis_matrix = self.basis_matrix

        # add by A. Mira : a more compressed way of initializing the cKDTree

        for surface in Surface:
            # Trees initialization
            surface_index = self.get_surface_vertex_start_end_index(surface)
            tree_points = self.et_pos[surface_index[0]: surface_index[1] + 1, :]
            if len(tree_points) == 0:
                continue
            surface_tree = cKDTree(tree_points)

            # loop over contours is faster, for the same contours we are using
            # the same tree, therefore the query operation can be done for all
            # points of the same contour: A. Mira 02/2020
            for contour in SURFACE_CONTOUR_MAP[surface.value]:
                contour_points_index = np.where(data_contour_type == contour)[0]
                contour_points = data_points[contour_points_index]
                weights_gp = data_weights[contour_points_index]

                if len(contour_points) == 0:
                    continue

                if surface.value < 4:  # these are the surfaces
                    distance, vertex_index = surface_tree.query(contour_points, k=1, p=2)
                    index_closest = [x + surface_index[0] for x in vertex_index]
                    # add by LDT (3/11/2021): perform preliminary operations for vertex points that are not in index
                    # instead of doing them in the 'else' below. This makes the for loop below faster.
                    unique_index_closest = list(OrderedDict.fromkeys(index_closest))
                    # creates a list of elements that are unique in index_closest

                    # create a dictionary = {'unique element': its list index}
                    dict_unique = dict(zip(unique_index_closest, range(0, len(unique_index_closest))))
                    vertex = list(dict_unique.keys())  # list of all the dictionary keys
                    # intersection between the array index_unique and the unique points in index_closest
                    common_elm = list(set(index_unique) & set(vertex))

                    def filter_new(full_list, excludes):
                        """
                        eliminates the items in 'exclude' out of the full_list
                        """
                        s = set(excludes)
                        return (x for x in full_list if x not in s)

                    # togli gli elementi comuni
                    # stores the new vertices that are NOT in already in the index_unique list
                    index_unique.append(list(filter_new(vertex, common_elm)))
                    index_unique = [item for sublist in index_unique for item in
                                    (sublist if isinstance(sublist, list) else [sublist])]

                    items_as_dict = dict(zip(index_unique, range(0, len(index_unique))))
                    # builds a dictionary = {vertices: indexes}

                    for i_idx, vertex_index in enumerate(index_closest):
                        if (len(set([vertex_index]).intersection(index)) == 0):  # changed by LDT (3/11/2021): faster
                            index.append(int(vertex_index))
                            data_points_index.append(contour_points_index[i_idx])
                            psi_matrix.append(basis_matrix[int(vertex_index), :])
                            w.append(weight * weights_gp[i_idx])
                            distance_d_prior.append(distance[i_idx])
                        else:
                            old_idx = items_as_dict[vertex_index]  # changed by LDT (3/11/2021)
                            distance_old = distance_d_prior[old_idx]
                            if distance[i_idx] < distance_old:
                                distance_d_prior[old_idx] = distance[i_idx]
                                data_points_index[old_idx] = contour_points_index[i_idx]
                                w[old_idx] = weight * weights_gp[i_idx]

                else:
                    # If it is a valve, we virtually translate the data points
                    # (only the ones belonging to the same surface) so their centroid
                    # matches the template's valve centroid.
                    # So instead of calculating the minimum distance between the point
                    # p and the model points pm, we calculate the minimum distance between
                    # the point p+t and pm,
                    # where t is the translation needed to match both centroids
                    # This is to make sure that the data points are going to be
                    # projected all around the valve and not only on one side.
                    if surface.value < 8:  # these are the landmarks without apex
                        # and rv inserts
                        centroid_valve = self.et_pos[surface_index[1]]
                        centroid_gp_valve = contour_points.mean(axis=0)
                        translation_gp_model = centroid_valve - centroid_gp_valve
                        translated_points = np.add(contour_points, translation_gp_model)
                    else:  # rv_inserts  and apex don't
                        # need to be translated
                        translated_points = contour_points

                    if contour in [
                        ContourType.MITRAL_PHANTOM,
                        ContourType.PULMONARY_PHANTOM,
                        ContourType.AORTA_PHANTOM,
                        ContourType.TRICUSPID_PHANTOM,
                    ]:
                        surface_tree = cKDTree(translated_points)
                        tree_points = tree_points[:-1]
                        distance, vertex_index = surface_tree.query(tree_points, k=1, p=2)
                        index_closest = [x + surface_index[0] for x in range(len(tree_points))]
                        weights_gp = [weights_gp[x] for x in vertex_index]

                        contour_points_index = [contour_points_index[x] for x in vertex_index]
                    else:
                        distance, vertex_index = surface_tree.query(translated_points, k=1, p=2)
                        index_closest = []
                        for x in vertex_index:
                            if (x + surface_index[0]) != surface_index[1]:
                                index_closest.append(x + surface_index[0])
                            else:
                                index_closest.append(x + surface_index[0] - 1)

                    index = index + index_closest
                    psi_matrix = psi_matrix + list(basis_matrix[index_closest, :])

                    w = w + [(weight * x) for x in weights_gp]
                    distance_d_prior = distance_d_prior + list(distance)
                    data_points_index = data_points_index + list(contour_points_index)

        return [np.asarray(data_points_index), np.asarray(w), np.asarray(distance_d_prior), np.asarray(psi_matrix)]

    def compute_data_xi_fast(self, weight: float, data: GPDataSet):
        """
        Semantics-preserving, faster version of compute_data_xi:
          - Rebuilds one KDTree per surface each call (since et_pos moves).
          - SURFACES (<4): query all points mapped to that surface (across contours),
            then keep the closest guide point per vertex (dedup).
          - VALVES/LANDMARKS (>=4): keep original behavior:
              * for PHANTOM contours: build KDTree on guide points and query vertices (minus last);
                one guide point per vertex; no dedup across contours.
              * for non-phantom: query surface KDTree on (optionally translated) guide points;
                clip last vertex to s1-1; no dedup across contours.
        Returns:
          data_points_index: (N_kept,) indices into data.points_coordinates for chosen guide points
          w:                 (N_kept,)   weights per kept correspondence
          distance_d_prior:  (N_kept,)   distances at the chosen correspondence
          psi_matrix:        (N_kept, NUM_NODES) basis rows at chosen vertices
        """
        data_points = np.asarray(data.points_coordinates, dtype=np.float64)
        data_contour_type = np.asarray(data.contour_type)
        data_weights = np.asarray(data.weights, dtype=np.float64)
        basis_matrix = np.asarray(self.basis_matrix, dtype=np.float64)

        out_vertex_ids = []
        out_point_idx = []
        out_weights = []
        out_dist = []

        # Rebuild a KD-tree per surface (et_pos shifts every accepted step)
        for surface in Surface:
            s0, s1 = self.get_surface_vertex_start_end_index(surface)
            if s1 < s0:
                continue
            surf_pts = self.et_pos[s0: s1 + 1, :]
            if surf_pts.size == 0:
                continue

            # Indices of all guide points that map to this surface via SURFACE_CONTOUR_MAP
            surface_contours = SURFACE_CONTOUR_MAP[surface.value]
            idx_all = np.nonzero(np.isin(data_contour_type, surface_contours))[0]
            if idx_all.size == 0:
                continue

            # Build once, use it in the non-phantom branches
            surf_tree = cKDTree(surf_pts)

            if surface.value < 4:
                # ---- SURFACE branch: dedup per vertex (closest distance wins) ----
                pts = data_points[idx_all, :]
                wts = (weight * data_weights[idx_all]).astype(np.float64)

                dist, vi_local = surf_tree.query(pts, k=1, p=2)  # (m,), (m,)
                gi = vi_local + s0  # global vertex ids

                # Strict min-distance per vertex, tie breaks by earliest occurrence
                # Sort by (vertex, distance, original_order)
                order = np.arange(gi.size, dtype=np.int64)
                sort_idx = np.lexsort((order, dist, gi))
                gi_s = gi[sort_idx]
                # first element of each new vertex group
                keep_sorted = np.empty_like(gi_s, dtype=bool)
                keep_sorted[0] = True
                keep_sorted[1:] = gi_s[1:] != gi_s[:-1]
                keep_idx = sort_idx[keep_sorted]

                out_vertex_ids.append(gi[keep_idx])
                out_point_idx.append(idx_all[keep_idx])
                out_weights.append(wts[keep_idx])
                out_dist.append(dist[keep_idx])

            else:
                # ---- VALVE / LANDMARK branch: preserve original behavior, no dedup ----
                # Process each contour independently (as in your code)
                for contour in SURFACE_CONTOUR_MAP[surface.value]:
                    idx_c = np.nonzero(data_contour_type == contour)[0]
                    if idx_c.size == 0:
                        continue

                    pts = data_points[idx_c, :].copy()
                    wts = (weight * data_weights[idx_c]).astype(np.float64)

                    # Translate for 4..7 (valves w/o apex/inserts)
                    if 4 <= surface.value < 8:
                        centroid_valve = self.et_pos[s1]
                        centroid_gp = pts.mean(axis=0)
                        pts += (centroid_valve - centroid_gp)[None, :]

                    is_phantom = contour in (
                        ContourType.MITRAL_PHANTOM,
                        ContourType.PULMONARY_PHANTOM,
                        ContourType.AORTA_PHANTOM,
                        ContourType.TRICUSPID_PHANTOM,
                    )

                    if is_phantom:
                        # Inverted query: build KDTree on (translated) guide points, query surface vertices (except last)
                        nvert = surf_pts.shape[0]
                        if nvert <= 1:
                            continue  # matches original, nothing to add
                        query_tp = surf_pts[:-1, :]  # drop last vertex
                        global_vid = np.arange(s0, s0 + nvert - 1, dtype=np.int64)

                        guide_tree = cKDTree(pts)
                        dist, vi_guide = guide_tree.query(query_tp, k=1, p=2)  # pick a guide point per vertex
                        gp_sel = idx_c[vi_guide]  # global point indices
                        w_sel = (weight * data_weights[gp_sel]).astype(np.float64)

                        out_vertex_ids.append(global_vid)
                        out_point_idx.append(gp_sel)
                        out_weights.append(w_sel)
                        out_dist.append(dist.astype(np.float64))

                    else:
                        # Normal landmark path: query surface KDTree for each (translated) guide point
                        dist, vi_local = surf_tree.query(pts, k=1, p=2)
                        gi = vi_local + s0
                        # if equals s1, replace with s1-1 (guard single-vertex case)
                        if s1 > s0:
                            gi = np.where(gi == s1, s1 - 1, gi).astype(np.int64)
                        else:
                            gi = gi.astype(np.int64)

                        out_vertex_ids.append(gi)
                        out_point_idx.append(idx_c)
                        out_weights.append(wts)
                        out_dist.append(dist.astype(np.float64))

        # Concatenate in lock-step
        if out_vertex_ids:
            vertex_ids = np.concatenate(out_vertex_ids)
            point_ids = np.concatenate(out_point_idx)
            weights_kept = np.concatenate(out_weights).astype(np.float64, copy=False)
            dist_kept = np.concatenate(out_dist).astype(np.float64, copy=False)
        else:
            vertex_ids = np.empty((0,), dtype=np.int64)
            point_ids = np.empty((0,), dtype=np.int64)
            weights_kept = np.empty((0,), dtype=np.float64)
            dist_kept = np.empty((0,), dtype=np.float64)

        # Sanity: all have same length
        N = vertex_ids.shape[0]
        assert point_ids.shape[0] == N and weights_kept.shape[0] == N and dist_kept.shape[0] == N, \
            f"Length mismatch: verts={N}, pts={point_ids.shape[0]}, w={weights_kept.shape[0]}, d={dist_kept.shape[0]}"

        # Basis rows for kept vertices
        psi = basis_matrix[vertex_ids, :]

        # Return (data_points_index, w, distance_d_prior, psi_matrix)
        return point_ids.astype(np.int64), weights_kept, dist_kept, psi

    def copy(self):
        """Lightweight copy: share constants; copy only arrays that mutate."""
        new = self.__class__.__new__(self.__class__)  # bypass __init__

        # 1) copy small scalars/flags
        for name in self._INPUT_MUTABLES:
            if hasattr(self, name):
                val = getattr(self, name)
                # sets/dicts should be shallow-copied if present
                if isinstance(val, (set, dict, list)):
                    val = val.copy()
                setattr(new, name, val)

        # 2) share big immutable arrays (no copy)
        for name in self._CONST_ARRAYS:
            if hasattr(self, name):
                setattr(new, name, getattr(self, name))

        # 3) copy mutable arrays
        if hasattr(self, "control_mesh"):
            new.control_mesh = self.control_mesh.copy()
        else:
            new.control_mesh = None

        # Recompute or copy derived state
        if hasattr(self, "matrix") and new.control_mesh is not None:
            # et_pos is derived from matrix @ control_mesh; compute to stay correct
            new.et_pos = self.matrix @ new.control_mesh
        elif hasattr(self, "et_pos"):
            # fallback: copy if present
            new.et_pos = self.et_pos.copy()

        return new

    # Optional: make Python's copy/deepcopy use the fast path
    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        # still use lightweight copy; only duplicates the truly mutable bits
        return self.copy()

    # Optional: shrink pickles (great for multiprocessing)
    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop/recompute-ables to reduce pickle size
        state.pop("et_pos", None)
        # If collision_detection/sets are large and recomputable, consider dropping here too
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore derived arrays cheaply
        if getattr(self, "matrix", None) is not None and getattr(self, "control_mesh", None) is not None:
            self.et_pos = self.matrix @ self.control_mesh
