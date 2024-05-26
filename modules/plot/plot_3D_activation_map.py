import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors

# class CustomColormap:
#     def __init__(self):
#         self.cmap_names = ['blue2red']

#     def __getitem__(self, item):
#         return self.__dict__[item]    
    
#     def get_colormap(self, cmap):
#         if cmap == 'blue2red':
#             colors = ['#0000FF', '#FF0000']  # Deep blue to deep red
#             cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors)
#             return cmap

def get_cmap(cmap):
    if cmap == 'blue2red':
        # colors = ['#0000FF', '#FF0000']
        # colors = ['#373EFF', '#FF4949']
        # colors = ['#86A7FF', '#FF7575']        
        colors = ['#0000FF','#86A7FF', '#FF7575', '#FF0000']
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors)
    elif cmap == 'green2yellow2red':
        colors = ['#00FF00', '#FFFF00', '#FF0000']
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors)
    else:
        cmap = plt.get_cmap(cmap)
    return cmap

def extract_labeled_faces(mat, TOSs=None, layerid = 3, take_18_only=True):
    faces = mat['AnalysisInfo']['fv']['faces'][mat['AnalysisInfo']['fv']['layerid'] == layerid]
    n_faces = faces.shape[0]
    if take_18_only:
        sector_width = n_faces // 18
        labeled_faces_indices = np.arange(sector_width //2, n_faces, sector_width)
        labeled_faces = faces[labeled_faces_indices]
    else:
        labeled_faces = faces
    labeled_faces_center_vertices = np.mean(mat['AnalysisInfo']['fv']['vertices'][labeled_faces], axis=1)
    # vertices = mat['AnalysisInfo']['fv']['vertices'][mat['AnalysisInfo']['fv']['layerid'] == layerid]
    return labeled_faces_center_vertices

def map_values_to_rgb(values, cmap_name, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    cmap = get_cmap(cmap_name)
    # colors = ['#0000FF', '#FF0000']  # Deep blue to deep red
    # cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors)
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False)
    rgba_colors = cmap(norm(values))
    # rgba_colors = cmap(values/values.max())
    # print(rgba_colors.shape)
    rgb_colors = np.delete(rgba_colors, 3, 1)
    # rgb_colors = np.delete(rgba_colors, 3, 2)
    return rgb_colors

def rotate_around_x(points, angle_deg):
    """Rotate points around the X axis by a given angle in degrees."""
    theta = np.radians(angle_deg)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return np.dot(points, rotation_matrix.T)

def rotate_around_y(points, angle_deg):
    """Rotate points around the Y axis by a given angle in degrees."""
    theta = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return np.dot(points, rotation_matrix.T)

def rotate_around_z(points, angle_deg):
    """Rotate points around the Z axis by a given angle in degrees."""
    theta = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.dot(points, rotation_matrix.T)


import numpy as np
from stl import mesh
from scipy.interpolate import griddata


def align_vertices_with_mesh(vertices_coords, mesh_data, z_scale_factor=1.0, xy_scale_factor=1.0):
    """
    Align vertices_coords with the mesh_data by rescaling and translating the coordinates.

    Parameters:
    - vertices_coords (Nx3 numpy array): XYZ coordinates of the vertices.
    - mesh_data (numpy-stl mesh): Mesh data to align with.
    - z_scale_factor (float, optional): The factor by which to rescale the Z axis. Default is 1.0.
    - xy_scale_factor (float, optional): The factor by which to rescale the XY plane. Default is 1.0.

    Returns:
    - aligned_vertices (Nx3 numpy array): Aligned XYZ coordinates of vertices_coords.
    """
    # 1. Rescale the Z-axis
    z_min, z_max = vertices_coords[:, 2].min(), vertices_coords[:, 2].max()
    mesh_z_min, mesh_z_max = mesh_data.vectors[:, :, 2].min(), mesh_data.vectors[:, :, 2].max()

    # Compute the center of Z range and the half span after rescaling
    mesh_z_center = (mesh_z_max + mesh_z_min) / 2
    half_span = z_scale_factor * (mesh_z_max - mesh_z_min) / 2

    # New mesh z bounds after rescaling around the center
    new_mesh_z_min = mesh_z_center - half_span
    new_mesh_z_max = mesh_z_center + half_span

    scale_factor_z = (new_mesh_z_max - new_mesh_z_min) / (z_max - z_min)
    vertices_coords[:, 2] = (vertices_coords[:, 2] - z_min) * scale_factor_z + new_mesh_z_min

    # 2. Align centers in the XY plane
    center_vertices = np.mean(vertices_coords[:, :2], axis=0)
    center_mesh = np.mean(mesh_data.vectors[:, :, :2], axis=(0, 1))

    translation = center_mesh - center_vertices
    vertices_coords[:, :2] += translation

    # 3. Rescale in the XY plane
    bbox_vertices = np.array([vertices_coords[:, :2].min(axis=0), vertices_coords[:, :2].max(axis=0)])
    bbox_mesh = np.array([mesh_data.vectors[:, :, :2].min(axis=(0, 1)), mesh_data.vectors[:, :, :2].max(axis=(0, 1))])

    scale_factor_xy = xy_scale_factor * (bbox_mesh[1] - bbox_mesh[0]) / (bbox_vertices[1] - bbox_vertices[0])
    vertices_coords[:, :2] = (vertices_coords[:, :2] - bbox_vertices[0]) * scale_factor_xy + bbox_mesh[0]

    return vertices_coords


# Create and save an .obj file with accompanying .mtl file for color information
def save_colored_obj(mesh_data, colors, obj_filename, mtl_filename):
    """
    Save a mesh as an .obj file with accompanying .mtl file for color information.

    Parameters:
    - mesh_data (numpy-stl mesh): Mesh data to save.
    - colors (Nx3 numpy array): RGB colors for each face of the mesh.
    - obj_filename (str): The name of the .obj file to save.
    - mtl_filename (str): The name of the .mtl file to save.

    Returns:
    None. This function saves the mesh data and color information to two files.
    """
    with open(obj_filename, 'w') as obj_file, open(mtl_filename, 'w') as mtl_file:
        obj_file.write(f"mtllib {mtl_filename.split('/')[-1]}\n")

        # Write vertices to the .obj file
        for vertex in mesh_data.vectors.reshape(-1, 3):
            obj_file.write(f"v {' '.join(map(str, vertex))}\n")

        # Write vertex colors as materials to the .mtl file and use them in the .obj file
        for i, face in enumerate(mesh_data.vectors):
            color = colors[i]
            material_name = f"material_{i}"
            mtl_file.write(f"newmtl {material_name}\n")
            mtl_file.write(f"Kd {' '.join(map(str, color/255.0))}\n")
            obj_file.write(f"usemtl {material_name}\n")
            obj_file.write(f"f {3*i+1} {3*i+2} {3*i+3}\n")


import numpy as np
from stl import mesh

def face_centers_from_mesh(mesh_data):
    return np.mean(mesh_data.vectors, axis=1)


from scipy.spatial import ConvexHull, Delaunay
def rescale_vertices_to_include(aligned_vertices, face_centers, initial_scale=1.01, step_size=0.01):
    """
    Rescale aligned_vertices so that all points in face_centers are inside its convex hull.

    Parameters:
    - aligned_vertices (Nx3 numpy array): XYZ coordinates of the vertices.
    - face_centers (Mx3 numpy array): XYZ coordinates of another set of vertices.

    Returns:
    - rescaled_vertices (Nx3 numpy array): Rescaled XYZ coordinates of aligned_vertices.
    """

    centroid = np.mean(aligned_vertices, axis=0)
    scale_factor = initial_scale

    while True:
        # Scale the vertices away from the centroid
        rescaled_vertices = centroid + (aligned_vertices - centroid) * scale_factor

        # Check if all points in face_centers are inside the convex hull of rescaled vertices
        hull = Delaunay(rescaled_vertices)

        if all(hull.find_simplex(face_centers) >= 0):
            return rescaled_vertices

        scale_factor += step_size


from stl import mesh
import scipy.io as sio
# from plot_3D_activation_map_utils import extract_labeled_faces, map_values_to_rgb
# from plot_3D_activation_map_utils import align_vertices_with_mesh
# from plot_3D_activation_map_utils import face_centers_from_mesh
from scipy.interpolate import griddata
loadmat = lambda f: sio.loadmat(f, struct_as_record=False, squeeze_me=True, simplify_cells=True)
def build_3D_activation_map_single(slice_data: list, mesh_file_name=None, plot_TOS_key='TOS_pred', cmap='seismic', vmin=17, vmax=85):
    """
    Builds a 3D activation map for a single patient.

    Parameters:
    slice_data (list): A list of dictionaries containing slice data for a patient. Each dictionary should contain 'DENSE_slice_mat_filename' and 'DENSE_slice_location' keys, among others.
    mesh_file_name (str, optional): The name of the file containing the mesh data. If None, the mesh data will be loaded from the .mat files specified in slice_data. Default is None.
    plot_TOS_key (str, optional): The key to use to retrieve the TOS values from the dictionaries in slice_data. Default is 'TOS_pred'.
    cmap (str, optional): The name of the colormap to use for the activation map. Default is 'seismic'.

    Returns:
    None. This function operates in-place on the input data and does not return a value.

    Raises:
    ValueError: If no corresponding slice can be found for a given slice location, or if more than one corresponding slice is found.
    """
    # load the .mat files to get the original faces
    
    trial_patient_full_mat_files = [slice_data['DENSE_slice_mat_filename'] for slice_data in slice_data]
    # print(f"Number of full mat files: {len(trial_patient_full_mat_files)}")

    # load the full mat files
    trial_patient_full_mats = []
    for mat_file in trial_patient_full_mat_files:
        trial_patient_full_mats.append(loadmat(mat_file))

    # sort the full mat files by slice location in descending order
    trial_patient_full_mats_spatial_locations = [float(mat['SequenceInfo'][0,0].SliceLocation) for mat in trial_patient_full_mats]
    trial_patient_full_mats_sorted = [mat for _, mat in sorted(zip(trial_patient_full_mats_spatial_locations, trial_patient_full_mats), reverse=False)]

    # Get the TOS coordinates and values
    
    for slice_idx, mat in enumerate(trial_patient_full_mats_sorted):
        print(mat['SequenceInfo'][0,0].SliceLocation, mat['SequenceInfo'][0,0].SliceThickness)
        curr_pred_slices = [d for d in slice_data if abs(float(d['DENSE_slice_location']) - float(mat['SequenceInfo'][0,0].SliceLocation))<1]
        if len(curr_pred_slices) == 0:
            print(f"Cannot find the corresponding slice for {mat['SequenceInfo'][0,0].SliceLocation}")
            continue
        elif len(curr_pred_slices) > 1:
            print(f"Found more than one corresponding slice for {mat['SequenceInfo'][0,0].SliceLocation}")
            continue
        else:
            curr_pred_slice = curr_pred_slices[0]
        
        curr_slice_face_centers_2d = extract_labeled_faces(mat, take_18_only=False)
        curr_slice_face_centers_3d = np.zeros((curr_slice_face_centers_2d.shape[0], 3))
        curr_slice_face_centers_3d[:,0:2] = curr_slice_face_centers_2d
        curr_slice_face_centers_3d[:,2] = slice_idx

        if slice_idx == 0:
            face_centers_3d = curr_slice_face_centers_3d
        else:
            face_centers_3d = np.concatenate((face_centers_3d, curr_slice_face_centers_3d), axis=0)

        curr_slice_TOS = curr_pred_slice[plot_TOS_key]        
        # print(curr_slice_TOS.min())
        curr_slice_TOS[curr_slice_TOS < 17] = 17
        curr_slice_TOS_colored = map_values_to_rgb(curr_slice_TOS, cmap, vmin=vmin, vmax=vmax)
        # curr_slice_TOS_colored = map_values_to_rgb(curr_slice_TOS, 'seismic')
        if slice_idx == 0:
            TOS_2d = curr_slice_TOS.reshape((1, curr_slice_TOS.shape[0]))
            TOS_colored = curr_slice_TOS_colored
        else:
            TOS_2d = np.concatenate((TOS_2d, curr_slice_TOS.reshape((1, curr_slice_TOS.shape[0]))), axis=0)
            TOS_colored = np.concatenate((TOS_colored, curr_slice_TOS_colored), axis=0)
    
    # Load mesh using numpy-stl
    if mesh_file_name is None:
        mesh_file_name = '/p/mmcardiac/Jerry/code/DENSE-guided-Cine-Registration/paper-writing/2023-11-02-build-3D-activation-map/set01ct11_inverted_6235_decimated-0.5.stl'
    
    mesh_data = mesh.Mesh.from_file(mesh_file_name)
    # Move the center of the object to the origin
    object_center = np.mean(mesh_data.vectors, axis=(0, 1))
    mesh_data.vectors -= object_center
    vertices_coords = face_centers_3d
    vertices_values = TOS_colored

    
    aligned_vertices = align_vertices_with_mesh(vertices_coords, mesh_data, z_scale_factor=0.8, xy_scale_factor=1.0)

    # Interpolate values for mesh face centers
    face_centers = np.mean(mesh_data.vectors, axis=1)
    rescaled_vertices = rescale_vertices_to_include(aligned_vertices, face_centers)
    aligned_vertices = rescaled_vertices

    # Interpolate values for mesh face centers
    # interpolation_method = 'linear'
    interpolation_method = 'cubic'
    values_interpolated = griddata(aligned_vertices, vertices_values, face_centers, method=interpolation_method)

    # Check for nans and fill them using nearest neighbor method
    nan_indices = np.isnan(values_interpolated).any(axis=1)
    print(f'Number of nans: {nan_indices.sum()}')
    
    values_interpolated[nan_indices] = griddata(aligned_vertices, vertices_values, face_centers[nan_indices], method='nearest')
    # Ensure interpolated colors are in the correct format (uint8)
    colors_uint8 = (values_interpolated * 255).astype(np.uint8)

    
    face_centers = face_centers_from_mesh(mesh_data)
    face_colors = values_interpolated

    return face_centers, face_colors


def build_3D_activation_map_multiple(slice_data, mesh_file_names=None, target_subject_ids=None, plot_TOS_key='TOS_pred', cmap='seismic', vmin=17, vmax=85):
    """
    Builds 3D activation maps for multiple patients.

    Parameters:
    slice_data (list): A list of dictionaries containing slice data for each patient. Each dictionary should contain 'subject_id' and 'augmented' keys, among others.
    mesh_file_names (list, optional): A list of file names containing the mesh data for each patient. If None, a default mesh file is used for all patients. Default is None.
    target_subject_ids (list, optional): A list of subject IDs for which to build the activation maps. If None, activation maps are built for all unique patients in slice_data. Default is None.

    Returns:
    patients_face_data (list): A list of dictionaries, each containing the 'subject_id', 'face_centers', and 'face_colors' for a patient.

    Raises:
    ValueError: If the number of mesh files does not equal the number of unique patients.
    """
    original_data = [d for d in slice_data if not d["augmented"]]
    unique_patients_names = list(set([d['subject_id'] for d in original_data]))

    if target_subject_ids is None:
        target_subject_ids = unique_patients_names

    target_data = [d for d in original_data if d['subject_id'] in target_subject_ids]

    if mesh_file_names is None:
        mesh_file_names = ['/p/mmcardiac/Jerry/code/DENSE-guided-Cine-Registration/paper-writing/2023-11-02-build-3D-activation-map/set01ct11_inverted_6235_decimated-0.5.stl'] * len(unique_patients_names)
    elif type(mesh_file_names) == str:
        mesh_file_names = [mesh_file_names] * len(unique_patients_names)
    elif len(mesh_file_names) != len(target_subject_ids):
        raise ValueError("The number of mesh files must equal the number of unique patients")

    patients_face_data = []
    for patient_idx, patient_name in enumerate(target_subject_ids):
        print(f"Building 3D activation map for {patient_name}")
        patient_data = [d for d in target_data if d['subject_id'] == patient_name]
        patient_face_centers, patient_face_colors = build_3D_activation_map_single(
            patient_data, 
            mesh_file_name=mesh_file_names[patient_idx], 
            plot_TOS_key=plot_TOS_key,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax)
        patients_face_data.append({
            'subject_id': patient_name,
            'face_centers': patient_face_centers,
            'face_colors': patient_face_colors
        })
    return patients_face_data

def plot_3D_activation_map(patients_face_data, view_names=None, save_fig=False, save_fig_path=None, save_fig_name_prefix=None):
    """
    Plots a 3D activation map for a given patient.

    Parameters:
    patients_face_data (dict): A dictionary containing 'subject_id', 'face_centers', and 'face_colors' for a patient.
    view_names (list, optional): A list of views to plot. Possible values are 'front-left-side', 'front-right-side', and 'back-side'. Default is all three.
    save_fig (bool, optional): Whether to save the figure. Default is False.
    save_fig_path (str, optional): The path where to save the figure. Default is a predefined path.
    save_fig_name_prefix (str, optional): The prefix for the saved figure name. Default is the subject_id of the patient.

    Returns:
    None. This function shows the plot and optionally saves it as a figure.

    Raises:
    ValueError: If a view name is not one of the following: 'front-left-side', 'front-right-side', 'back-side'.
    """
    if view_names is None:
        view_names = ['front-left-side', 'front-right-side', 'back-side']
    if save_fig_name_prefix is None:
        save_fig_name_prefix = patients_face_data['subject_id']
    if save_fig_path is None:
        save_fig_path = '/p/mmcardiac/Jerry/code/DENSE-guided-Cine-Registration/test_results/2023-11-06-3D-Map-default'
    # view_names = ['front-left-side']
    face_centers = patients_face_data['face_centers']
    face_colors = patients_face_data['face_colors']
    for view_name in view_names:
        if view_name == 'front-left-side':
            elev, azim, roll = 55, 75, 90
        elif view_name == 'front-right-side':
            elev, azim, roll = -25, 80, 90
        elif view_name == 'back-side':
            elev, azim, roll = -110, 50, 90
        else:
            raise ValueError('view_name should be one of the following: front-left-side, front-right-side, back-side')

        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # normalize face_colors to [0,1]
        face_colors = (face_colors - face_colors.min()) / (face_colors.max() - face_colors.min())

        # Plot the face centers with their colors
        ax.scatter(face_centers[:,0], face_centers[:,1], face_centers[:,2], c=face_colors, s=50)

        # set the view angle
        # elev: elevation angle in the z plane
        # azim: azimuth angle in the x,y plane
        # roll: roll angle in the x,y plane
        ax.view_init(elev=elev, azim=azim, roll=roll) 



        # hide the axes
        ax.set_axis_off()

        # remove background color
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


        # plt.show()
        # save the figure without background color
        if save_fig:
            save_fig_fullname = Path(
                save_fig_path, 
                f'{save_fig_name_prefix}-{view_name}.png'
            )
            plt.savefig(
                save_fig_fullname, 
                bbox_inches='tight', pad_inches=0, transparent=True)