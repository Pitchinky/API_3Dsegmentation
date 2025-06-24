#mesh_functions.py
import pylab as plt
import numpy as np
import open3d as o3d
import os

def load_mesh(file_path):
    """
    Load a .obj (Wavefront) file with texture support and return an Open3D mesh object.
    
    Args:
        file_path: The path of the wavefront file
    Returns:
        the loaded mesh
    """
    try:
        mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=True)
        print("Has textures:", mesh.has_textures())
        print("Triangle UVs:", mesh.has_triangle_uvs())

        
        # Ensure the mesh has normals
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Ensure the mesh has texture coordinates
        if not mesh.has_triangle_uvs():
            print("Warning: The mesh does not have texture coordinates (UVs).")
        
        print(f"Successfully loaded {file_path} with texture support")
        return mesh
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        return None

def decimate_mesh(mesh, target_number_of_triangles):
    return mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)


def create_vertex_colors(mesh, verbose=False):
    """
    Creates a numpy array of the vertex colors of a mesh using its textures
    Args:
        mesh: The mesh to use.
        verbose: Whether to display progress.
    
    Returns:
        vertex_colors: The resulting array.
    """

    if len(mesh.textures) == 0:
        raise ValueError("Le mesh n’a pas de textures chargées.")

    if not mesh.has_triangle_uvs():
        raise ValueError("Le mesh ne contient pas de coordonnées UV.")
    
    vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    
    if verbose:
        print("Creating vertex colors, please wait...")
    
    num_triangles = len(mesh.triangles)
    progress_step = max(1, num_triangles // 100)
    
    for triangle_index in range(num_triangles):
        # Choose the texture corresponding to this triangle based on its material id
        texture_index = mesh.triangle_material_ids[triangle_index]
        texture_image = mesh.textures[texture_index]
        texture_np = np.asarray(texture_image)
        height, width, _ = texture_np.shape
        
        # Process each vertex of the triangle
        for local_vertex in range(3):
            u, v = mesh.triangle_uvs[triangle_index * 3 + local_vertex]
            x = int(u * (width - 1))
            y = int(v * (height - 1))
            global_vertex_index = mesh.triangles[triangle_index][local_vertex]
            vertex_colors[global_vertex_index] = texture_np[y, x] / 255.0
        
        if verbose and (triangle_index % progress_step) == 0:
            print('#', end='')
    
    if verbose:
        print()
    
    return o3d.utility.Vector3dVector(vertex_colors)


def create_texture_from_vertex_colors(mesh):
    vertex_colors = np.asarray(mesh.vertex_colors)

    texture_dimension = int(np.ceil(np.sqrt(len(mesh.triangles) * 3)))

    texture = np.full((texture_dimension, texture_dimension, 3), 0, dtype=np.float32)
    triangle_uvs = np.full((len(mesh.triangles) * 3, 2), 0, dtype=np.float64)

    for triangle_index, triangle in enumerate(mesh.triangles):
        for vertex_sub_index in range(3):
            UV_index = 3 * triangle_index + vertex_sub_index
            U = UV_index % texture_dimension
            V = UV_index // texture_dimension
            texture[V, U] = vertex_colors[triangle[vertex_sub_index]]
            triangle_uvs[UV_index] = (
                (U + 0.5) / texture_dimension,
                (V + 0.5) / texture_dimension
            )
    mesh.textures = [o3d.geometry.Image(texture)]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))
    return mesh


def display_mesh(mesh, mask = [], color = [1, 0, 0], use_vertex_colors = True, use_textures = False):
    """
    Displays a 3D mesh using Open3D's visualization window.
    
    Args:
        mesh: The 3D mesh to be displayed.
        mask: a mask to be displayed in the color parameter
        color: parameter used when a mask is given
        use_vertex_colors: if set to false ignores vertices colors for display
        use_textures: if set to true uses textures for display
    """
    if not use_textures:
        textures_backup = mesh.textures
        mesh.textures = []
    
    if not use_vertex_colors:
        vertex_colors_backup = np.asarray(mesh.vertex_colors).copy()
        

    if not mesh.has_vertex_colors() or not use_vertex_colors:
        num_vertices = len(mesh.vertices)
        grey_value = 0.5
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((num_vertices, 3), grey_value))

    if len(mask)>0:
        for vertex in range(len(mesh.vertices)):
            if mask[vertex]:
                mesh.vertex_colors[vertex] = color
    
    o3d.visualization.draw_geometries([mesh])

    if not use_textures:
        mesh.textures = textures_backup
    
    if not use_vertex_colors:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_backup)

class Projection:
    """
    Class to store a projection informations.
    """
    def __init__(self, mesh, mask, file_path, angle, non_bg_ratio, depth_buffer, vertex_map):
        self.mesh = mesh
        self.mask = mask
        self.file_path = file_path
        self.angle = angle #angle name (not radians)
        self.non_bg_ratio = non_bg_ratio
        self.depth_buffer = depth_buffer
        self.vertex_map = vertex_map

# Defaults views for the first 26 projections.
views = {
        "front":        [0, 0],
        "back":         [0, -np.pi],
        "left":         [0, -np.pi/2],
        "right":        [0, np.pi/2],
        "top":          [-np.pi/2, 0],
        "bottom":       [np.pi/2, 0],
        "front-right":  [0, np.pi/4],
        "front-left":   [0, -np.pi/4],
        "back-right":   [0, 3*np.pi/4],
        "back-left":    [0, -3*np.pi/4],
        "top-front":    [-np.pi/4, 0],
        "top-right":    [-np.pi/4, np.pi/2],
        "top-back":     [-np.pi/4, -np.pi],
        "top-left":     [-np.pi/4, -np.pi/2],
        "bottom-front": [np.pi/4, 0],
        "bottom-right": [np.pi/4, np.pi/2],
        "bottom-back":  [np.pi/4, -np.pi],
        "bottom-left":  [np.pi/4, -np.pi/2],
        "top-front-right":    [-np.pi/4, np.pi/4],
        "top-front-left":     [-np.pi/4, -np.pi/4],
        "top-back-right":     [-np.pi/4, 3*np.pi/4],
        "top-back-left":      [-np.pi/4, -3*np.pi/4],
        "bottom-front-right": [np.pi/4, np.pi/4],
        "bottom-front-left":  [np.pi/4, -np.pi/4],
        "bottom-back-right":  [np.pi/4, 3*np.pi/4],
        "bottom-back-left":   [np.pi/4, -3*np.pi/4],
    }

def project_mesh(mesh, views={"front": [0, 0]}, RES=256, output_dir=".", filename="projection", compute_vertex_map = False):
    """
    Creates a list of projections at the desired views angles, with RES resolution and saves the projections
    Args:
        mesh: the mesh to project
        views: the views angles to use
        RES: the resolution (RES*RES*RES)
        output_dir: the folder in which to save the projections
        filename: the base filename for the projections (E.G saves filename_viewangle.png)
        compute_vertex_map: whether or not to compute the vertex map of the projection. Setting it to false can save time if the map is not required.
    Returns:
        The list of projections
    """
    projections = []
    
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use deepcopy to keep an unmodified copy of the original mesh.
    original_vertices = np.asarray(mesh.vertices).copy()
    
    # Loop through each view.
    for view_name, angles in views.items():
        print(f"creating projection for {view_name} angle, please wait...")
        pitch, yaw = angles
        
        R = mesh.get_rotation_matrix_from_xyz([pitch, yaw, 0])
        mesh.rotate(R, center=mesh.get_center())

        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        # Define camera direction as +Z in the mesh's coordinate system and create the mask
        camera_dir = np.array([0, 0, 1], dtype=float)
        visibility_mask = np.dot(normals, camera_dir) > 0
        if np.count_nonzero(visibility_mask)  < 0.5 * len(visibility_mask):
            print(f"not enough vertices are visible. Skipping {view_name} angle.")
            mesh.vertices = o3d.utility.Vector3dVector(original_vertices)
            continue


        # Determine grid scale.
        # Initialize the output image and a depth buffer.
        image = np.full((RES, RES, 3), 255, dtype=np.uint8)
        depth_buffer = np.full((RES, RES), np.inf, dtype=float)
        voxel_to_vertices = {}

        
        # Get the axis-aligned bounding box and its minimum bound.
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()  # [min_x, min_y, min_z]
        extent = bbox.get_extent()  # [width, height, depth] of the bounding box.
        max_extent_xy = max(extent[0], extent[1])
        voxel_size = max_extent_xy / RES  # World-space size per grid cell.
        
        # If the mesh has vertex colors, use them; otherwise, default to white.
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)  # assumed in [0,1]
        else:
            vertex_colors = None

        # Helper function to check if a 2D point is inside a triangle.
        def point_in_triangle(pt, a, b, c):
            def sign(p, q, r):
                return (p[0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[1] - r[1])
            d1 = sign(pt, a, b)
            d2 = sign(pt, b, c)
            d3 = sign(pt, c, a)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (has_neg and has_pos)

        # Process triangles: iterate over each triangle in the mesh.
        progress_step = len(mesh.triangles)//100
        for idx in range(len(mesh.triangles)):
            tri = mesh.triangles[idx]
            if idx % progress_step == 0:
                print('#', end='')
            # Each triangle is given by indices of its three vertices.
            p1, p2, p3 = tri
            
            # Process triangle only if all its vertices are visible.
            if not (visibility_mask[p1] and visibility_mask[p2] and visibility_mask[p3]):
                continue
            
            # Retrieve the vertices (world coordinates).
            v1 = vertices[p1]
            v2 = vertices[p2]
            v3 = vertices[p3]
            
            # Get vertex colors for the triangle if available;
            # here we take the average color of its three vertices.
            if vertex_colors is not None:
                tri_color = (vertex_colors[[p1, p2, p3]].mean(axis=0) * 255).astype(np.uint8)
            else:
                tri_color = np.array([255, 255, 255], dtype=np.uint8)
            
            # Project the triangle's vertices to the 2D plane (x, y).
            v1_xy = v1[:2]
            v2_xy = v2[:2]
            v3_xy = v3[:2]

            # Create a bounding box for the triangle
            tri_min = np.min(np.array([v1, v2, v3]), axis=0)
            tri_max = np.max(np.array([v1, v2, v3]), axis=0)

            # Select all the voxels touched by the triangle
            x_min = max(int(np.floor((tri_min[0] - min_bound[0]) / voxel_size)), 0)
            x_max = min(int(np.ceil((tri_max[0] - min_bound[0]) / voxel_size)), RES - 1)
            y_min = max(int(np.floor((tri_min[1] - min_bound[1]) / voxel_size)), 0)
            y_max = min(int(np.ceil((tri_max[1] - min_bound[1]) / voxel_size)), RES - 1)
            z_avg = int((((tri_min[2] + tri_max[2])/2) - min_bound[2]) / voxel_size)
            #NOTE: We do not need to clamp z_avg as we are projecting along the Z axis, some values may exceed RES

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    center_x = min_bound[0] + (x + 0.5) * voxel_size
                    center_y = min_bound[1] + (y + 0.5) * voxel_size
                    pt = np.array([center_x, center_y])

                    # Check if the voxel center lies inside the projected triangle.
                    if point_in_triangle(pt, v1_xy, v2_xy, v3_xy):
                        if z_avg < depth_buffer[y, x]:
                            depth_buffer[y, x] = z_avg
                            image[y, x] = tri_color
            
        print()
        non_bg_ratio = np.sum(depth_buffer < np.inf)/(RES**2)
        NON_BG_THRESHOLD = 0.10
        voxel_to_vertices = {}
        progress_step = len(vertices)//100
        if compute_vertex_map and non_bg_ratio >= NON_BG_THRESHOLD:
            print("computing vertex map, please wait...")
            for idx, vertex in enumerate(vertices):
                if idx % progress_step == 0:
                    print('#', end='')
                if not visibility_mask[idx]:
                    continue
                x, y, z = vertex
                # Compute grid cell indices.
                x = int((x - min_bound[0]) / voxel_size)
                y = int((y - min_bound[1]) / voxel_size)
                z = int((z - min_bound[2]) / voxel_size)
                # Clamp indices to ensure they are within bounds.
                if x < 0 or x >= RES or y < 0 or y >= RES:
                    continue
                # Append the vertex (its 3D coordinate) to the corresponding voxel.
                if (x, y, z) not in voxel_to_vertices:
                    voxel_to_vertices[(x, y, z)] = []
                voxel_to_vertices[(x, y, z)].append(idx)
            print()
        mesh.vertices = o3d.utility.Vector3dVector(original_vertices)
        if non_bg_ratio >= NON_BG_THRESHOLD:
            # Save the image.
            file_path = os.path.join(output_dir, f"{filename}_{view_name}.png")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.imsave(file_path, image)
            print(f"Saving {file_path} ({non_bg_ratio*100:.2f}% visibility)...")
            proj = Projection(mesh=mesh, mask=visibility_mask, file_path=file_path, angle=view_name, non_bg_ratio=non_bg_ratio, depth_buffer=depth_buffer, vertex_map=voxel_to_vertices)
            projections.append(proj)
    return projections

def reconstruct_projection(projection:Projection):
    """
    Reconstructs and displays a projection based on its informations.
    Only useful for debugging purposes

    Args:
        projection: The projection to reconstruct
    """
    resolution = len(projection.depth_buffer)
    image = np.ones((resolution, resolution, 3), dtype=float)
    vertex_colors = np.asarray(projection.mesh.vertex_colors)[projection.mask]
    for y in range(resolution):
        for x in range(resolution):
            z = projection.depth_buffer[y, x]
            if z == -np.inf:
                continue
            local_vertices = projection.vertex_map[(x, y, z)]
            local_color_avg = [0, 0, 0]
            for vertex in local_vertices:
                local_color_avg += vertex_colors[vertex]
            local_color_avg /= len(local_vertices)
            image[y, x] = local_color_avg
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def compute_mesh_adjacency(mesh):
    """
    computes a mesh adjency list and returns it
    """
    adjacency = np.empty(len(mesh.vertices), dtype=object)
    adjacency[:] = [[] for _ in range(len(mesh.vertices))]

    for (i, j, k) in mesh.triangles:
        adjacency[i].append(j)
        adjacency[i].append(k)
        adjacency[j].append(i)
        adjacency[j].append(k)
        adjacency[k].append(i)
        adjacency[k].append(j)
    return adjacency