from mesh_functions import load_mesh, decimate_mesh, project_mesh, create_vertex_colors, create_texture_from_vertex_colors, display_mesh, views
from segmentation_functions import WoundsImages, compute_average_angle, generate_segmentation_mask, fill_mask_holes

import pylab as plt
import numpy as np
import open3d as o3d
from datetime import datetime
import os

projection_directory = "projections/"

def segment_wound(obj_file):
    start = datetime.now()

    mesh = load_mesh(file_path=obj_file)
    print(f"{len(mesh.vertices)} vertices in the mesh before merging")
    mesh.merge_close_vertices(eps=1e-8)
    print(f"{len(mesh.vertices)} vertices in the mesh after merging")

    if not mesh.has_vertex_colors:
        mesh.vertex_colors = create_vertex_colors(mesh, verbose=True)

    target_number_of_triangles = 10000
    low_res_mesh = decimate_mesh(mesh, target_number_of_triangles)
    print(f"Low-res mesh has {len(low_res_mesh.triangles)} triangles.")

    low_res_mesh = create_texture_from_vertex_colors(low_res_mesh)

    projections = project_mesh(mesh=low_res_mesh, views=views, RES=256, output_dir=projection_directory, filename=obj_file.split("/")[-1][:-4], compute_vertex_map=False)

    woundsImages = WoundsImages(projections, output_dir="outputs", logging=True)

    angle_shift = np.pi / 6
    avg_pitch, avg_yaw = compute_average_angle(woundsImages)
    result_views = {
        "front-result": [avg_pitch, avg_yaw],
        "left-result": [avg_pitch, avg_yaw - angle_shift],
        "right-result": [avg_pitch, avg_yaw + angle_shift],
        "top-result": [avg_pitch - angle_shift, avg_yaw],
        "bottom-result": [avg_pitch + angle_shift, avg_yaw]
    }
    print(f"Result angle: pitch {avg_pitch}, yaw {avg_yaw}")

    final_projections = project_mesh(low_res_mesh, views=result_views, RES=256, output_dir="outputs/results", filename=obj_file.split("/")[-1][:-4], compute_vertex_map=True)

    SecondSegmentation = WoundsImages(final_projections, output_dir="outputs", logging=True)
    SecondSegmentation.generate(show=False, save=True)

    mask = generate_segmentation_mask(low_res_mesh, SecondSegmentation)
    mask = fill_mask_holes(low_res_mesh, mask, fill_depth=5, color_threshold=0.2, adjacency=[])

    #display_mesh(low_res_mesh, mask)

    mask_vertex_colors = np.zeros((len(low_res_mesh.vertices), 3), dtype=np.float32)
    for i in range(len(mask)):
        if mask[i]:
            mask_vertex_colors[i] = [1, 1, 1]
    low_res_mesh.vertex_colors = o3d.utility.Vector3dVector(mask_vertex_colors)

    low_res_mesh = create_texture_from_vertex_colors(low_res_mesh)
    mask_texture = np.asarray(low_res_mesh.textures[0])

    # Correct orientation if needed for web viewer
    mask_texture = np.flipud(mask_texture)

    export_dir = "outputs/exported_mesh"
    os.makedirs(export_dir, exist_ok=True)
    obj_out_path = os.path.join(export_dir, "output.obj")
    texture_out_path = os.path.join(export_dir, "segmented_texture.png")

    plt.imsave(texture_out_path, mask_texture)

    low_res_mesh.vertex_colors = o3d.utility.Vector3dVector()

    # Ensure triangle UVs are present
    if not low_res_mesh.has_triangle_uvs():
        print("Warning: mesh has no triangle UVs! Texture may not render properly.")

    o3d.io.write_triangle_mesh(obj_out_path, low_res_mesh,
                               write_vertex_colors=True,
                               write_triangle_uvs=True)

    # Fix .obj content manually
    with open(obj_out_path, "r") as f:
        lines = f.readlines()

    with open(obj_out_path, "w") as f:
        found_mtl = False
        found_usemtl = False
        for line in lines:
            if line.strip().startswith("mtllib"):
                f.write("mtllib material.mtl\n")
                found_mtl = True
            elif line.strip().startswith("usemtl"):
                f.write("usemtl material_0\n")
                found_usemtl = True
            elif line.strip().startswith("f ") and not found_usemtl:
                f.write("usemtl material_0\n")
                found_usemtl = True
                f.write(line)
            else:
                f.write(line)
        if not found_mtl:
            f.write("mtllib material.mtl\n")
        if not found_usemtl:
            f.write("usemtl material_0\n")

    print(f"Estimated PWAT: {int(np.mean([img.pwat for img in SecondSegmentation.wounds_images]))}")
    print(f"It took {datetime.now() - start} to perform the segmentation.")

    return obj_out_path, texture_out_path, int(np.mean([img.pwat for img in SecondSegmentation.wounds_images]))
