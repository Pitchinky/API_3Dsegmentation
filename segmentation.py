# segmentation.py

import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from wound_segmentation import segment_wound
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from wound_segmentation import segment_wound
import matplotlib.pyplot as plt
import shutil

def process_obj_with_texture(obj_path, texture_path, output_dir, method="kmeans Jay"):
    print(f"Segmentation method: {method}")
    os.makedirs(output_dir, exist_ok=True)

    output_obj_path = os.path.join(output_dir, "output.obj")
    mask_path = os.path.join(output_dir, "segmented_texture.png")

    if method == "kmeans Jay":
        image = cv2.imread(texture_path)
        if image is None:
            raise FileNotFoundError(f"Texture not found: {texture_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = kmeans_segmentation_Jay(image_rgb)

        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)

        plt.imsave(mask_path, mask, cmap='gray')

        with open(obj_path, 'r') as f_in, open(output_obj_path, 'w') as f_out:
            found_mtl = False
            for line in f_in:
                if line.strip().startswith('mtllib'):
                    f_out.write('mtllib material.mtl\n')
                    found_mtl = True
                else:
                    f_out.write(line)
            if not found_mtl:
                f_out.write('mtllib material.mtl\n')

    elif method == "projection Mathieu":
        obj_out, texture_out, _ = segment_wound(obj_path)

        # Vérifie que la texture segmentée a la même taille que l'originale
        original_texture = cv2.imread(texture_path)
        segmented_texture = cv2.imread(texture_out)

        if original_texture.shape[:2] != segmented_texture.shape[:2]:
            # Redimensionne la texture segmentée à la taille d'origine
            segmented_texture_resized = cv2.resize(segmented_texture, (original_texture.shape[1], original_texture.shape[0]))
            texture_out_resized_path = os.path.join(output_dir, "segmented_texture.png")
            cv2.imwrite(texture_out_resized_path, segmented_texture_resized)
        else:
            texture_out_resized_path = os.path.join(output_dir, "segmented_texture.png")
            shutil.copyfile(texture_out, texture_out_resized_path)

        # Copie l'OBJ
        output_obj_path = os.path.join(output_dir, "output.obj")
        shutil.copyfile(obj_out, output_obj_path)

        # Copie/écrit le MTL
        mtl_path = os.path.join(output_dir, "material.mtl")
        with open(mtl_path, 'w') as f:
            f.write("newmtl material_0\n")
            f.write("Ka 1.000 1.000 1.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write("d 1.0\n")
            f.write("illum 2\n")
            f.write("map_Kd segmented_texture.png\n")

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    # Création du fichier .mtl
    mtl_path = os.path.join(output_dir, "material.mtl")
    with open(mtl_path, 'w') as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write("map_Kd segmented_texture.png\n")

    return output_obj_path, mask_path, mtl_path




def kmeans_segmentation_Jay(image_rgb, K=2):
    original_shape = image_rgb.shape[:2]
    small_img = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    img_reshaped = small_img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(img_reshaped)
    labels = kmeans.labels_

    cluster_means = np.mean(kmeans.cluster_centers_, axis=1)
    wound_cluster = np.argmin(cluster_means)

    mask_small = (labels == wound_cluster).astype(np.uint8) * 255
    mask_resized = mask_small.reshape((256, 256))
    mask_full = cv2.resize(mask_resized, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask_full

def projections_Mathieu(obj_path):
    obj_out_path, texture_out_path, _ = segment_wound(obj_path)
    return obj_out_path, texture_out_path
