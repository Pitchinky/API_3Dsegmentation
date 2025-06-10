#segmentation_function.py
from deepskin import wound_segmentation
from deepskin import evaluate_PWAT_score
from deepskin import wound_segmentation, evaluate_PWAT_score
from deepskin.imgproc import imfill, get_perilesion_mask
from mesh_functions import compute_mesh_adjacency, Projection, views

import pylab as plt
import numpy as np
import cv2
import os

class WoundImage:
    """
    Class to store and compute the segmentation on an image.
    """
    def __init__(self, projection:Projection,
                 output_dir: str, logging: bool):
        self.output_dir: str = output_dir
        self.logging: bool = logging
        self.projection = projection

        self.img = cv2.imread(self.projection.file_path)[..., ::-1]

        self.segmentation = wound_segmentation(
            img=self.img, tol=0.95, verbose=self.logging
        )
        self.wound_mask, self.body_mask, self.bg_mask = cv2.split(
            self.segmentation)
        
        self.wound_area = cv2.countNonZero(self.wound_mask)

        # Add property to keep only the biggest wound mask
        # Use connected components to identify the largest connected component in wound_mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.wound_mask, connectivity=8)
        if num_labels > 1:
            # Ignore the background (label 0) and find the largest component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            self.biggest_wound_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        else:
            self.biggest_wound_mask = self.wound_mask.copy()
        self.biggest_wound_area = cv2.countNonZero(self.biggest_wound_mask)

        self.wound_masked = cv2.bitwise_and(
            self.img, self.img, mask=self.wound_mask)
        pwm = get_perilesion_mask(mask=self.wound_mask)
        self.peri_wound_mask = cv2.bitwise_and(
            pwm, pwm, mask=imfill(self.body_mask | self.wound_mask)
        )
        self.peri_wound_masked = cv2.bitwise_and(
            self.img, self.img, mask=self.peri_wound_mask
        )
        self.pwat = evaluate_PWAT_score(
            img=self.img, mask=self.segmentation, verbose=self.logging
        )

        self.pertinence = self.pwat * self.projection.non_bg_ratio

        self.figsize = (7, 7)
        self.fontsize = 24
        self.axis = "off"
        self.alpha = 0.75
        self.bbox_inches = "tight"
        self.dpi = 300

    def generate(self, show: bool = True, save: bool = False):
        """datasets/3D-Models/labeled/spider_proj1_v1/
        Display or save the Deepskin results
        Args:
            show: whether or not to display the results
            save: whether or not to save the results
        """
        self._original(show=show, save=save)
        self._segmentation_mask(show=show, save=save)
        self._segmentation_semantic(show=show, save=save)
        self._mask_wound(show=show, save=save)
        self._biggest_mask_wound(show=show, save=save)
        self._mask_peri_wound(show=show, save=save)
        self._pwat_estimation(show=show, save=save)

    def _original(self, show: bool, save: bool):
        plt.figure(figsize=self.figsize)
        plt.imshow(self.img)
        plt.title("Original Image", fontsize=self.fontsize)
        plt.axis(self.axis)
        if save:
            dir = os.path.join(self.output_dir, "original")
            os.makedirs(dir, exist_ok=True)
            plt.savefig(
                os.path.join(dir, os.path.basename(self.projection.file_path)),
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close()

    def _segmentation_mask(self, show: bool, save: bool):
        plt.figure(figsize=self.figsize)
        plt.imshow(self.segmentation)
        plt.title("Segmentation Mask", fontsize=self.fontsize)
        plt.axis(self.axis)
        if save:
            dir = os.path.join(self.output_dir, "segmentation_mask")
            os.makedirs(dir, exist_ok=True)
            plt.savefig(
                os.path.join(dir, os.path.basename(self.projection.file_path)),
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close()

    def _segmentation_semantic(self, show: bool, save: bool):
        plt.figure(figsize=self.figsize)
        plt.imshow(self.img)
        plt.contour(self.body_mask, colors="blue", linewidths=1)
        plt.contour(self.wound_mask, colors="lime", linewidths=2)
        plt.title("Semantic Segmentation", fontsize=self.fontsize)
        plt.axis(self.axis)
        if save:
            dir = os.path.join(self.output_dir, "segmentation_semantic")
            os.makedirs(dir, exist_ok=True)
            plt.savefig(
                os.path.join(dir, os.path.basename(self.projection.file_path)),
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close()

    def _mask_wound(self, show: bool, save: bool):
        plt.figure(figsize=self.figsize)
        plt.imshow(self.wound_masked)
        plt.imshow(self.img, alpha=self.alpha)
        plt.contour(self.wound_mask, colors="lime", linewidths=1)
        plt.title("Wound Mask", fontsize=self.fontsize)
        plt.axis(self.axis)
        if save:
            dir = os.path.join(self.output_dir, "mask_wound")
            os.makedirs(dir, exist_ok=True)
            plt.savefig(
                os.path.join(dir, os.path.basename(self.projection.file_path)),
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close()
    
    def _biggest_mask_wound(self, show: bool, save: bool):
        plt.figure(figsize=self.figsize)
        plt.imshow(self.biggest_wound_mask)
        plt.imshow(self.img, alpha=self.alpha)
        plt.contour(self.wound_mask, colors="lime", linewidths=1)
        plt.title("Biggest Wound Mask", fontsize=self.fontsize)
        plt.text(
            0.3,
            0.05,
            f"wounds area: {self.wound_area:.3f}\nbiggest wound area: {self.biggest_wound_area:.3f}",
            transform=plt.gca().transAxes,
            fontsize=20,
            bbox=dict(facecolor="white", alpha=self.alpha, edgecolor="k"),
        )
        plt.axis(self.axis)
        if save:
            dir = os.path.join(self.output_dir, "biggest_mask_wound")
            os.makedirs(dir, exist_ok=True)
            plt.savefig(
                os.path.join(dir, os.path.basename(self.projection.file_path)),
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close()

    def _mask_peri_wound(self, show: bool, save: bool):
        plt.figure(figsize=self.figsize)
        plt.imshow(self.peri_wound_masked)
        plt.imshow(self.img, alpha=self.alpha)
        plt.contour(self.peri_wound_mask, colors="lime", linewidths=1)
        plt.title("Peri-Wound Mask", fontsize=self.fontsize)
        plt.axis(self.axis)
        if save:
            dir = os.path.join(self.output_dir, "mask_peri_wound")
            os.makedirs(dir, exist_ok=True)
            plt.savefig(
                os.path.join(dir, os.path.basename(self.projection.file_path)),
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close()

    def _pwat_estimation(self, show: bool, save: bool):
        plt.figure(figsize=self.figsize)
        plt.imshow(self.wound_masked)
        plt.imshow(self.img, alpha=self.alpha)
        plt.contour(self.wound_mask, colors="lime", linewidths=1)
        plt.title("PWAT Estimation", fontsize=self.fontsize)
        plt.text(
            0.3,
            0.05,
            f"PWAT score: {self.pwat:.3f}",
            transform=plt.gca().transAxes,
            fontsize=20,
            bbox=dict(facecolor="white", alpha=self.alpha, edgecolor="k"),
        )
        plt.axis(self.axis)
        if save:
            dir = os.path.join(self.output_dir, "pwat_estimation")
            os.makedirs(dir, exist_ok=True)
            plt.savefig(
                os.path.join(dir, os.path.basename(self.projection.file_path)),
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close()


class WoundsImages:
    """
    Class to store the segmentations on the projections.
    """
    def __init__(self, projections, output_dir: str, logging: bool):
        self.wounds_images: list[WoundImage] = [
            WoundImage(projection, output_dir, logging)
            for projection in projections
        ]

    def generate(self, show: bool = True, save: bool = False):
        """
        saves and/or save the stored segmentations
        Args:
            show: whether or not to display the results
            save: whether or not to save the results
        """
        for wound_image in self.wounds_images:
            wound_image.generate(show=show, save=save)

def normalize_angles(pitch, yaw):
    """
    Normalizes a (pitch, yaw) angle between pitch [-pi/2, pi/2] and yaw [-pi, pi[
    In case of Gimbal Lock, sets yaw to 0
    
    Args:
        pitch: the pitch angle
        yaw: the yaw angle
    Returns:
        pitch, yaw: The normalized angle
    """
    # First, normalize yaw into the range [-pi, pi[
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    # Adjust pitch (rotation about X-axis) if it's outside [-pi/2, pi/2]
    if pitch > np.pi / 2:
        pitch = np.pi - pitch
        yaw += np.pi
    elif pitch < -np.pi / 2:
        pitch = -np.pi - pitch
        yaw += np.pi

    # Re-normalize yaw after adjustment
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    # In case of Gimbal Lock, set the yaw to 0
    if abs(pitch) == np.pi / 2:
        yaw = 0
    return pitch, yaw

def get_wound_image_weight(woundsImages:WoundsImages, angle):
    """
    Returns weight for the wound image segmentation for the specified angle

    Args:
        woundsImages: the list of woundsImages containing one at angle
        angle: the target angle
    Returns:
        The weight if angle is found in woundsImages None otherwise
    """
    x, y = angle
    x, y = normalize_angles(x, y)
    view_angle = [x, y]
    for wound_image in woundsImages.wounds_images:
        if view_angle == views[wound_image.projection.angle]:
            return wound_image.pertinence
    return 0

def weighted_average_pitch_yaw(angles, weights):
    """
    Compute the weighted average of 3D orientations (pitch, yaw).

    Args:
        angles: list of tuples (pitch, yaw) in radians.
        weights: list of corresponding weights.

    Returns:
        avg_pitch, avg_yaw representing the average orientation in radians,
        or None if the weighted sum is near zero.
    """
    sum_vec = np.array([0.0, 0.0, 0.0])
    total_weight = 0.0
    
    for (pitch, yaw), w in zip(angles, weights):
        # Convert pitch and yaw to a unit vector.
        x = np.cos(pitch) * np.sin(yaw)
        y = np.sin(pitch)
        z = np.cos(pitch) * np.cos(yaw)
        vec = np.array([x, y, z])
        
        # Weighted accumulation.
        sum_vec += w * vec
        total_weight += w

    # Check if the sum is effectively zero.
    norm = np.linalg.norm(sum_vec)
    if norm < 1e-9:
        return None  # No well-defined average (directions cancel out)

    avg_vec = sum_vec / norm

    # Convert back to pitch and yaw.
    avg_pitch = np.asin(avg_vec[1])
    avg_yaw = np.atan2(avg_vec[0], avg_vec[2])
    return avg_pitch, avg_yaw

def closest_view(pitch, yaw, views=views):
    """
    Returns the key from the `views` dictionary that is closest to the given pitch and yaw.
    
    Args:
        pitch (float): The pitch angle in radians.
        yaw (float): The yaw angle in radians.
        views: the views. Defaults to views (previously defined)
    
    Returns:
        The key corresponding to the closest view.
    """
    def angular_diff(a, b):
        # Compute minimal angular difference, accounting for wrap-around.
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    # Find the view with the minimal Euclidean distance in (pitch, yaw) space.
    return min(views.items(),
               key=lambda kv: np.sqrt(
                   angular_diff(pitch, kv[1][0])**2 +
                   angular_diff(yaw, kv[1][1])**2
               ))[0]

    
def compute_average_angle(woundsImages:WoundsImages):
    """
    Computes and returns the average angle using image and neighboors value.

    Args:
        woundsImages: the woundsImages you want to compute the best angle from.
    Returns:
        avg_pitch, avg_yaw: The average pitch and yaw of the determined angle
    """
    angles = []
    weights = []
    for wound_image in woundsImages.wounds_images:
        angle = views[wound_image.projection.angle]
        angles.append(angle)
        self_weight = get_wound_image_weight(woundsImages=woundsImages, angle=angle) * 4
        # Taking neighboors weight into account prevents the average angle to be placed such as the side
        # projection would not display the wound
        left_weight = get_wound_image_weight(woundsImages=woundsImages, angle=[angle[0], angle[1]+np.pi/4])
        right_weight = get_wound_image_weight(woundsImages=woundsImages, angle=[angle[0], angle[1]-np.pi/4])
        top_weight = get_wound_image_weight(woundsImages=woundsImages, angle=[angle[0]-np.pi/4, angle[1]])
        bottom_weight = get_wound_image_weight(woundsImages=woundsImages, angle=[angle[0]+np.pi/4, angle[1]])
        weights.append(self_weight + left_weight + right_weight + top_weight + bottom_weight)
    avg_pitch, avg_yaw = weighted_average_pitch_yaw(angles=angles, weights=weights)
    return avg_pitch, avg_yaw

#woundsImages.generate(show=False, save=True)
def print_seg_info(woundsImages:WoundsImages, angle):
    """
    Prints informations about woundsImages. For debugging purposes

    Args:
        woundsImages: the list of woundsImages containing one at angle
        angle: the target angle
    """
    x, y = angle
    x, y = normalize_angles(x, y)
    view_angle = [x, y]
    for wound_image in woundsImages.wounds_images:
            if view_angle == views[wound_image.projection.angle]:
                print(f"Infos for segmentation at {wound_image.projection.angle} angle :\n\
                        \r\tPWAT {wound_image.pwat:.2f}\n\
                        \r\tBiggest Wound Area {wound_image.biggest_wound_area} pixels\n\
                        \r\tNon-bg-score {wound_image.projection.non_bg_ratio*100:.2f}%\n\
                        \r\tSelection score {(get_wound_image_weight(woundsImages, angle)):.2f}")

angle_shift = np.pi / 6

def generate_segmentation_mask(mesh, woundsImages:WoundsImages):
    """
    Creates a mask for the mesh accordingly to the segmentation of the woundsImages

    Args:
        mesh: the mesh for which to generate a mask
        woundsImages: the wound segmentations with which to generate the mask
    Returns:
        The mask as an array of boolean
    """
    mask = np.full(len(mesh.vertices), False)
    for woundImage in woundsImages.wounds_images:
        for y in range(len(woundImage.wound_mask)):
            for x in range(len(woundImage.wound_mask[y])):
                if woundImage.wound_mask[y, x] != 0:
                    z = woundImage.projection.depth_buffer[y, x]
                    if z == np.inf or not (x, y, z) in woundImage.projection.vertex_map:
                        continue
                    for vertex in woundImage.projection.vertex_map[(x, y, z)]:
                        mask[vertex] = True
    return mask

def fill_mask_holes(mesh, mask, fill_depth = 1, color_threshold = 0.05, adjacency = []):
    """
    Uses BFS on the mesh to fill the holes bony marking all the vertices within fill_depth and color_threshold of the average original mask of any masked vertex
    """
    if len(adjacency) == 0:
        adjacency = compute_mesh_adjacency(mesh)
    starting_vertices = np.flatnonzero(mask)
    colors = np.asarray(mesh.vertex_colors)
    mark = np.copy(mask)
    new_mask = np.copy(mask)
    queue = []
    for index in starting_vertices:
        queue.append((index, 0, -1))
        mark[index] = True
    while queue:
        current, depth, source = queue.pop(0)
        mark[current] = True
        if depth <= fill_depth and source != -1 and np.max(np.abs(colors[current]-colors[source]))<=color_threshold:
            new_mask[current] = True
            for neighbor in adjacency[current]:
                if not mark[neighbor]:
                    queue.append((neighbor, depth + 1, current))

    return new_mask
