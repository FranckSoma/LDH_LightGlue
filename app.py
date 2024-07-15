from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

torch.set_grad_enabled(False)
images = Path("assets")

# Load extractor and Matcher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# Testing LightGlue
image0 = load_image(images / "city_1.png")
image1 = load_image(images / "city_2.png")

# Debug: Check if images are loaded correctly
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image0.permute(1, 2, 0).cpu().numpy())
plt.title('Image 0')
plt.subplot(1, 2, 2)
plt.imshow(image1.permute(1, 2, 0).cpu().numpy())
plt.title('Image 1')
plt.show()

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# Debug: Check if matched keypoints are extracted correctly
print(f'Number of matches: {matches.shape[0]}')

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

plt.show()

# # Extract matched keypoints for homography
# src_pts = kpts0[matches[:, 0]].cpu().numpy()
# dst_pts = kpts1[matches[:, 1]].cpu().numpy()

# # Dynamic homography functions
# def normalize(v):
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     return v / norm

# def calculate_frustum_intersection(R, t, ground_plane_normal, ground_plane_point, f, w_pi, h_pi):
#     P_inv = np.linalg.inv(np.vstack((np.hstack((R.T, -R.T @ t)), [0, 0, 0, 1])))
#     corners = [
#         [-w_pi / 2, -h_pi / 2, f, 1],
#         [w_pi / 2, -h_pi / 2, f, 1],
#         [-w_pi / 2, h_pi / 2, f, 1],
#         [w_pi / 2, h_pi / 2, f, 1]
#     ]
#     rays = [P_inv @ np.array(corner).T for corner in corners]
#     rays = [ray[:3] / ray[3] if ray[3] != 0 else ray[:3] for ray in rays]
#     rays = [ray - t.flatten() for ray in rays]
#     rays = [normalize(ray) for ray in rays]

#     t_vals = [(ground_plane_point - t.flatten()) @ ground_plane_normal / (ray @ ground_plane_normal) if (ray @ ground_plane_normal) != 0 else 0 for ray in rays]
#     intersections = [t.flatten() + t_val * ray for t_val, ray in zip(t_vals, rays)]
    
#     return intersections

# def find_overlap_translation(intersections_i, intersections_j, overlap_x, overlap_y):
#     A_prime_i, B_prime_i, E_prime_i, D_prime_i = intersections_i
#     A_prime_j, B_prime_j, E_prime_j, D_prime_j = intersections_j
    
#     B_prime_j_new = A_prime_i + overlap_x * (B_prime_i - A_prime_i)
#     D_prime_j_new = E_prime_i + overlap_x * (D_prime_i - E_prime_i)
    
#     return B_prime_j_new, D_prime_j_new

# def calculate_new_ground_points(intersections_i, intersections_j, overlap_x, overlap_y):
#     B_prime_j_new, D_prime_j_new = find_overlap_translation(intersections_i, intersections_j, overlap_x, overlap_y)
    
#     A_prime_i, B_prime_i, E_prime_i, D_prime_i = intersections_i
#     rho = normalize(E_prime_i - A_prime_i)
    
#     angle_D_prime_jCjB_prime_j = np.arccos(np.dot(normalize(intersections_j[1] - t.flatten()), rho))
#     angle_CjB_prime_jD_prime_j = np.pi - (np.pi / 2 + angle_D_prime_jCjB_prime_j)
#     distance_B_prime_jD_prime_j = np.linalg.norm(intersections_j[1] - t.flatten()) * np.sin(np.pi / 2) / np.sin(angle_CjB_prime_jD_prime_j)
    
#     D_prime_j_new = B_prime_j_new + rho * distance_B_prime_jD_prime_j
    
#     return B_prime_j_new, D_prime_j_new

# def calculate_new_rotation_translation(B_prime_j_new, D_prime_j_new, f, w_pi, h_pi, alpha_h, alpha_v, Cj):
#     d = np.linalg.norm(D_prime_j_new - Cj.flatten())
    
#     k = normalize(D_prime_j_new - B_prime_j_new)
#     m = normalize(Cj.flatten() + d * (D_prime_j_new - B_prime_j_new) / np.linalg.norm(D_prime_j_new - B_prime_j_new))
    
#     alpha_h = alpha_h / 2
#     alpha_v = alpha_v / 2
    
#     z_prime = (m * np.cos(alpha_h) + np.cross(k, m) * np.sin(alpha_h) + k * (np.dot(k, m) * (1 - np.cos(alpha_h))))
#     z_prime = normalize(z_prime)
    
#     y = k
#     x = normalize(np.cross(y, z_prime))
    
#     R_new = np.vstack((x, y, z_prime)).T
#     t_new = -R_new @ Cj.flatten()
    
#     return R_new, t_new

# def dynamic_homography(point_cloud, R, t, ground_plane_normal, ground_plane_point, overlap_x, overlap_y, f, w_pi, h_pi, alpha_h, alpha_v):
#     intersections_i = calculate_frustum_intersection(R, t, ground_plane_normal, ground_plane_point, f, w_pi, h_pi)
#     intersections_j = calculate_frustum_intersection(R, t, ground_plane_normal, ground_plane_point, f, w_pi, h_pi)
    
#     B_prime_j_new, D_prime_j_new = calculate_new_ground_points(intersections_i, intersections_j, overlap_x, overlap_y)
    
#     R_new, t_new = calculate_new_rotation_translation(B_prime_j_new, D_prime_j_new, f, w_pi, h_pi, alpha_h, alpha_v, t)
    
#     return R_new, t_new

# # Example parameters for dynamic homography
# R = np.eye(3)
# t = np.array([[0], [0], [0]])
# ground_plane_normal = np.array([0, 1, 0])
# ground_plane_point = np.array([0, 0, 0])
# overlap_x = 0.5
# overlap_y = 0.5
# f = 1.0
# w_pi = 1.0
# h_pi = 1.0
# alpha_h = np.pi / 4
# alpha_v = np.pi / 4

# # Compute dynamic homography
# R_new, t_new = dynamic_homography(None, R, t, ground_plane_normal, ground_plane_point, overlap_x, overlap_y, f, w_pi, h_pi, alpha_h, alpha_v)

# # Debug: Check if dynamic homography is computed correctly
# print("New Rotation Matrix:\n", R_new)
# print("New Translation Vector:\n", t_new)

# # Apply the new rotation and translation
# H_dynamic = np.hstack((R_new, t_new.reshape(-1, 1)))
# H_dynamic = np.vstack((H_dynamic, [0, 0, 0, 1]))

# # Apply the dynamic homography to the image
# image0_np = image0.permute(1, 2, 0).cpu().numpy()
# image1_np = image1.permute(1, 2, 0).cpu().numpy()

# # Determine the size of the canvas
# height, width, channels = image1_np.shape
# canvas_width = width + image0_np.shape[1]

# # Warp image0 to align with image1 using dynamic homography
# warped_image_dynamic = cv2.warpPerspective(image0_np, H_dynamic[:3], (canvas_width, height))

# # Debug: Check dimensions of the warped image
# print(f'Warped image dimensions: {warped_image_dynamic.shape}')

# # Create a canvas to combine the images
# canvas_dynamic = np.zeros((height, canvas_width, channels), dtype=np.uint8)

# # Place the images on the canvas
# canvas_dynamic[:, :width] = image1_np
# canvas_dynamic[:, :warped_image_dynamic.shape[1]] = np.maximum(canvas_dynamic[:, :warped_image_dynamic.shape[1]], warped_image_dynamic)

# # Debug: Check if the images are placed correctly on the canvas
# plt.figure(figsize=(20, 10))
# plt.imshow(canvas_dynamic)
# plt.title("Canvas with Combined Images using Dynamic Homography")
# plt.axis("off")
# plt.show()
