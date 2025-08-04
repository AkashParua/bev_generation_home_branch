#!/usr/bin/env python3
"""
Depth Map and Segmentation BEV Visualizer

This script processes depth maps (.pfm), segmentation masks (.npy), and camera calibration
files to generate various visualizations including Bird's Eye View (BEV) plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import yaml
import re
import argparse
import os
from pathlib import Path
from matplotlib.patches import Patch
from scipy.spatial.transform import Rotation as R

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
SCALE = 15000  # Scale factor to convert inverse depth to meters
MAX_DEPTH_FILTER = 30  # Maximum depth in meters (filter out distant objects)
VALID_MASK_THRESHOLD = 0.01  # Minimum threshold to filter invalid depth
INVERT_DEPTH = True  # Whether to apply 1/depth (for MiDaS-style inverse depth)
BEV_DOT_SIZE = 1.0  # Point size in BEV scatter plots
BEV_RANGE_X = 50  # Width of BEV plot (left-right) in meters
BEV_RANGE_Y = 50  # Height of BEV plot (forward-back) in meters

# =============================================================================
# CITYSCAPES COLOR PALETTE
# =============================================================================
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],      # 0: road
    [244, 35, 232],      # 1: sidewalk
    [70, 70, 70],        # 2: building
    [102, 102, 156],     # 3: wall
    [190, 153, 153],     # 4: fence
    [153, 153, 153],     # 5: pole
    [250, 170, 30],      # 6: traffic light
    [220, 220, 0],       # 7: traffic sign
    [107, 142, 35],      # 8: vegetation
    [152, 251, 152],     # 9: terrain
    [70, 130, 180],      # 10: sky
    [220, 20, 60],       # 11: person
    [255, 0, 0],         # 12: rider
    [0, 0, 142],         # 13: car
    [0, 0, 70],          # 14: truck
    [0, 60, 100],        # 15: bus
    [0, 80, 100],        # 16: train
    [0, 0, 230],         # 17: motorcycle
    [119, 11, 32]        # 18: bicycle
], dtype=np.uint8)

CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def read_pfm(path):
    """Read a PFM (Portable Float Map) file."""
    with open(path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header != 'Pf':
            raise Exception('Not a valid grayscale PFM file')

        dims = f.readline().decode('utf-8')
        width, height = map(int, re.findall(r'\d+', dims))

        scale = float(f.readline().decode('utf-8').strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        data = np.reshape(data, (height, width))
        data = np.flipud(data)

        return data

def load_intrinsics_and_extrinsics(yaml_path):
    """Load camera intrinsics and translation from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_intrinsic'], dtype=np.float32)
    translation = np.array(data['translation'], dtype=np.float32)
    return K, translation

def load_camera_extrinsics(yaml_path):
    """Load camera extrinsics (rotation and translation) from YAML file."""
    with open(yaml_path, 'r') as f:
        cam = yaml.safe_load(f)
    rotation_quat = cam["rotation"]  # [w, x, y, z]
    translation = np.array(cam["translation"])

    # Convert quaternion to rotation matrix
    r = R.from_quat([rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]])
    rotation_matrix = r.as_matrix()

    return rotation_matrix, translation

def apply_cityscapes_colormap(mask):
    """Apply Cityscapes color palette to segmentation mask."""
    return CITYSCAPES_COLORS[mask]

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_pfm(pfm_path, output_dir):
    """Visualize PFM depth map and save to file."""
    depth = read_pfm(pfm_path)

    print(f"✅ Loaded: {pfm_path}")
    print(f"Shape: {depth.shape}")
    print(f"Min depth value: {np.min(depth)}")
    print(f"Max depth value: {np.max(depth)}")

    plt.figure(figsize=(10, 5))
    plt.imshow(depth, cmap='plasma')
    plt.title("MiDaS Depth Map (.pfm)")
    plt.colorbar(label="Relative Depth")
    plt.axis('off')
    
    output_path = os.path.join(output_dir, "depth_map_raw.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved raw depth map to {output_path}")

def visualize_segmentation_npy(npy_path, output_dir):
    """Visualize segmentation mask from NPY file and save to file."""
    mask = np.load(npy_path)

    print(f"✅ Loaded: {npy_path}")
    print(f"Shape: {mask.shape}")
    print(f"Dtype: {mask.dtype}")

    # Handle logits
    if mask.ndim == 3:
        if mask.shape[0] < 100:
            print("⚠️ Detected logits (C, H, W) → taking argmax along axis 0")
            mask = np.argmax(mask, axis=0)
        elif mask.shape[-1] < 100:
            print("⚠️ Detected logits (H, W, C) → taking argmax along last axis")
            mask = np.argmax(mask, axis=-1)

    unique_classes = np.unique(mask)
    print(f"Unique classes: {unique_classes}")

    color_mask = apply_cityscapes_colormap(mask)

    # Plot mask
    plt.figure(figsize=(12, 8))
    plt.imshow(color_mask)
    plt.title("Cityscapes Segmentation Mask (.npy)")
    plt.axis('off')

    # Add legend
    legend_patches = []
    for class_id in unique_classes:
        if class_id < len(CITYSCAPES_NAMES):
            color = CITYSCAPES_COLORS[class_id] / 255  # normalize RGB to [0,1]
            name = CITYSCAPES_NAMES[class_id]
            patch = Patch(facecolor=color, label=name)
            legend_patches.append(patch)

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_path = os.path.join(output_dir, "segmentation_mask.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved segmentation mask to {output_path}")

def depth_to_3d_and_plot(pfm_path, yaml_path, output_dir):
    """Convert PFM depth map to 3D camera space and save visualization."""
    depth = read_pfm(pfm_path)
    K, translation = load_intrinsics_and_extrinsics(yaml_path)

    # Step 1: Mask invalid values
    depth[depth <= VALID_MASK_THRESHOLD] = np.nan

    # Step 2: Invert if needed (MiDaS-style inverse depth)
    if INVERT_DEPTH:
        print("↩️ Inverting inverse depth map (1 / pfm)")
        eps = 1e-6
        depth = 1.0 / (depth + eps)

    # Step 3: Apply scale factor
    print(f"📐 Applying scale factor: {SCALE}")
    depth *= SCALE  # Now in real-world meters

    # Step 4: Remove distant sky pixels
    depth[depth > MAX_DEPTH_FILTER] = np.nan

    # Step 5: Reproject to 3D using intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = depth.shape
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    X = (u_coords - cx) * depth / fx
    Y = (v_coords - cy) * depth / fy
    Z = depth

    points_3d = np.stack((X, Y, Z), axis=-1)

    # Step 6: Visualize scaled depth
    valid_depth = depth[~np.isnan(depth)]
    vmin, vmax = np.nanpercentile(valid_depth, [2, 98])
    depth_viz = np.clip(depth, vmin, vmax)

    plt.figure(figsize=(10, 5))
    im = plt.imshow(depth_viz, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Depth (meters)')
    plt.title(f"Depth Map (vmin={vmin:.2f} m, vmax={vmax:.2f} m)\nScaled using factor {SCALE}")
    plt.axis('off')
    
    output_path = os.path.join(output_dir, "depth_map_scaled.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved scaled depth map to {output_path}")

    return points_3d, depth

def plot_bev(points_3d, output_dir, filename="bev_depth.png"):
    """Plot Bird's Eye View of 3D points and save to file."""
    H, W, _ = points_3d.shape
    flattened = points_3d.reshape(-1, 3)
    X, Y, Z = flattened[:, 0], flattened[:, 1], flattened[:, 2]

    valid_mask = np.isfinite(X) & np.isfinite(Z)
    X, Z = X[valid_mask], Z[valid_mask]

    xlim = (-BEV_RANGE_X/2, BEV_RANGE_X/2)
    zlim = (0, BEV_RANGE_Y)

    plt.figure(figsize=(8, 8))
    plt.scatter(X, Z, s=BEV_DOT_SIZE, c=Z, cmap='plasma', alpha=0.7)
    plt.colorbar(label='Depth (meters)')
    plt.xlabel("X (left ↔ right)")
    plt.ylabel("Z (forward)")
    plt.title("Bird's Eye View (Depth-colored)")
    plt.xlim(xlim)
    plt.ylim(zlim)
    plt.grid(True)
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved BEV plot to {output_path}")

def plot_bev_with_segmentation(points_3d, seg_mask, output_dir, filename="bev_segmentation.png"):
    """Plot BEV with segmentation coloring and save to file."""
    H, W, _ = points_3d.shape
    assert seg_mask.shape == (H, W), "Segmentation mask must match shape of points_3d"

    # Flatten everything
    flat_points = points_3d.reshape(-1, 3)
    flat_classes = seg_mask.flatten()

    X = flat_points[:, 0]
    Z = flat_points[:, 2]
    class_ids = flat_classes

    valid = np.isfinite(X) & np.isfinite(Z) & (class_ids >= 0) & (class_ids < len(CITYSCAPES_COLORS))
    X, Z, class_ids = X[valid], Z[valid], class_ids[valid]

    # Map class_id to RGB colors
    colors = CITYSCAPES_COLORS[class_ids] / 255.0  # normalize RGB to [0, 1]

    xlim = (-BEV_RANGE_X/2, BEV_RANGE_X/2)
    zlim = (0, BEV_RANGE_Y)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(X, Z, s=BEV_DOT_SIZE, c=colors, marker='.', alpha=0.8)
    plt.xlabel("X (left ↔ right)")
    plt.ylabel("Z (forward)")
    plt.title("BEV: Segmentation-colored 3D Projection")
    plt.xlim(xlim)
    plt.ylim(zlim)
    plt.grid(True)

    # Add legend
    unique_classes = np.unique(class_ids)
    legend_patches = [Patch(facecolor=CITYSCAPES_COLORS[c]/255.0, label=CITYSCAPES_NAMES[c])
                      for c in unique_classes if c < len(CITYSCAPES_NAMES)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved segmentation BEV to {output_path}")

def plot_bev_with_segmentation_and_ego(points_3d, seg_mask, yaml_path, output_dir, filename="bev_ego_frame.png"):
    """Plot BEV in ego frame with segmentation coloring and save to file."""
    H, W, _ = points_3d.shape
    assert seg_mask.shape == (H, W), "Segmentation mask must match shape of points_3d"

    # Flatten and filter
    flat_points = points_3d.reshape(-1, 3)
    flat_classes = seg_mask.flatten()
    valid = (
        np.isfinite(flat_points[:, 0]) &
        np.isfinite(flat_points[:, 2]) &
        (flat_classes >= 0) & (flat_classes < len(CITYSCAPES_COLORS))
    )
    flat_points = flat_points[valid]
    class_ids = flat_classes[valid]

    # Load extrinsics
    R_cam2ego, T_cam2ego = load_camera_extrinsics(yaml_path)
    points_ego = (R_cam2ego @ flat_points.T).T + T_cam2ego

    # Map to colors
    colors = CITYSCAPES_COLORS[class_ids] / 255.0

    # Define plot limits centered at camera
    center_x, center_y = T_cam2ego[0], T_cam2ego[1]
    xlim = (center_x - BEV_RANGE_X / 2, center_x + BEV_RANGE_X / 2)
    ylim = (center_y - BEV_RANGE_Y / 2, center_y + BEV_RANGE_Y / 2)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(points_ego[:, 0], points_ego[:, 1], s=BEV_DOT_SIZE, c=colors, marker='.', alpha=0.8)
    plt.xlabel("X (left ↔ right)")
    plt.ylabel("Y (forward)")
    plt.title("BEV (Ego Frame): Segmentation-colored 3D Projection")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)

    # Add legend
    unique_classes = np.unique(class_ids)
    legend_patches = [Patch(facecolor=CITYSCAPES_COLORS[c]/255.0, label=CITYSCAPES_NAMES[c])
                      for c in unique_classes if c < len(CITYSCAPES_NAMES)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"💾 Saved ego frame BEV to {output_path}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process depth maps and segmentation masks to generate BEV visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--pfm_path", type=str, required=True,
                        help="Path to the PFM depth map file")
    parser.add_argument("--seg_path", type=str, required=True,
                        help="Path to the NPY segmentation mask file")
    parser.add_argument("--calib_path", type=str, required=True,
                        help="Path to the YAML camera calibration file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output images")
    parser.add_argument("--all_plots", action="store_true",
                        help="Generate all possible plots (default: only BEV with segmentation)")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting depth and segmentation processing...")
    print(f"PFM Path: {args.pfm_path}")
    print(f"Segmentation Path: {args.seg_path}")
    print(f"Calibration Path: {args.calib_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Scale Factor: {SCALE}")
    print(f"Max Depth Filter: {MAX_DEPTH_FILTER}m")
    print("-" * 50)

    # Load segmentation mask
    segmentation_mask = np.load(args.seg_path)
    
    # Handle logits if present
    if segmentation_mask.ndim == 3:
        if segmentation_mask.shape[0] < 100:
            print("⚠️ Detected logits (C, H, W) → taking argmax along axis 0")
            segmentation_mask = np.argmax(segmentation_mask, axis=0)
        elif segmentation_mask.shape[-1] < 100:
            print("⚠️ Detected logits (H, W, C) → taking argmax along last axis")
            segmentation_mask = np.argmax(segmentation_mask, axis=-1)

    # Convert depth to 3D points
    points_3d, depth_meters = depth_to_3d_and_plot(args.pfm_path, args.calib_path, args.output_dir)

    if args.all_plots:
        # Generate all visualizations
        print("\n📊 Generating all visualizations...")
        visualize_pfm(args.pfm_path, args.output_dir)
        visualize_segmentation_npy(args.seg_path, args.output_dir)
        plot_bev(points_3d, args.output_dir, "bev_depth_only.png")
        plot_bev_with_segmentation(points_3d, segmentation_mask, args.output_dir, "bev_camera_frame.png")
        plot_bev_with_segmentation_and_ego(points_3d, segmentation_mask, args.calib_path, args.output_dir, "bev_ego_frame.png")
    else:
        # Generate only the main BEV plots
        print("\n📊 Generating BEV visualizations...")
        plot_bev_with_segmentation(points_3d, segmentation_mask, args.output_dir, "bev_camera_frame.png")
        plot_bev_with_segmentation_and_ego(points_3d, segmentation_mask, args.calib_path, args.output_dir, "bev_ego_frame.png")

    print(f"\n✅ Processing complete! All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
