#!/usr/bin/env python3
"""
NuScenes Sample Extractor - Simple File Collection

This script extracts all samples from nuScenes scenes and organizes them into directories.
It simply copies files without modifying calibration data.

Usage:
    python export_scene_samples.py --config config.yaml --output_dir ./exported_data [--scene_names scene1 scene2]
"""

import os
import argparse
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes


def main():
    parser = argparse.ArgumentParser(description='Extract all samples from nuScenes scenes.')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to nuScenes dataset root directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save exported data')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='nuScenes dataset version (default: v1.0-mini)')
    parser.add_argument('--scene_names', type=str, nargs='+', default=None, help='Scene names to export')
    parser.add_argument('--all_scenes', action='store_true', help='Export all scenes')
    parser.add_argument('--max_scenes', type=int, default=None, help='Maximum scenes to export')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    dataroot = args.dataroot
    nusc_version = args.version
    verbose = args.verbose
    
    print(f"🚀 Loading nuScenes {nusc_version} from {dataroot}")
    nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=verbose)
    print(f"✅ Loaded {len(nusc.scene)} scenes")
    
    # Select scenes to export
    if args.all_scenes:
        selected_scenes = nusc.scene
    elif args.scene_names:
        selected_scenes = [scene for scene in nusc.scene if scene['name'] in args.scene_names]
        if not selected_scenes:
            print(f"❌ No scenes found matching: {args.scene_names}")
            return
    else:
        # Default: export first scene
        selected_scenes = [nusc.scene[0]]
        print(f"ℹ️  No scenes specified, exporting first scene: {selected_scenes[0]['name']}")
    
    # Apply max_scenes limit
    if args.max_scenes and len(selected_scenes) > args.max_scenes:
        selected_scenes = selected_scenes[:args.max_scenes]
    
    print(f"📁 Exporting {len(selected_scenes)} scenes to {args.output_dir}")
    
    # Export each scene
    for scene in tqdm(selected_scenes, desc="Exporting scenes"):
        scene_name = scene['name']
        scene_dir = Path(args.output_dir) / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 Processing scene: {scene_name}")
        
        # Get all samples in the scene
        sample_token = scene['first_sample_token']
        sample_count = 0
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            sample_dir = scene_dir / f"sample_{sample_count:03d}_{sample_token[:8]}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all sensor data for this sample
            for sensor, token in sample['data'].items():
                data = nusc.get('sample_data', token)
                src_path = Path(dataroot) / data['filename']
                
                # Determine sensor type and create appropriate directory
                channel = data['channel'].lower()
                if 'cam' in channel:
                    sensor_dir = sample_dir / 'cameras'
                elif 'lidar' in channel:
                    sensor_dir = sample_dir / 'lidar'
                elif 'radar' in channel:
                    sensor_dir = sample_dir / 'radar'
                else:
                    sensor_dir = sample_dir / 'other'
                
                sensor_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                dst_path = sensor_dir / f"{sensor}{src_path.suffix}"
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"⚠️  Failed to copy {src_path}: {e}")
                
                # Copy original calibration file (without modification)
                calib_dir = sample_dir / 'calibration'
                calib_dir.mkdir(parents=True, exist_ok=True)
                cs = nusc.get('calibrated_sensor', data['calibrated_sensor_token'])
                with open(calib_dir / f'{sensor}_calib.yaml', 'w') as f:
                    yaml.dump(cs, f, default_flow_style=False)
            
            sample_count += 1
            sample_token = sample['next'] if sample['next'] != '' else None
        
        print(f"✅ Exported {sample_count} samples from scene {scene_name}")
    
    print(f"\n🎉 Export completed! Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
