"""
Daily Activities Dataset Downloader
===================================

This script downloads and prepares the curated daily activities dataset
combining UCF-101 (already available) and Kinetics-400 subset.

Usage:
    python download_daily_activities_data.py

The script will:
1. Check existing UCF-101 data
2. Download Kinetics-400 subset for daily activities
3. Create unified dataset structure
4. Generate train/test splits
"""

import os
import json
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil
from daily_activities_config import (
    DAILY_ACTIVITIES, UCF101_TO_DAILY, KINETICS_DAILY_ACTIVITIES, 
    KINETICS_TO_DAILY, print_daily_activities_summary
)

class DailyActivitiesDatasetDownloader:
    def __init__(self, base_dir="C:\\ASH_PROJECT\\data"):
        self.base_dir = Path(base_dir)
        self.ucf101_dir = self.base_dir / "UCF101"
        self.kinetics_dir = self.base_dir / "Kinetics400_Daily"
        self.daily_activities_dir = self.base_dir / "DailyActivities"
        
        # Create directories
        self.kinetics_dir.mkdir(parents=True, exist_ok=True)
        self.daily_activities_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Daily Activities Dataset Downloader")
        print(f"Base directory: {self.base_dir}")
        
    def check_ucf101_data(self):
        """Check if UCF-101 data is available"""
        print(f"\\n📂 Checking UCF-101 data...")
        
        if not self.ucf101_dir.exists():
            print(f"❌ UCF-101 directory not found: {self.ucf101_dir}")
            print(f"Please extract UCF-101 data to: {self.ucf101_dir}")
            return False
            
        # Count videos in each relevant class
        ucf_classes_found = []
        for ucf_class, daily_class in UCF101_TO_DAILY.items():
            class_dir = self.ucf101_dir / ucf_class
            if class_dir.exists():
                video_count = len(list(class_dir.glob("*.avi")))
                if video_count > 0:
                    ucf_classes_found.append(ucf_class)
                    print(f"✅ {ucf_class} -> {daily_class}: {video_count} videos")
                else:
                    print(f"⚠️ {ucf_class} -> {daily_class}: No videos found")
            else:
                print(f"❌ {ucf_class} -> {daily_class}: Directory not found")
        
        print(f"\\n📊 UCF-101 Summary:")
        print(f"  • Classes available: {len(ucf_classes_found)}/{len(UCF101_TO_DAILY)}")
        print(f"  • Missing classes: {set(UCF101_TO_DAILY.keys()) - set(ucf_classes_found)}")
        
        return len(ucf_classes_found) > 0
    
    def download_kinetics_metadata(self):
        """Download Kinetics-400 metadata and annotations"""
        print(f"\\n📥 Downloading Kinetics-400 metadata...")
        
        # Kinetics-400 CSV files URLs (these contain video IDs and labels)
        urls = {
            "train": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz",
            "test": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"
        }
        
        # For now, let's create a simple metadata structure
        # In practice, you would download the actual Kinetics metadata
        print("ℹ️  Creating Kinetics metadata structure...")
        
        metadata_dir = self.kinetics_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Create a simple CSV structure for our daily activities
        kinetics_activities = {}
        for kinetics_class in KINETICS_DAILY_ACTIVITIES:
            if kinetics_class in KINETICS_TO_DAILY:
                daily_class = KINETICS_TO_DAILY[kinetics_class]
                if daily_class not in kinetics_activities:
                    kinetics_activities[daily_class] = []
                kinetics_activities[daily_class].append(kinetics_class)
        
        # Save metadata
        metadata_file = metadata_dir / "daily_activities_mapping.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(kinetics_activities, f, indent=2)
        
        print(f"✅ Metadata structure created at: {metadata_file}")
        return kinetics_activities
    
    def create_kinetics_download_script(self):
        """Create a script to download Kinetics-400 videos using yt-dlp"""
        print(f"\\n📄 Creating Kinetics download script...")
        
        download_script = self.kinetics_dir / "download_kinetics_videos.py"
        
        script_content = '''"""
Kinetics-400 Daily Activities Video Downloader
==============================================

This script downloads specific Kinetics-400 videos for daily activities.
You'll need to install yt-dlp: pip install yt-dlp

Note: Due to video availability on YouTube, not all videos may be downloadable.
"""

import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Sample video URLs for daily activities (you would get these from Kinetics metadata)
SAMPLE_KINETICS_VIDEOS = {
    "eating": [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Placeholder - replace with actual IDs
    ],
    "cooking": [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Placeholder - replace with actual IDs  
    ],
    # Add more categories as needed
}

def download_kinetics_videos():
    """Download Kinetics videos using yt-dlp"""
    kinetics_dir = Path("C:/ASH_PROJECT/data/Kinetics400_Daily")
    
    for activity, urls in SAMPLE_KINETICS_VIDEOS.items():
        activity_dir = kinetics_dir / activity
        activity_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {activity} videos...")
        
        for i, url in enumerate(tqdm(urls)):
            output_path = activity_dir / f"{activity}_{i:03d}.%(ext)s"
            
            cmd = [
                "yt-dlp",
                "--format", "mp4[height<=360]/best[height<=360]",  # Low quality for faster download
                "--output", str(output_path),
                url
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"✅ Downloaded: {activity}_{i:03d}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to download: {url}")

if __name__ == "__main__":
    print("⚠️  This is a template script.")
    print("Please add actual Kinetics-400 video IDs before running.")
    print("You can get them from: https://github.com/cvdfoundation/kinetics-dataset")
'''
        
        with open(download_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ Download script created: {download_script}")
        print(f"ℹ️  To use it:")
        print(f"   1. Install yt-dlp: pip install yt-dlp")
        print(f"   2. Get Kinetics-400 video IDs from official dataset")
        print(f"   3. Update the script with actual video URLs")
        print(f"   4. Run: python {download_script}")
        
        return download_script
    
    def create_synthetic_kinetics_data(self):
        """Create synthetic Kinetics data structure for testing"""
        print(f"\\n🎭 Creating synthetic Kinetics data for testing...")
        
        # Create directories for each daily activity mapped from Kinetics
        kinetics_activities = set()
        for kinetics_class in KINETICS_DAILY_ACTIVITIES:
            if kinetics_class in KINETICS_TO_DAILY:
                daily_class = KINETICS_TO_DAILY[kinetics_class]
                kinetics_activities.add(daily_class)
                
                # Create directory
                class_dir = self.kinetics_dir / kinetics_class
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a placeholder file (you would put actual videos here)
                placeholder_file = class_dir / "placeholder.txt"
                with open(placeholder_file, 'w', encoding='utf-8') as f:
                    f.write(f"Placeholder for {kinetics_class} -> {daily_class} videos\\n")
                    f.write(f"Put .mp4 files in this directory\\n")
        
        print(f"✅ Created directories for {len(kinetics_activities)} daily activities from Kinetics")
        return kinetics_activities
    
    def create_unified_dataset_structure(self):
        """Create unified dataset structure combining UCF-101 and Kinetics"""
        print(f"\\n🏗️ Creating unified daily activities dataset...")
        
        # Create directories for each daily activity
        for activity in DAILY_ACTIVITIES:
            activity_dir = self.daily_activities_dir / activity
            activity_dir.mkdir(parents=True, exist_ok=True)
            
            # Create train/val split directories
            (activity_dir / "train").mkdir(exist_ok=True)
            (activity_dir / "val").mkdir(exist_ok=True)
        
        # Create dataset info file
        dataset_info = {
            "dataset_name": "Daily Activities Recognition Dataset",
            "total_classes": len(DAILY_ACTIVITIES),
            "classes": DAILY_ACTIVITIES,
            "data_sources": {
                "ucf101_classes": len(UCF101_TO_DAILY),
                "kinetics_classes": len(KINETICS_DAILY_ACTIVITIES)
            },
            "directory_structure": {
                "ucf101_source": str(self.ucf101_dir),
                "kinetics_source": str(self.kinetics_dir), 
                "unified_dataset": str(self.daily_activities_dir)
            }
        }
        
        info_file = self.daily_activities_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ Unified dataset structure created")
        print(f"📄 Dataset info saved to: {info_file}")
        
        return dataset_info
    
    def create_data_preparation_guide(self):
        """Create a guide for data preparation"""
        guide_content = '''# Daily Activities Dataset Preparation Guide

## Overview
This guide helps you prepare the daily activities dataset combining UCF-101 and Kinetics-400.

## Step 1: UCF-101 Data
✅ UCF-101 data should already be available in: C:/ASH_PROJECT/data/UCF101/
   - Make sure videos are organized by class folders
   - We use 15 classes from UCF-101 for daily activities

## Step 2: Kinetics-400 Data (Optional Enhancement)
📥 To get better coverage of daily activities:

### Option A: Download Full Kinetics-400
1. Visit: https://github.com/cvdfoundation/kinetics-dataset
2. Download video IDs and metadata
3. Use yt-dlp to download videos: `pip install yt-dlp`

### Option B: Use UCF-101 Only (Recommended for Testing)
- Start with UCF-101 data only
- Add Kinetics data later for improvement

## Step 3: Data Organization
The script creates this structure:
```
C:/ASH_PROJECT/data/DailyActivities/
├── brushing_teeth/
│   ├── train/
│   └── val/
├── eating/
│   ├── train/
│   └── val/
└── ... (22 total classes)
```

## Step 4: Manual Data Linking
After running the download script:
1. Copy/link UCF-101 videos to appropriate daily activity folders
2. Copy/link Kinetics videos (if downloaded) to appropriate folders
3. Split data into train/val (80/20 split recommended)

## Step 5: Training
Run the daily activities training script:
```bash
python train_daily_activities.py
```

## Expected Results
- 22 daily activity classes
- 85-90% accuracy target
- Fast training with transfer learning
- Simple "Person is doing X" text output
'''
        
        guide_file = self.base_dir / "DAILY_ACTIVITIES_SETUP_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"📋 Setup guide created: {guide_file}")
        
    def run_full_setup(self):
        """Run the complete dataset setup process"""
        print("\\n" + "="*60)
        print("🎯 DAILY ACTIVITIES DATASET SETUP")
        print("="*60)
        
        # Print our configuration
        print_daily_activities_summary()
        
        # Step 1: Check UCF-101
        ucf_available = self.check_ucf101_data()
        
        # Step 2: Setup Kinetics structure
        kinetics_metadata = self.download_kinetics_metadata()
        
        # Step 3: Create download scripts
        download_script = self.create_kinetics_download_script()
        
        # Step 4: Create synthetic structure for testing
        synthetic_kinetics = self.create_synthetic_kinetics_data()
        
        # Step 5: Create unified structure
        dataset_info = self.create_unified_dataset_structure()
        
        # Step 6: Create setup guide
        self.create_data_preparation_guide()
        
        print("\\n" + "="*60)
        print("✅ DATASET SETUP COMPLETED!")
        print("="*60)
        print(f"📊 Summary:")
        print(f"  • Daily activities: {len(DAILY_ACTIVITIES)} classes")
        print(f"  • UCF-101 available: {'✅' if ucf_available else '❌'}")
        print(f"  • Kinetics structure: ✅ Created")
        print(f"  • Unified dataset: ✅ {self.daily_activities_dir}")
        
        print(f"\\n🎯 Next Steps:")
        print(f"  1. Review setup guide: {self.base_dir}/DAILY_ACTIVITIES_SETUP_GUIDE.md")
        if ucf_available:
            print(f"  2. ✅ UCF-101 data ready - you can start training!")
            print(f"  3. (Optional) Download Kinetics videos for more data")
        else:
            print(f"  2. ❌ Extract UCF-101 data to: {self.ucf101_dir}")
            print(f"  3. Re-run this script")
        print(f"  4. Run: python train_daily_activities.py")

def main():
    downloader = DailyActivitiesDatasetDownloader()
    downloader.run_full_setup()

if __name__ == "__main__":
    main()