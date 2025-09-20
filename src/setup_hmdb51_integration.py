"""
HMDB-51 Dataset Integration System
=================================

Downloads, processes, and integrates HMDB-51 with UCF-101 for Phase 4.
Creates a unified dataset with 18 daily activities (9 UCF + 9 HMDB).

Usage:
    python setup_hmdb51_integration.py --download
    python setup_hmdb51_integration.py --process
    python setup_hmdb51_integration.py --integrate
"""

import os
import requests
import rarfile
import shutil
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import json
from collections import defaultdict

class HMDB51Integrator:
    """HMDB-51 dataset integration system"""
    
    def __init__(self):
        self.project_root = Path("C:/ASH_PROJECT")
        self.data_root = self.project_root / "data"
        self.hmdb_root = self.data_root / "HMDB51"
        self.ucf_root = self.data_root / "UCF101"
        self.integrated_root = self.data_root / "UCF_HMDB_Combined"
        
        # HMDB-51 download info
        self.hmdb_url = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
        self.hmdb_rar_file = self.data_root / "hmdb51_org.rar"
        
        # Activity mappings
        self.ucf_activities = [
            "WalkingWithDog", "Swimming", "Biking", "Lifting", "JumpingJack",
            "Pullups", "Pushups", "Typing", "BrushingTeeth"
        ]
        
        # HMDB-51 to daily activity mapping
        self.hmdb_to_daily = {
            "eat": "eating",
            "pour": "cooking",
            "drink": "cooking",  # Drinking can be considered cooking/food prep
            "sit": "sitting",
            "situp": "yoga",  # Situps are exercise/yoga
            "walk": "walking", 
            "run": "running",
            "ride_bike": "biking",  # Map to biking from UCF-101
            "ride_horse": "driving",  # Similar to driving (riding)
            "push": "pushups",  # Map to pushups from UCF-101
            "pullup": "pullups",  # Map to pullups from UCF-101
            "brush_hair": "brushing_teeth",  # Similar brushing activity
            "climb_stairs": "walking"  # Stair climbing is similar to walking
        }
        
        # Combined activities (UCF + HMDB mapped activities)
        self.combined_activities = [
            # UCF-101 activities (renamed for consistency)
            "walking_dog", "swimming", "biking", "weight_lifting", "jumping_jacks",
            "pullups", "pushups", "typing", "brushing_teeth",
            # HMDB-51 activities + overlapping mapped activities
            "eating", "cooking", "yoga", "walking", "running", "sitting"
        ]
        
        print(f"🔧 HMDB-51 Integrator Initialized")
        print(f"📁 Project root: {self.project_root}")
        print(f"💾 Available space: {shutil.disk_usage(self.data_root)[2] / (1024**3):.1f} GB")
    
    def download_hmdb51(self):
        """Download HMDB-51 dataset"""
        print(f"\n📥 Downloading HMDB-51 dataset...")
        
        if self.hmdb_rar_file.exists():
            print(f"✅ HMDB-51 RAR file already exists: {self.hmdb_rar_file}")
            return True
        
        # Create data directory
        self.data_root.mkdir(exist_ok=True)
        
        try:
            print(f"🌐 Downloading from: {self.hmdb_url}")
            print(f"📁 Saving to: {self.hmdb_rar_file}")
            
            # Download with progress bar
            response = requests.get(self.hmdb_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.hmdb_rar_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Download completed: {self.hmdb_rar_file.stat().st_size / (1024**2):.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if self.hmdb_rar_file.exists():
                self.hmdb_rar_file.unlink()
            return False
    
    def extract_hmdb51(self):
        """Extract HMDB-51 RAR file using 7-Zip"""
        print(f"\n📦 Extracting HMDB-51 dataset...")
        
        if not self.hmdb_rar_file.exists():
            print(f"❌ HMDB-51 RAR file not found: {self.hmdb_rar_file}")
            return False
        
        if self.hmdb_root.exists() and list(self.hmdb_root.iterdir()):
            print(f"✅ HMDB-51 already extracted: {self.hmdb_root}")
            return self.extract_activity_rars()
        
        try:
            # Create extraction directory
            self.hmdb_root.mkdir(exist_ok=True)
            
            # Use 7-Zip to extract RAR file
            print(f"🗜️  Extracting main RAR to: {self.hmdb_root}")
            
            import subprocess
            cmd = [
                r"C:\Program Files\7-Zip\7z.exe",
                "x", str(self.hmdb_rar_file),
                f"-o{self.hmdb_root}",
                "-y"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 7-Zip extraction failed: {result.stderr}")
                return False
            
            print(f"✅ Main extraction completed")
            
            # Now extract individual activity RAR files
            return self.extract_activity_rars()
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def extract_activity_rars(self):
        """Extract individual activity RAR files"""
        print(f"\n🔄 Extracting individual activity RAR files...")
        
        # Find all RAR files in HMDB51 directory
        rar_files = list(self.hmdb_root.glob("*.rar"))
        if not rar_files:
            print(f"❌ No activity RAR files found in {self.hmdb_root}")
            return False
        
        print(f"📁 Found {len(rar_files)} activity RAR files")
        
        import subprocess
        extracted_count = 0
        
        for rar_file in tqdm(rar_files, desc="Extracting activities"):
            activity_name = rar_file.stem
            activity_dir = self.hmdb_root / activity_name
            
            # Skip if already extracted
            if activity_dir.exists() and list(activity_dir.glob("*.avi")):
                extracted_count += 1
                continue
            
            # Create activity directory
            activity_dir.mkdir(exist_ok=True)
            
            # Extract activity RAR
            cmd = [
                r"C:\Program Files\7-Zip\7z.exe",
                "x", str(rar_file),
                f"-o{activity_dir}",
                "-y"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    extracted_count += 1
                else:
                    print(f"⚠️  Failed to extract {activity_name}: {result.stderr}")
            except Exception as e:
                print(f"⚠️  Error extracting {activity_name}: {e}")
        
        print(f"✅ Extracted {extracted_count}/{len(rar_files)} activities")
        
        # Clean up RAR files to save space
        print(f"🧹 Cleaning up RAR files...")
        for rar_file in rar_files:
            try:
                rar_file.unlink()
            except:
                pass
        
        return extracted_count > 0
    
    def process_hmdb51_activities(self):
        """Process HMDB-51 activities and map to daily activities"""
        print(f"\n🔄 Processing HMDB-51 activities...")
        
        # Find HMDB-51 video directories
        hmdb_video_dirs = []
        for item in self.hmdb_root.rglob("*"):
            if item.is_dir() and any(video_file.suffix.lower() in ['.avi', '.mp4'] 
                                   for video_file in item.iterdir() if video_file.is_file()):
                hmdb_video_dirs.append(item)
        
        if not hmdb_video_dirs:
            print(f"❌ No HMDB-51 video directories found")
            return False
        
        print(f"📁 Found {len(hmdb_video_dirs)} HMDB-51 activity directories")
        
        # Map activities to daily activities
        activity_mapping = {}
        video_counts = defaultdict(int)
        
        for video_dir in hmdb_video_dirs:
            activity_name = video_dir.name.lower()
            
            # Find matching daily activity (exact match first, then substring)
            daily_activity = None
            
            # First try exact match
            if activity_name in self.hmdb_to_daily:
                daily_activity = self.hmdb_to_daily[activity_name]
            else:
                # Then try substring match (but prioritize longer matches)
                best_match = ""
                for hmdb_name, daily_name in self.hmdb_to_daily.items():
                    if hmdb_name in activity_name and len(hmdb_name) > len(best_match):
                        daily_activity = daily_name
                        best_match = hmdb_name
            
            if daily_activity:
                activity_mapping[activity_name] = daily_activity
                
                # Count videos
                video_files = [f for f in video_dir.iterdir() 
                             if f.suffix.lower() in ['.avi', '.mp4']]
                video_counts[daily_activity] += len(video_files)
                
                print(f"  ✅ {activity_name}: {len(video_files)} videos -> {daily_activity}")
            else:
                print(f"  ⚠️  {activity_name}: No mapping found, skipping")
        
        # Save mapping
        mapping_file = self.hmdb_root / "activity_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                'hmdb_to_daily': activity_mapping,
                'video_counts': dict(video_counts),
                'total_videos': sum(video_counts.values())
            }, f, indent=2)
        
        print(f"\n📊 HMDB-51 Processing Summary:")
        print(f"  • Mapped activities: {len(activity_mapping)}")
        print(f"  • Total videos: {sum(video_counts.values())}")
        print(f"  • Daily activities covered: {len(video_counts)}")
        
        for activity, count in sorted(video_counts.items()):
            print(f"    - {activity}: {count} videos")
        
        return True
    
    def create_unified_dataset(self):
        """Create unified UCF-101 + HMDB-51 dataset"""
        print(f"\n🔗 Creating unified dataset...")
        
        # Create unified dataset directory
        self.integrated_root.mkdir(exist_ok=True)
        
        # UCF-101 activity name mapping
        ucf_name_mapping = {
            "WalkingWithDog": "walking_dog",
            "Swimming": "swimming", 
            "Biking": "biking",
            "Lifting": "weight_lifting",
            "JumpingJack": "jumping_jacks",
            "Pullups": "pullups",
            "Pushups": "pushups", 
            "Typing": "typing",
            "BrushingTeeth": "brushing_teeth"
        }
        
        dataset_info = {
            'ucf_activities': {},
            'hmdb_activities': {},
            'combined_stats': defaultdict(int)
        }
        
        # Process UCF-101 activities
        print(f"📁 Processing UCF-101 activities...")
        if self.ucf_root.exists():
            for ucf_activity in self.ucf_activities:
                ucf_dir = self.ucf_root / ucf_activity
                if ucf_dir.exists():
                    daily_name = ucf_name_mapping.get(ucf_activity, ucf_activity.lower())
                    
                    # Create symlinks or copy files
                    target_dir = self.integrated_root / daily_name
                    target_dir.mkdir(exist_ok=True)
                    
                    # Copy/link video files
                    video_files = list(ucf_dir.glob("*.avi"))
                    for video_file in video_files:
                        target_file = target_dir / f"ucf_{video_file.name}"
                        if not target_file.exists():
                            shutil.copy2(video_file, target_file)
                    
                    dataset_info['ucf_activities'][daily_name] = len(video_files)
                    dataset_info['combined_stats'][daily_name] += len(video_files)
                    
                    print(f"  ✅ {ucf_activity} -> {daily_name}: {len(video_files)} videos")
        
        # Process HMDB-51 activities
        print(f"📁 Processing HMDB-51 activities...")
        mapping_file = self.hmdb_root / "activity_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            hmdb_to_daily = mapping_data['hmdb_to_daily']
            
            # Find and process each HMDB activity
            for hmdb_dir in self.hmdb_root.iterdir():
                if hmdb_dir.is_dir() and hmdb_dir.name.lower() in hmdb_to_daily:
                    daily_name = hmdb_to_daily[hmdb_dir.name.lower()]
                    
                    # Create target directory
                    target_dir = self.integrated_root / daily_name
                    target_dir.mkdir(exist_ok=True)
                    
                    # Look for video files in nested directories (HMDB structure)
                    video_files = []
                    for item in hmdb_dir.rglob("*.avi"):
                        if item.is_file():
                            video_files.append(item)
                    
                    # Copy video files
                    copied_count = 0
                    for video_file in video_files:
                        target_file = target_dir / f"hmdb_{video_file.name}"
                        if not target_file.exists():
                            shutil.copy2(video_file, target_file)
                            copied_count += 1
                    
                    # Update counts
                    current_count = dataset_info['hmdb_activities'].get(daily_name, 0)
                    dataset_info['hmdb_activities'][daily_name] = current_count + copied_count
                    dataset_info['combined_stats'][daily_name] += copied_count
                    
                    print(f"  ✅ {hmdb_dir.name} -> {daily_name}: {copied_count} videos")
        
        # Save dataset info
        info_file = self.integrated_root / "dataset_info.json"
        dataset_info['total_videos'] = sum(dataset_info['combined_stats'].values())
        dataset_info['total_activities'] = len(dataset_info['combined_stats'])
        
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create train/test splits
        self.create_train_test_splits()
        
        print(f"\n✅ Unified dataset created!")
        print(f"📊 Summary:")
        print(f"  • Total activities: {dataset_info['total_activities']}")
        print(f"  • Total videos: {dataset_info['total_videos']}")
        print(f"  • UCF-101 contribution: {sum(dataset_info['ucf_activities'].values())} videos")
        print(f"  • HMDB-51 contribution: {sum(dataset_info['hmdb_activities'].values())} videos")
        
        return True
    
    def create_train_test_splits(self):
        """Create train/test splits for the unified dataset"""
        print(f"\n📝 Creating train/test splits...")
        
        splits = {'train': [], 'test': []}
        
        for activity_dir in self.integrated_root.iterdir():
            if activity_dir.is_dir() and activity_dir.name != '__pycache__':
                activity_name = activity_dir.name
                video_files = list(activity_dir.glob("*.avi"))
                
                # 80/20 split
                train_count = int(0.8 * len(video_files))
                
                for i, video_file in enumerate(video_files):
                    relative_path = f"{activity_name}/{video_file.name}"
                    
                    if i < train_count:
                        splits['train'].append(f"{relative_path} {activity_name}")
                    else:
                        splits['test'].append(f"{relative_path} {activity_name}")
        
        # Save splits
        for split_name, split_data in splits.items():
            split_file = self.integrated_root / f"{split_name}_split.txt"
            with open(split_file, 'w') as f:
                for line in split_data:
                    f.write(line + '\n')
            
            print(f"  ✅ {split_name}: {len(split_data)} videos")
        
        return True
    
    def run_full_integration(self):
        """Run the complete integration process"""
        print(f"🚀 Starting HMDB-51 Full Integration Process")
        print("=" * 50)
        
        steps = [
            ("Download HMDB-51", self.download_hmdb51),
            ("Extract HMDB-51", self.extract_hmdb51), 
            ("Process Activities", self.process_hmdb51_activities),
            ("Create Unified Dataset", self.create_unified_dataset)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 Step: {step_name}")
            success = step_func()
            
            if not success:
                print(f"❌ Failed at step: {step_name}")
                return False
            
            print(f"✅ Completed: {step_name}")
        
        print(f"\n🎉 HMDB-51 Integration Complete!")
        print(f"📁 Unified dataset: {self.integrated_root}")
        
        # Final summary
        info_file = self.integrated_root / "dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            print(f"\n📊 Final Dataset Summary:")
            print(f"  • Activities: {info['total_activities']}")
            print(f"  • Total videos: {info['total_videos']}")
            print(f"  • Ready for Phase 4 training!")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='HMDB-51 Integration System')
    parser.add_argument('--download', action='store_true', help='Download HMDB-51 dataset')
    parser.add_argument('--extract', action='store_true', help='Extract HMDB-51 RAR file')
    parser.add_argument('--process', action='store_true', help='Process HMDB-51 activities')
    parser.add_argument('--integrate', action='store_true', help='Create unified dataset')
    parser.add_argument('--full', action='store_true', help='Run complete integration process')
    
    args = parser.parse_args()
    
    integrator = HMDB51Integrator()
    
    if args.full:
        integrator.run_full_integration()
    elif args.download:
        integrator.download_hmdb51()
    elif args.extract:
        integrator.extract_hmdb51()
    elif args.process:
        integrator.process_hmdb51_activities()
    elif args.integrate:
        integrator.create_unified_dataset()
    else:
        print("🎯 HMDB-51 Integration System")
        print("=" * 30)
        print()
        print("Available commands:")
        print("  --full      : Run complete integration process")
        print("  --download  : Download HMDB-51 dataset only")
        print("  --extract   : Extract RAR file only")
        print("  --process   : Process activities only")
        print("  --integrate : Create unified dataset only")
        print()
        print("Recommended usage:")
        print("  python setup_hmdb51_integration.py --full")

if __name__ == "__main__":
    main()