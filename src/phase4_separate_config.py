"""
Phase 4 Separate Training Configuration
=====================================

Final configuration for 17 separate daily life activities.
No mixing of UCF-101 and HMDB-51 classes for higher accuracy.

Target: 80+ accuracy with separate, high-quality classes.
"""

import os
import json
from pathlib import Path
import shutil
from collections import defaultdict

class Phase4SeparateConfig:
    """Configuration for Phase 4 separate training"""
    
    def __init__(self):
        self.project_root = Path("C:/ASH_PROJECT")
        self.data_root = self.project_root / "data"
        self.ucf_root = self.data_root / "UCF101"
        self.hmdb_root = self.data_root / "HMDB51"
        self.phase4_root = self.data_root / "Phase4_Separate"
        
        # Final 17 activities configuration
        self.final_activities = {
            # UCF-101 Classes (10 activities)
            "ucf101_classes": {
                "brushing_teeth": "BrushingTeeth",
                "typing": "Typing",
                "biking_ucf": "Biking", 
                "pullups_ucf": "PullUps",
                "pushups_ucf": "PushUps",
                "writing": "WritingOnBoard",
                "walking_dog": "WalkingWithDog",
                "cooking_ucf": "CuttingInKitchen",
                "breast_stroke": "BreastStroke",
                "front_crawl": "FrontCrawl"
            },
            
            # HMDB-51 Classes (7 activities) 
            "hmdb51_classes": {
                "walking": "walk",
                "running": "run",
                "sitting": "sit",
                "eating": "eat", 
                "brushing_hair": "brush_hair",
                "talking": "talk",
                "pouring": "pour"
            }
        }
        
        print(f"🎯 Phase 4 Separate Configuration Initialized")
        print(f"📁 UCF-101 classes: {len(self.final_activities['ucf101_classes'])}")
        print(f"📁 HMDB-51 classes: {len(self.final_activities['hmdb51_classes'])}")
        print(f"📊 Total classes: {len(self.final_activities['ucf101_classes']) + len(self.final_activities['hmdb51_classes'])}")
    
    def create_separate_dataset(self):
        """Create Phase 4 separate dataset"""
        print(f"\n🔄 Creating Phase 4 Separate Dataset...")
        
        # Create Phase 4 directory
        self.phase4_root.mkdir(exist_ok=True)
        
        dataset_stats = {
            'ucf101_stats': {},
            'hmdb51_stats': {},
            'total_videos': 0,
            'activities': []
        }
        
        # Process UCF-101 classes
        print(f"\n📁 Processing UCF-101 classes...")
        for class_name, ucf_activity in self.final_activities['ucf101_classes'].items():
            ucf_dir = self.ucf_root / ucf_activity
            
            if ucf_dir.exists():
                # Create target directory
                target_dir = self.phase4_root / class_name
                target_dir.mkdir(exist_ok=True)
                
                # Copy videos
                video_files = list(ucf_dir.glob("*.avi"))
                copied_count = 0
                
                for video_file in video_files:
                    target_file = target_dir / f"ucf_{video_file.name}"
                    if not target_file.exists():
                        shutil.copy2(video_file, target_file)
                        copied_count += 1
                
                dataset_stats['ucf101_stats'][class_name] = copied_count
                dataset_stats['total_videos'] += copied_count
                dataset_stats['activities'].append(class_name)
                
                print(f"  ✅ {class_name} ({ucf_activity}): {copied_count} videos")
            else:
                print(f"  ❌ {ucf_activity} directory not found")
        
        # Process HMDB-51 classes  
        print(f"\n📁 Processing HMDB-51 classes...")
        for class_name, hmdb_activity in self.final_activities['hmdb51_classes'].items():
            hmdb_dir = self.hmdb_root / hmdb_activity
            
            if hmdb_dir.exists():
                # Create target directory
                target_dir = self.phase4_root / class_name
                target_dir.mkdir(exist_ok=True)
                
                # Find videos in nested structure
                video_files = []
                for item in hmdb_dir.rglob("*.avi"):
                    if item.is_file():
                        video_files.append(item)
                
                # Copy videos
                copied_count = 0
                for video_file in video_files:
                    target_file = target_dir / f"hmdb_{video_file.name}"
                    if not target_file.exists():
                        shutil.copy2(video_file, target_file)
                        copied_count += 1
                
                dataset_stats['hmdb51_stats'][class_name] = copied_count
                dataset_stats['total_videos'] += copied_count
                dataset_stats['activities'].append(class_name)
                
                print(f"  ✅ {class_name} ({hmdb_activity}): {copied_count} videos")
            else:
                print(f"  ❌ {hmdb_activity} directory not found")
        
        # Create train/test splits
        self.create_train_test_splits()
        
        # Save dataset statistics
        stats_file = self.phase4_root / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        print(f"\n✅ Phase 4 Separate Dataset Created!")
        print(f"📊 Statistics:")
        print(f"   • Total activities: {len(dataset_stats['activities'])}")
        print(f"   • Total videos: {dataset_stats['total_videos']}")
        print(f"   • UCF-101 videos: {sum(dataset_stats['ucf101_stats'].values())}")
        print(f"   • HMDB-51 videos: {sum(dataset_stats['hmdb51_stats'].values())}")
        
        return dataset_stats
    
    def create_train_test_splits(self):
        """Create train/test splits for Phase 4 separate dataset"""
        print(f"\n📝 Creating train/test splits...")
        
        splits = {'train': [], 'test': []}
        
        for activity_dir in self.phase4_root.iterdir():
            if activity_dir.is_dir() and activity_dir.name not in ['__pycache__']:
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
            split_file = self.phase4_root / f"{split_name}_split.txt"
            with open(split_file, 'w') as f:
                for line in split_data:
                    f.write(line + '\n')
            
            print(f"  ✅ {split_name}: {len(split_data)} videos")
        
        return splits

def main():
    """Main function to create Phase 4 separate dataset"""
    config = Phase4SeparateConfig()
    stats = config.create_separate_dataset()
    
    print(f"\n🎉 Phase 4 Separate Dataset Ready!")
    print(f"📁 Location: {config.phase4_root}")
    print(f"🚀 Ready for training with 17 separate activities!")
    
    return stats

if __name__ == "__main__":
    main()