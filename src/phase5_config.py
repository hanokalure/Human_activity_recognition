"""
Phase 5 Configuration - 25 Daily Life Activities
===============================================

Extends Phase 4 (17 classes) with 8 high-priority daily activities.
Total: 25 classes for comprehensive daily life recognition.

NEW Phase 5 Activities (8):
- climbing_stairs (HMDB-51: climb_stairs)
- hugging (HMDB-51: hug)
- waving (HMDB-51: wave)  
- laughing (HMDB-51: laugh)
- drinking (HMDB-51: drink)
- yoga (HMDB-51: situp)
- cleaning (UCF-101: MoppingFloor)
- weight_lifting (UCF-101: Lifting)

Target: 80+ accuracy with 25 separate activities.
"""

import os
import json
from pathlib import Path
import shutil
from collections import defaultdict

class Phase5Config:
    """Configuration for Phase 5 - 25 activities training"""
    
    def __init__(self):
        self.project_root = Path("C:/ASH_PROJECT")
        self.data_root = self.project_root / "data"
        self.ucf_root = self.data_root / "UCF101"
        self.hmdb_root = self.data_root / "HMDB51"
        self.phase5_root = self.data_root / "Phase5_25Classes"
        
        # Phase 4 existing 17 activities (keeping same structure)
        self.phase4_activities = {
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
        
        # NEW Phase 5 activities (8 high-priority additions)
        self.phase5_new_activities = {
            # UCF-101 New Classes (2 activities)
            "ucf101_new": {
                "cleaning": "MoppingFloor",
                "weight_lifting": "weight_lifting"
            },
            
            # HMDB-51 New Classes (6 activities) 
            "hmdb51_new": {
                "climbing_stairs": "climb_stairs",
                "hugging": "hug",
                "waving": "wave",
                "laughing": "laugh", 
                "drinking": "drink",
                "yoga": "situp"
            }
        }
        
        # Combined all 25 activities
        self.all_activities = {
            **self.phase4_activities["ucf101_classes"],
            **self.phase4_activities["hmdb51_classes"],
            **self.phase5_new_activities["ucf101_new"],
            **self.phase5_new_activities["hmdb51_new"]
        }
        
        print(f"🎯 Phase 5 Configuration Initialized")
        print(f"📁 Phase 4 classes: 17 (10 UCF + 7 HMDB)")
        print(f"📁 Phase 5 new classes: 8 (2 UCF + 6 HMDB)")
        print(f"📊 Total classes: 25")
    
    def create_phase5_dataset(self):
        """Create Phase 5 dataset with 25 activities"""
        print(f"\n🔄 Creating Phase 5 Dataset (25 Classes)...")
        
        # Create Phase 5 directory
        self.phase5_root.mkdir(exist_ok=True)
        
        dataset_stats = {
            'phase4_stats': {'ucf101': {}, 'hmdb51': {}},
            'phase5_new_stats': {'ucf101': {}, 'hmdb51': {}},
            'total_videos': 0,
            'activities': []
        }
        
        # Copy Phase 4 existing activities
        print(f"\n📁 Processing Phase 4 existing activities...")
        
        # UCF-101 Phase 4 classes
        for class_name, ucf_activity in self.phase4_activities['ucf101_classes'].items():
            count = self._process_ucf_activity(class_name, ucf_activity)
            dataset_stats['phase4_stats']['ucf101'][class_name] = count
            dataset_stats['total_videos'] += count
            dataset_stats['activities'].append(class_name)
            print(f"  ✅ {class_name} ({ucf_activity}): {count} videos")
        
        # HMDB-51 Phase 4 classes  
        for class_name, hmdb_activity in self.phase4_activities['hmdb51_classes'].items():
            count = self._process_hmdb_activity(class_name, hmdb_activity)
            dataset_stats['phase4_stats']['hmdb51'][class_name] = count
            dataset_stats['total_videos'] += count
            dataset_stats['activities'].append(class_name)
            print(f"  ✅ {class_name} ({hmdb_activity}): {count} videos")
        
        # Add Phase 5 NEW activities
        print(f"\n📁 Processing Phase 5 NEW activities...")
        
        # UCF-101 NEW classes
        for class_name, ucf_activity in self.phase5_new_activities['ucf101_new'].items():
            count = self._process_ucf_activity(class_name, ucf_activity)
            dataset_stats['phase5_new_stats']['ucf101'][class_name] = count
            dataset_stats['total_videos'] += count
            dataset_stats['activities'].append(class_name)
            print(f"  🆕 {class_name} ({ucf_activity}): {count} videos")
        
        # HMDB-51 NEW classes
        for class_name, hmdb_activity in self.phase5_new_activities['hmdb51_new'].items():
            count = self._process_hmdb_activity(class_name, hmdb_activity)
            dataset_stats['phase5_new_stats']['hmdb51'][class_name] = count
            dataset_stats['total_videos'] += count
            dataset_stats['activities'].append(class_name)
            print(f"  🆕 {class_name} ({hmdb_activity}): {count} videos")
        
        # Create train/test splits
        self.create_train_test_splits()
        
        # Save dataset statistics
        stats_file = self.phase5_root / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        print(f"\n✅ Phase 5 Dataset Created!")
        print(f"📊 Statistics:")
        print(f"   • Total activities: 25")
        print(f"   • Total videos: {dataset_stats['total_videos']}")
        print(f"   • Phase 4 existing: {sum(dataset_stats['phase4_stats']['ucf101'].values()) + sum(dataset_stats['phase4_stats']['hmdb51'].values())} videos")
        print(f"   • Phase 5 new: {sum(dataset_stats['phase5_new_stats']['ucf101'].values()) + sum(dataset_stats['phase5_new_stats']['hmdb51'].values())} videos")
        
        return dataset_stats
    
    def _process_ucf_activity(self, class_name, ucf_activity):
        """Process UCF-101 activity"""
        ucf_dir = self.ucf_root / ucf_activity
        
        if ucf_dir.exists():
            # Create target directory
            target_dir = self.phase5_root / class_name
            target_dir.mkdir(exist_ok=True)
            
            # Copy videos
            video_files = list(ucf_dir.glob("*.avi"))
            copied_count = 0
            
            for video_file in video_files:
                target_file = target_dir / f"ucf_{video_file.name}"
                if not target_file.exists():
                    shutil.copy2(video_file, target_file)
                    copied_count += 1
            
            return copied_count
        else:
            print(f"  ❌ {ucf_activity} directory not found")
            return 0
    
    def _process_hmdb_activity(self, class_name, hmdb_activity):
        """Process HMDB-51 activity"""
        hmdb_dir = self.hmdb_root / hmdb_activity
        
        if hmdb_dir.exists():
            # Create target directory
            target_dir = self.phase5_root / class_name
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
            
            return copied_count
        else:
            print(f"  ❌ {hmdb_activity} directory not found")
            return 0
    
    def create_train_test_splits(self):
        """Create train/test splits for Phase 5 dataset"""
        print(f"\n📝 Creating train/test splits...")
        
        splits = {'train': [], 'test': []}
        
        for activity_dir in self.phase5_root.iterdir():
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
            split_file = self.phase5_root / f"{split_name}_split.txt"
            with open(split_file, 'w') as f:
                for line in split_data:
                    f.write(line + '\n')
            
            print(f"  ✅ {split_name}: {len(split_data)} videos")
        
        return splits

def main():
    """Main function to create Phase 5 dataset"""
    config = Phase5Config()
    stats = config.create_phase5_dataset()
    
    print(f"\n🎉 Phase 5 Dataset Ready!")
    print(f"📁 Location: {config.phase5_root}")
    print(f"🚀 Ready for training with 25 activities!")
    
    # List all 25 activities
    print(f"\n📋 All 25 Activities:")
    print(f"Phase 4 existing (17):")
    for i, activity in enumerate(sorted(list(config.phase4_activities['ucf101_classes'].keys()) + 
                                       list(config.phase4_activities['hmdb51_classes'].keys())), 1):
        print(f"  {i:2d}. {activity}")
    
    print(f"Phase 5 new (8):")
    for i, activity in enumerate(sorted(list(config.phase5_new_activities['ucf101_new'].keys()) + 
                                       list(config.phase5_new_activities['hmdb51_new'].keys())), 18):
        print(f"  {i:2d}. {activity}")
    
    return stats

if __name__ == "__main__":
    main()