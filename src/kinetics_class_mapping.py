"""
Kinetics-400 Class Mapping for Phase 2 Daily Activities
======================================================

Maps official Kinetics-400 class IDs to our 10 target daily activities.
Based on the official Kinetics-400 dataset class list.
"""

# Official Kinetics-400 class mapping (0-399 indexed)
# From: https://github.com/deepmind/kinetics-i3d/blob/master/data/label_map.txt

KINETICS_400_CLASSES = {
    # Our target activities mapped to Kinetics class IDs
    36: "eating_burger",
    37: "eating_cake", 
    38: "eating_chips",
    39: "eating_doughnuts",
    40: "eating_hotdog",
    41: "eating_ice_cream",
    42: "eating_spaghetti",
    43: "eating_watermelon",
    
    # Cooking activities
    69: "cooking_chicken",
    70: "cooking_egg",
    71: "cooking_on_campfire",
    
    # Reading activities  
    272: "reading_book",
    273: "reading_newspaper",
    
    # Sleeping
    319: "sleeping",
    
    # Cleaning activities
    62: "cleaning_floor",
    63: "cleaning_gutters", 
    64: "cleaning_windows",
    225: "mopping_floor",
    
    # Exercise/Yoga
    136: "doing_aerobics",
    391: "yoga",
    336: "stretching_arm",
    337: "stretching_leg",
    
    # Walking/Running
    378: "walking",
    379: "walking_the_dog", 
    269: "running",
    270: "running_on_treadmill",
    171: "jogging",
    
    # Driving
    104: "driving_car",
    105: "driving_tractor",
    
    # Sitting
    310: "sitting"
}

# Map Kinetics classes to our daily activities
KINETICS_TO_DAILY_ACTIVITIES = {
    # Eating (multiple Kinetics classes -> single daily activity)
    36: "eating",   # eating_burger
    37: "eating",   # eating_cake
    38: "eating",   # eating_chips  
    39: "eating",   # eating_doughnuts
    40: "eating",   # eating_hotdog
    41: "eating",   # eating_ice_cream
    42: "eating",   # eating_spaghetti
    43: "eating",   # eating_watermelon
    
    # Cooking
    69: "cooking",  # cooking_chicken
    70: "cooking",  # cooking_egg
    71: "cooking",  # cooking_on_campfire
    
    # Reading  
    272: "reading", # reading_book
    273: "reading", # reading_newspaper
    
    # Sleeping
    319: "sleeping", # sleeping
    
    # Cleaning
    62: "cleaning",  # cleaning_floor
    63: "cleaning",  # cleaning_gutters
    64: "cleaning",  # cleaning_windows  
    225: "cleaning", # mopping_floor
    
    # Yoga (combine aerobics and yoga into single class)
    136: "yoga",      # doing_aerobics
    391: "yoga",      # yoga
    
    # Walking
    378: "walking",     # walking
    379: "walking_dog", # walking_the_dog (keep separate as Phase 1 has this)
    
    # Running  
    269: "running",     # running
    171: "running",     # jogging -> running
    
    # Driving
    104: "driving",  # driving_car
    105: "driving",  # driving_tractor
    
    # Sitting
    310: "sitting"   # sitting
}

# Phase 1 activities (from UCF-101)
PHASE_1_ACTIVITIES = [
    "walking_dog",      # walking_with_dog
    "swimming",         # swimming activities  
    "biking",          # cycling activities
    "weight_lifting",   # bench pressing
    "jumping_jacks",    # jumping jack exercises
    "pullups",         # pull-up exercises
    "pushups",         # push-up exercises
    "typing",          # typing on keyboard
    "brushing_teeth"   # dental hygiene
]

# Phase 2 activities (from Kinetics-400) - exactly 10 as requested 
PHASE_2_ACTIVITIES = [
    "eating",
    "cooking", 
    "reading",
    "sleeping",
    "cleaning",
    "yoga",
    "walking",
    "running",
    "driving",
    "sitting"
]

# Combined 19 classes for unified model
ALL_DAILY_ACTIVITIES = PHASE_1_ACTIVITIES + PHASE_2_ACTIVITIES

def get_kinetics_class_mapping():
    """Get mapping from Kinetics class IDs to daily activities"""
    return KINETICS_TO_DAILY_ACTIVITIES

def get_phase_2_target_classes():
    """Get the Kinetics class IDs we need for Phase 2"""
    return list(KINETICS_TO_DAILY_ACTIVITIES.keys())

def get_all_activities():
    """Get all 19 daily activities (Phase 1 + Phase 2)"""
    return ALL_DAILY_ACTIVITIES

def print_class_mapping_summary():
    """Print summary of class mappings"""
    print("\n🎯 Phase 2 Class Mapping Summary")
    print("=" * 50)
    
    print(f"\n📊 Phase 1 Activities: {len(PHASE_1_ACTIVITIES)}")
    for i, activity in enumerate(PHASE_1_ACTIVITIES, 1):
        print(f"  {i:2d}. {activity}")
    
    print(f"\n📊 Phase 2 Activities: {len(PHASE_2_ACTIVITIES)}")
    for i, activity in enumerate(PHASE_2_ACTIVITIES, 1):
        print(f"  {i:2d}. {activity}")
        
    print(f"\n📊 Total Activities: {len(ALL_DAILY_ACTIVITIES)}")
    
    print(f"\n🔗 Kinetics Classes Needed:")
    kinetics_classes = {}
    for class_id, activity in KINETICS_TO_DAILY_ACTIVITIES.items():
        if activity not in kinetics_classes:
            kinetics_classes[activity] = []
        kinetics_classes[activity].append(class_id)
    
    for activity, class_ids in kinetics_classes.items():
        print(f"  • {activity}: {class_ids}")

if __name__ == "__main__":
    print_class_mapping_summary()