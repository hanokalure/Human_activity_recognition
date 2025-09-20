"""
Phase 3: Unified Daily Activities Class Mapping
==============================================

Maps 19 frequent daily human activities to Kinetics-400 classes.
All activities sourced from Kinetics-400 for consistent data quality.
Optimized for R(2+1)D-34 model training.
"""

# Phase 3: 19 Daily Activities (your selected activities)
PHASE3_ACTIVITIES = [
    "eating",           # 0
    "drinking",         # 1  
    "cooking",          # 2
    "sleeping",         # 3
    "sitting",          # 4
    "walking",          # 5
    "typing",           # 6
    "reading",          # 7
    "using_computer",   # 8
    "driving",          # 9
    "biking",           # 10
    "brushing_teeth",   # 11
    "running",          # 12
    "yoga",             # 13
    "exercising",       # 14
    "stretching",       # 15
    "watching_tv",      # 16
    "playing_games",    # 17
    "dancing"           # 18
]

# Kinetics-400 class ID mapping for Phase 3 activities
KINETICS_TO_PHASE3 = {
    # Eating - Multiple Kinetics classes mapped to single activity
    36: "eating",    # eating burger
    37: "eating",    # eating cake
    38: "eating",    # eating chips
    39: "eating",    # eating doughnuts
    40: "eating",    # eating hotdog
    41: "eating",    # eating ice cream
    42: "eating",    # eating spaghetti
    43: "eating",    # eating watermelon
    
    # Drinking
    106: "drinking", # drinking
    107: "drinking", # drinking beer
    108: "drinking", # drinking shots
    
    # Cooking
    69: "cooking",   # cooking chicken
    70: "cooking",   # cooking egg
    71: "cooking",   # cooking on campfire
    72: "cooking",   # cooking sausages
    
    # Sleeping
    319: "sleeping", # sleeping
    
    # Sitting
    310: "sitting",  # sitting
    
    # Walking
    378: "walking",  # walking
    379: "walking",  # walking the dog (map to general walking)
    
    # Typing
    374: "typing",   # typing
    
    # Reading
    272: "reading",  # reading book
    273: "reading",  # reading newspaper
    
    # Using computer
    375: "using_computer",  # texting
    
    # Driving  
    104: "driving",  # driving car
    105: "driving",  # driving tractor
    
    # Biking
    29: "biking",    # biking through snow
    30: "biking",    # biking
    
    # Brushing teeth
    48: "brushing_teeth", # brushing teeth
    
    # Running
    269: "running",  # running
    270: "running",  # running on treadmill
    171: "running",  # jogging
    
    # Yoga
    391: "yoga",     # yoga
    392: "yoga",     # zumba (yoga-like)
    
    # Exercising 
    136: "exercising", # doing aerobics
    137: "exercising", # doing nails (self-care exercise)
    165: "exercising", # exercising arm
    166: "exercising", # exercising with an exercise ball
    
    # Stretching
    336: "stretching", # stretching arm
    337: "stretching", # stretching leg
    
    # Watching TV
    380: "watching_tv", # watching tv
    
    # Playing games
    244: "playing_games", # playing cards
    245: "playing_games", # playing chess  
    246: "playing_games", # playing cricket
    247: "playing_games", # playing cymbals
    248: "playing_games", # playing didgeridoo
    249: "playing_games", # playing drums
    250: "playing_games", # playing flute
    251: "playing_games", # playing guitar
    252: "playing_games", # playing harmonica
    253: "playing_games", # playing harp
    254: "playing_games", # playing keyboard
    255: "playing_games", # playing kickball
    256: "playing_games", # playing monopoly
    257: "playing_games", # playing organ
    258: "playing_games", # playing piano
    259: "playing_games", # playing poker
    260: "playing_games", # playing recorder
    261: "playing_games", # playing saxophone
    262: "playing_games", # playing squash or racquetball
    263: "playing_games", # playing tennis
    264: "playing_games", # playing trombone
    265: "playing_games", # playing trumpet
    266: "playing_games", # playing ukulele
    267: "playing_games", # playing violin
    268: "playing_games", # playing volleyball
    
    # Dancing
    80: "dancing",   # dancing ballet
    81: "dancing",   # dancing charleston
    82: "dancing",   # dancing gangnam style
    83: "dancing",   # dancing macarena
    84: "dancing",   # dancing
}

def get_phase3_activities():
    """Get list of Phase 3 activities"""
    return PHASE3_ACTIVITIES.copy()

def get_phase3_class_mapping():
    """Get mapping from Kinetics class IDs to Phase 3 activities"""
    return KINETICS_TO_PHASE3.copy()

def get_phase3_kinetics_classes():
    """Get list of Kinetics class IDs needed for Phase 3"""
    return list(KINETICS_TO_PHASE3.keys())

def create_activity_to_index_mapping():
    """Create mapping from activity names to class indices"""
    return {activity: idx for idx, activity in enumerate(PHASE3_ACTIVITIES)}

def print_phase3_mapping():
    """Print detailed Phase 3 class mapping"""
    print("\n🎯 PHASE 3: Unified Daily Activities Mapping")
    print("=" * 55)
    
    print(f"\n📊 Total Activities: {len(PHASE3_ACTIVITIES)}")
    for i, activity in enumerate(PHASE3_ACTIVITIES):
        print(f"  {i:2d}. {activity}")
    
    print(f"\n🔗 Kinetics Classes Used: {len(set(KINETICS_TO_PHASE3.keys()))}")
    
    # Group by activity
    activity_mapping = {}
    for kinetics_id, activity in KINETICS_TO_PHASE3.items():
        if activity not in activity_mapping:
            activity_mapping[activity] = []
        activity_mapping[activity].append(kinetics_id)
    
    print("\n📋 Detailed Mapping:")
    for activity in PHASE3_ACTIVITIES:
        if activity in activity_mapping:
            kinetics_ids = activity_mapping[activity]
            print(f"  • {activity}: {kinetics_ids} ({len(kinetics_ids)} classes)")
        else:
            print(f"  • {activity}: No mapping found ❌")
    
    print(f"\n✅ All activities mapped to Kinetics-400 classes")
    print(f"🎯 Ready for R(2+1)D-34 training!")

if __name__ == "__main__":
    print_phase3_mapping()