"""
Daily Activities Configuration
=============================

This module defines the curated set of daily activities we'll focus on,
mapping them from UCF-101 and Kinetics-400 datasets for high accuracy
human activity recognition.

Target: 85-90% accuracy on 22 daily activity classes
"""

# Our curated daily activities (22 classes for focused accuracy)
DAILY_ACTIVITIES = [
    # Indoor Activities (7 classes)
    "brushing_teeth",
    "typing", 
    "eating",
    "cooking",
    "reading",
    "sleeping",
    "cleaning",
    
    # Fitness/Exercise (8 classes) 
    "pushups",
    "pullups",
    "squats",
    "jumping_jacks",
    "yoga",
    "weight_lifting",
    "running_treadmill",
    "stretching",
    
    # Outdoor/General (7 classes)
    "walking", 
    "running",
    "biking",
    "swimming", 
    "walking_dog",
    "driving",
    "sitting"
]

# Mapping from UCF-101 classes to our daily activities
UCF101_TO_DAILY = {
    # Direct matches from UCF-101
    "BrushingTeeth": "brushing_teeth",
    "Typing": "typing",
    "CuttingInKitchen": "cooking",
    "MoppingFloor": "cleaning",
    "PushUps": "pushups", 
    "PullUps": "pullups",
    "BodyWeightSquats": "squats",
    "JumpingJack": "jumping_jacks",
    "TaiChi": "yoga",  # Close to yoga/stretching
    "BenchPress": "weight_lifting",
    "WalkingWithDog": "walking_dog",
    "Biking": "biking",
    "Swimming": "swimming",  # We'll use BreastStroke/FrontCrawl as swimming
    "BreastStroke": "swimming",
    "FrontCrawl": "swimming",
}

# Kinetics-400 classes we want to download (these have better daily activity coverage)
KINETICS_DAILY_ACTIVITIES = [
    # Indoor activities
    "eating_burger", "eating_cake", "eating_chips", "eating_doughnuts",  # -> eating
    "cooking_chicken", "cooking_egg", "cooking_on_campfire",  # -> cooking  
    "reading_book", "reading_newspaper",  # -> reading
    "sleeping",  # -> sleeping
    "texting", "using_computer", "typing_on_computer",  # -> typing
    "brushing_teeth",  # -> brushing_teeth
    "mopping_floor", "cleaning_windows", "cleaning_gutters",  # -> cleaning
    
    # Exercise/Fitness  
    "doing_aerobics", "exercising_with_an_exercise_ball", "stretching_arm", "stretching_leg",  # -> stretching/yoga
    "doing_laundry", "push_up", "pull_ups", "squat", "lunges",  # -> fitness
    "jumping_jacks", "exercising_arm", "yoga",  # -> exercise
    "bench_pressing", "deadlifting", "clean_and_jerk",  # -> weight_lifting
    "running_on_treadmill", "walking_the_dog",  # -> specific activities
    
    # Outdoor/General
    "walking", "jogging", "running", "marching",  # -> walking/running
    "cycling_on_stationary_bike", "biking_through_snow", "mountain_biking",  # -> biking  
    "swimming_backstroke", "swimming_breast_stroke", "swimming_butterfly_stroke", "swimming_front_crawl",  # -> swimming
    "driving_car", "driving_tractor",  # -> driving
    "sitting_in_chair", "getting_a_haircut",  # -> sitting
]

# Mapping from Kinetics to our daily activities
KINETICS_TO_DAILY = {
    # Eating activities -> eating
    "eating_burger": "eating", "eating_cake": "eating", "eating_chips": "eating", 
    "eating_doughnuts": "eating", "eating_spaghetti": "eating",
    
    # Cooking activities -> cooking  
    "cooking_chicken": "cooking", "cooking_egg": "cooking", "cooking_on_campfire": "cooking",
    "making_pizza": "cooking", "making_sandwich": "cooking",
    
    # Reading -> reading
    "reading_book": "reading", "reading_newspaper": "reading",
    
    # Technology use -> typing
    "texting": "typing", "using_computer": "typing", "typing_on_computer": "typing",
    
    # Cleaning -> cleaning
    "mopping_floor": "cleaning", "cleaning_windows": "cleaning", "cleaning_gutters": "cleaning",
    
    # Exercise activities
    "doing_aerobics": "stretching", "exercising_with_an_exercise_ball": "stretching",
    "stretching_arm": "stretching", "stretching_leg": "stretching", "yoga": "yoga",
    "push_up": "pushups", "pull_ups": "pullups", "squat": "squats", "lunges": "squats",
    "jumping_jacks": "jumping_jacks", "exercising_arm": "stretching",
    "bench_pressing": "weight_lifting", "deadlifting": "weight_lifting", "clean_and_jerk": "weight_lifting",
    "running_on_treadmill": "running_treadmill",
    
    # Movement activities
    "walking": "walking", "jogging": "running", "running": "running", "marching": "walking",
    "walking_the_dog": "walking_dog",
    
    # Biking
    "cycling_on_stationary_bike": "biking", "biking_through_snow": "biking", "mountain_biking": "biking",
    
    # Swimming  
    "swimming_backstroke": "swimming", "swimming_breast_stroke": "swimming",
    "swimming_butterfly_stroke": "swimming", "swimming_front_crawl": "swimming",
    
    # Other
    "driving_car": "driving", "driving_tractor": "driving",
    "sitting_in_chair": "sitting", "getting_a_haircut": "sitting",
    "sleeping": "sleeping", "brushing_teeth": "brushing_teeth",
}

# Simple text descriptions for each activity (for video-to-text output)
ACTIVITY_DESCRIPTIONS = {
    "brushing_teeth": "Person is brushing teeth",
    "typing": "Person is typing",
    "eating": "Person is eating", 
    "cooking": "Person is cooking",
    "reading": "Person is reading",
    "sleeping": "Person is sleeping",
    "cleaning": "Person is cleaning",
    "pushups": "Person is doing push-ups",
    "pullups": "Person is doing pull-ups", 
    "squats": "Person is doing squats",
    "jumping_jacks": "Person is doing jumping jacks",
    "yoga": "Person is doing yoga",
    "weight_lifting": "Person is lifting weights",
    "running_treadmill": "Person is running on treadmill",
    "stretching": "Person is stretching",
    "walking": "Person is walking",
    "running": "Person is running", 
    "biking": "Person is biking",
    "swimming": "Person is swimming",
    "walking_dog": "Person is walking with dog",
    "driving": "Person is driving",
    "sitting": "Person is sitting"
}

def get_daily_activities_info():
    """Get summary of daily activities configuration"""
    info = {
        "total_classes": len(DAILY_ACTIVITIES),
        "indoor_activities": 7,
        "fitness_activities": 8, 
        "outdoor_activities": 7,
        "ucf101_mappings": len(UCF101_TO_DAILY),
        "kinetics_classes": len(KINETICS_DAILY_ACTIVITIES),
        "kinetics_mappings": len(KINETICS_TO_DAILY)
    }
    return info

def print_daily_activities_summary():
    """Print a nice summary of our daily activities setup"""
    print("\n🎯 Daily Activities Recognition System")
    print("=" * 50)
    
    print(f"\n📊 Total Classes: {len(DAILY_ACTIVITIES)}")
    
    print(f"\n🏠 Indoor Activities ({len([a for a in DAILY_ACTIVITIES if a in ['brushing_teeth', 'typing', 'eating', 'cooking', 'reading', 'sleeping', 'cleaning']])}):")
    for activity in ["brushing_teeth", "typing", "eating", "cooking", "reading", "sleeping", "cleaning"]:
        print(f"  • {activity}")
    
    print(f"\n💪 Fitness Activities ({len([a for a in DAILY_ACTIVITIES if a in ['pushups', 'pullups', 'squats', 'jumping_jacks', 'yoga', 'weight_lifting', 'running_treadmill', 'stretching']])}):")
    for activity in ["pushups", "pullups", "squats", "jumping_jacks", "yoga", "weight_lifting", "running_treadmill", "stretching"]:
        print(f"  • {activity}")
        
    print(f"\n🚶 Outdoor/General ({len([a for a in DAILY_ACTIVITIES if a in ['walking', 'running', 'biking', 'swimming', 'walking_dog', 'driving', 'sitting']])}):")  
    for activity in ["walking", "running", "biking", "swimming", "walking_dog", "driving", "sitting"]:
        print(f"  • {activity}")
    
    print(f"\n📈 Data Sources:")
    print(f"  • UCF-101 mappings: {len(UCF101_TO_DAILY)} classes")
    print(f"  • Kinetics-400 classes: {len(KINETICS_DAILY_ACTIVITIES)} classes") 
    print(f"  • Total unique activities: {len(DAILY_ACTIVITIES)}")
    
    print(f"\n🎯 Target Accuracy: 85-90% on focused daily activities")
    print("=" * 50)

if __name__ == "__main__":
    print_daily_activities_summary()