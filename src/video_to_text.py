import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import json
from models_optimized import get_optimized_model
from dataset_optimized import OptimizedVideoTransform
import warnings
warnings.filterwarnings('ignore')

class VideoActionRecognizer:
    """Video-to-text action recognition system"""
    
    def __init__(self, model_path, model_type='efficient', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.transform = OptimizedVideoTransform(size=(112, 112))
        
        # Load UCF-101 class names
        self.class_names = self._load_ucf101_classes()
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Action descriptions mapping
        self.action_descriptions = self._create_action_descriptions()
        
        print(f"🚀 VideoActionRecognizer initialized")
        print(f"Device: {self.device}")
        print(f"Model: {model_type}")
        print(f"Classes: {len(self.class_names)}")
    
    def _load_ucf101_classes(self):
        """Load UCF-101 class names"""
        # UCF-101 class names (sorted alphabetically as they appear in the dataset)
        ucf101_classes = [
            'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
            'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
            'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
            'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
            'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
            'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
            'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering',
            'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
            'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow',
            'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting',
            'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
            'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf',
            'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
            'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
            'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing',
            'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding',
            'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
            'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot',
            'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
            'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard',
            'YoYo'
        ]
        return ucf101_classes
    
    def _load_model(self, model_path):
        """Load trained model"""
        # Create model
        model = get_optimized_model(
            model_type=self.model_type,
            num_classes=len(self.class_names),
            pretrained=False
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _create_action_descriptions(self):
        """Create human-readable descriptions for actions"""
        descriptions = {
            'ApplyEyeMakeup': "A person is applying eye makeup",
            'ApplyLipstick': "A person is applying lipstick",
            'Archery': "A person is practicing archery, shooting arrows at a target",
            'BabyCrawling': "A baby is crawling on the floor",
            'BalanceBeam': "A gymnast is performing on the balance beam",
            'BandMarching': "A marching band is performing",
            'BaseballPitch': "A baseball player is pitching the ball",
            'Basketball': "People are playing basketball",
            'BasketballDunk': "A basketball player is dunking the ball",
            'BenchPress': "A person is doing bench press exercise",
            'Biking': "A person is riding a bicycle",
            'Billiards': "People are playing billiards/pool",
            'BlowDryHair': "Someone is blow drying their hair",
            'BlowingCandles': "A person is blowing out candles",
            'BodyWeightSquats': "A person is doing bodyweight squats",
            'Bowling': "A person is bowling",
            'BoxingPunchingBag': "A person is boxing with a punching bag",
            'BoxingSpeedBag': "A person is training with a boxing speed bag",
            'BreastStroke': "A swimmer is doing the breast stroke",
            'BrushingTeeth': "A person is brushing their teeth",
            'CleanAndJerk': "A weightlifter is performing clean and jerk",
            'CliffDiving': "A person is cliff diving",
            'CricketBowling': "A cricket player is bowling",
            'CricketShot': "A cricket player is batting",
            'CuttingInKitchen': "Someone is cutting food in the kitchen",
            'Diving': "A person is diving into water",
            'Drumming': "A person is playing drums",
            'Fencing': "People are fencing",
            'FieldHockeyPenalty': "A field hockey penalty is being taken",
            'FloorGymnastics': "A gymnast is performing floor exercises",
            'FrisbeeCatch': "People are playing frisbee",
            'FrontCrawl': "A swimmer is doing front crawl/freestyle",
            'GolfSwing': "A person is swinging a golf club",
            'Haircut': "Someone is getting a haircut",
            'Hammering': "A person is hammering",
            'HammerThrow': "An athlete is throwing hammer",
            'HandstandPushups': "A person is doing handstand pushups",
            'HandstandWalking': "A person is walking on their hands",
            'HeadMassage': "Someone is receiving a head massage",
            'HighJump': "An athlete is doing high jump",
            'HorseRace': "Horses are racing",
            'HorseRiding': "A person is riding a horse",
            'HulaHoop': "A person is using a hula hoop",
            'IceDancing': "People are ice dancing",
            'JavelinThrow': "An athlete is throwing javelin",
            'JugglingBalls': "A person is juggling balls",
            'JumpingJack': "A person is doing jumping jacks",
            'JumpRope': "A person is jumping rope",
            'Kayaking': "A person is kayaking",
            'Knitting': "A person is knitting",
            'LongJump': "An athlete is doing long jump",
            'Lunges': "A person is doing lunges exercise",
            'MilitaryParade': "A military parade is taking place",
            'Mixing': "Someone is mixing ingredients",
            'MoppingFloor': "A person is mopping the floor",
            'Nunchucks': "A person is using nunchucks",
            'ParallelBars': "A gymnast is performing on parallel bars",
            'PizzaTossing': "A person is tossing pizza dough",
            'PlayingCello': "A person is playing the cello",
            'PlayingDaf': "A person is playing the daf (drum)",
            'PlayingDhol': "A person is playing the dhol (drum)",
            'PlayingFlute': "A person is playing the flute",
            'PlayingGuitar': "A person is playing the guitar",
            'PlayingPiano': "A person is playing the piano",
            'PlayingSitar': "A person is playing the sitar",
            'PlayingTabla': "A person is playing tabla drums",
            'PlayingViolin': "A person is playing the violin",
            'PoleVault': "An athlete is doing pole vault",
            'PommelHorse': "A gymnast is performing on pommel horse",
            'PullUps': "A person is doing pull-ups",
            'Punch': "A person is punching",
            'PushUps': "A person is doing push-ups",
            'Rafting': "People are rafting",
            'RockClimbingIndoor': "A person is rock climbing indoors",
            'RopeClimbing': "A person is climbing a rope",
            'Rowing': "A person is rowing",
            'SalsaSpin': "A person is doing salsa dance spins",
            'ShavingBeard': "A person is shaving their beard",
            'Shotput': "An athlete is throwing shot put",
            'SkateBoarding': "A person is skateboarding",
            'Skiing': "A person is skiing",
            'Skijet': "A person is riding a jet ski",
            'SkyDiving': "A person is skydiving",
            'SoccerJuggling': "A person is juggling a soccer ball",
            'SoccerPenalty': "A soccer penalty kick is being taken",
            'StillRings': "A gymnast is performing on still rings",
            'SumoWrestling': "Sumo wrestlers are wrestling",
            'Surfing': "A person is surfing",
            'Swing': "A person is swinging",
            'TableTennisShot': "People are playing table tennis",
            'TaiChi': "A person is practicing Tai Chi",
            'TennisSwing': "A person is playing tennis",
            'ThrowDiscus': "An athlete is throwing discus",
            'TrampolineJumping': "A person is jumping on a trampoline",
            'Typing': "A person is typing on a keyboard",
            'UnevenBars': "A gymnast is performing on uneven bars",
            'VolleyballSpiking': "A volleyball player is spiking the ball",
            'WalkingWithDog': "A person is walking with their dog",
            'WallPushups': "A person is doing wall push-ups",
            'WritingOnBoard': "A person is writing on a board",
            'YoYo': "A person is playing with a yo-yo"
        }
        return descriptions
    
    def load_video_frames(self, video_path, frames_per_clip=8):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frames uniformly
        if total_frames >= frames_per_clip:
            frame_indices = np.linspace(0, total_frames - 1, frames_per_clip, dtype=int)
        else:
            # Repeat frames if video is too short
            frame_indices = np.arange(total_frames)
            while len(frame_indices) < frames_per_clip:
                frame_indices = np.concatenate([
                    frame_indices, 
                    frame_indices[:frames_per_clip - len(frame_indices)]
                ])
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])  # Repeat last frame
                else:
                    frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
        
        cap.release()
        return frames[:frames_per_clip]
    
    def predict_single_video(self, video_path, frames_per_clip=8, top_k=3):
        """Predict action for a single video"""
        try:
            # Load and preprocess video
            frames = self.load_video_frames(video_path, frames_per_clip)
            video_tensor = self.transform(frames)
            video_tensor = video_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Predict
            with torch.no_grad():
                outputs = self.model(video_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                class_name = self.class_names[class_idx]
                confidence = top_probs[0][i].item()
                description = self.action_descriptions.get(class_name, f"A person is performing {class_name}")
                
                predictions.append({
                    'action': class_name,
                    'confidence': confidence,
                    'description': description
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return [{
                'action': 'Unknown',
                'confidence': 0.0,
                'description': 'Could not analyze the video'
            }]
    
    def video_to_text(self, video_path, frames_per_clip=8, detailed=False):
        """Convert video to human-readable text description"""
        predictions = self.predict_single_video(video_path, frames_per_clip, top_k=3)
        
        if not predictions:
            return "Could not analyze the video."
        
        best_prediction = predictions[0]
        confidence = best_prediction['confidence']
        
        if detailed:
            # Detailed response with confidence and alternatives
            text = f"In this video, {best_prediction['description'].lower()}"
            text += f" (Confidence: {confidence:.1%})."
            
            if len(predictions) > 1 and predictions[1]['confidence'] > 0.1:
                text += f" Alternatively, it could be {predictions[1]['description'].lower()}"
                text += f" (Confidence: {predictions[1]['confidence']:.1%})."
            
            return text
        else:
            # Simple response
            if confidence > 0.7:
                return f"In this video, {best_prediction['description'].lower()}."
            elif confidence > 0.4:
                return f"This appears to show {best_prediction['description'].lower()}."
            else:
                return f"This might show {best_prediction['description'].lower()}, but I'm not very confident."
    
    def batch_process_videos(self, video_directory, output_file=None):
        """Process multiple videos in a directory"""
        video_dir = Path(video_directory)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
            video_files.extend(video_dir.glob(f'*{ext.upper()}'))
        
        results = []
        
        print(f"Found {len(video_files)} videos to process...")
        
        for video_path in video_files:
            print(f"Processing: {video_path.name}")
            
            predictions = self.predict_single_video(video_path)
            description = self.video_to_text(video_path, detailed=True)
            
            result = {
                'video_file': video_path.name,
                'description': description,
                'top_predictions': predictions
            }
            
            results.append(result)
            print(f"Result: {description}")
            print("-" * 50)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        return results

def create_video_analyzer(model_path, model_type='efficient'):
    """Create a video analyzer instance"""
    return VideoActionRecognizer(model_path, model_type)

# Example usage
def main():
    """Example usage of the video-to-text system"""
    # Path to your trained model
    model_path = r"C:\ASH_PROJECT\outputs\checkpoints\best_model.pth"
    
    # Create analyzer
    analyzer = create_video_analyzer(model_path, model_type='efficient')
    
    # Example: Analyze a single video
    video_path = r"C:\path\to\your\video.mp4"  # Replace with actual path
    
    if Path(video_path).exists():
        # Get simple description
        description = analyzer.video_to_text(video_path)
        print(f"Simple: {description}")
        
        # Get detailed description
        detailed_description = analyzer.video_to_text(video_path, detailed=True)
        print(f"Detailed: {detailed_description}")
        
        # Get raw predictions
        predictions = analyzer.predict_single_video(video_path)
        print("Top predictions:")
        for pred in predictions:
            print(f"  {pred['action']}: {pred['confidence']:.3f} - {pred['description']}")
    else:
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")

if __name__ == "__main__":
    main()