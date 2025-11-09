#camera.py
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Generator
import time
from collections import deque
import json

@dataclass
class PostureMetrics:
    """Data class to store posture analysis metrics"""
    shoulder_angle: float
    head_tilt: float
    forward_lean: float
    is_slouching: bool
    is_tilted: bool
    is_leaning: bool
    head_motion_score: float
    hand_motion_score: float
    eye_contact_maintained: bool
    eye_contact_duration: float
    is_head_moving: bool
    is_hand_fidgeting: bool
    left_iris_relative: Optional[float]
    right_iris_relative: Optional[float]
    timestamp: float
    issues: List[str]
    
    def to_json(self) -> str:
        """Convert metrics to JSON string"""
        data = asdict(self)
        return json.dumps(data)
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        return asdict(self)


class PostureDetector:
    """Detects and analyzes posture using MediaPipe Pose"""
    
    def __init__(self):
        # Initialize MediaPipe Pose and Hands
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Initialize face mesh for eye tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Thresholds for posture issues
        self.FORWARD_LEAN_THRESHOLD = 0.15
        self.HEAD_TILT_MIN = 165
        self.HEAD_TILT_MAX = 195
        self.SHOULDER_ANGLE_MIN = 165
        self.SHOULDER_ANGLE_MAX = 195
        
        # New thresholds
        self.HEAD_MOTION_THRESHOLD = 15.0
        self.HAND_MOTION_THRESHOLD = 20.0
        self.EYE_CONTACT_DURATION_THRESHOLD = 2.0
        self.LEFT_EYE_MIN = 2.75
        self.LEFT_EYE_MAX = 2.95
        self.RIGHT_EYE_MIN = -1.95
        self.RIGHT_EYE_MAX = -1.775
        
        # For smoothing measurements
        self.metric_history = {
            'shoulder_angle': [],
            'head_tilt': [],
            'forward_lean': []
        }
        self.history_size = 5
        
        # Head motion tracking
        self.head_position_history = deque(maxlen=10)
        self.head_motion_buffer = deque(maxlen=30)
        
        # Hand motion tracking
        self.hand_positions_history = deque(maxlen=10)
        self.hand_motion_buffer = deque(maxlen=30)
        
        # Eye contact tracking
        self.eye_center_history = deque(maxlen=10)
        self.looking_away_start_time = None
        self.total_looking_away_time = 0.0
        
        # For debugging iris positions
        self.last_left_iris = None
        self.last_right_iris = None
        
    def calculate_angle(self, point1: tuple, point2: tuple, point3: tuple) -> float:
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_slope_angle(self, point1: tuple, point2: tuple) -> float:
        """Calculate the angle of a line from horizontal"""
        x1, y1 = point1
        x2, y2 = point2
        
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return abs(angle)
    
    def smooth_metric(self, metric_name: str, value: float) -> float:
        """Apply moving average smoothing to reduce jitter"""
        self.metric_history[metric_name].append(value)
        
        if len(self.metric_history[metric_name]) > self.history_size:
            self.metric_history[metric_name].pop(0)
        
        return np.mean(self.metric_history[metric_name])
    
    def calculate_head_motion(self, nose_coords: tuple) -> float:
        """Calculate head movement across frames"""
        self.head_position_history.append(nose_coords)
        
        if len(self.head_position_history) < 2:
            return 0.0
        
        prev_pos = self.head_position_history[-2]
        curr_pos = self.head_position_history[-1]
        
        displacement = np.sqrt(
            (curr_pos[0] - prev_pos[0])**2 + 
            (curr_pos[1] - prev_pos[1])**2
        )
        
        self.head_motion_buffer.append(displacement)
        
        avg_motion = np.mean(list(self.head_motion_buffer))
        return avg_motion
    
    def calculate_hand_motion(self, hand_landmarks_list) -> float:
        """Calculate hand movement and fidgeting"""
        if not hand_landmarks_list:
            self.hand_positions_history.append(None)
            return 0.0
        
        hand_centers = []
        for hand_landmarks in hand_landmarks_list:
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            hand_centers.append((wrist.x, wrist.y))
        
        avg_hand_pos = np.mean(hand_centers, axis=0) if hand_centers else None
        
        self.hand_positions_history.append(avg_hand_pos)
        
        if len(self.hand_positions_history) < 2 or avg_hand_pos is None:
            return 0.0
        
        if self.hand_positions_history[-2] is not None:
            prev_pos = self.hand_positions_history[-2]
            curr_pos = avg_hand_pos
            
            displacement = np.sqrt(
                ((curr_pos[0] - prev_pos[0]) * 1280)**2 + 
                ((curr_pos[1] - prev_pos[1]) * 720)**2
            )
            
            self.hand_motion_buffer.append(displacement)
        
        avg_motion = np.mean(list(self.hand_motion_buffer)) if self.hand_motion_buffer else 0.0
        return avg_motion
    
    def check_eye_contact(self, face_landmarks, image_shape) -> tuple:
        """Check if eyes are looking at camera"""
        if not face_landmarks:
            return True, 0.0, None, None
        
        h, w = image_shape[:2]
        
        left_iris_indices = [474, 475, 476, 477]
        right_iris_indices = [469, 470, 471, 472]
        
        left_iris_x = np.mean([face_landmarks.landmark[i].x for i in left_iris_indices])
        right_iris_x = np.mean([face_landmarks.landmark[i].x for i in right_iris_indices])
        
        left_eye_left = face_landmarks.landmark[33].x
        left_eye_right = face_landmarks.landmark[133].x
        right_eye_left = face_landmarks.landmark[263].x
        right_eye_right = face_landmarks.landmark[362].x
        
        left_eye_width = abs(left_eye_right - left_eye_left)
        right_eye_width = abs(right_eye_right - right_eye_left)
        
        left_iris_relative = (left_iris_x - left_eye_left) / left_eye_width if left_eye_width > 0 else 0.5
        right_iris_relative = (right_iris_x - right_eye_right) / right_eye_width if right_eye_width > 0 else 0.5
        
        left_centered = self.LEFT_EYE_MIN < left_iris_relative < self.LEFT_EYE_MAX
        right_centered = self.RIGHT_EYE_MIN < right_iris_relative < self.RIGHT_EYE_MAX
        
        is_looking = left_centered and right_centered
        
        current_time = time.time()
        
        if not is_looking:
            if self.looking_away_start_time is None:
                self.looking_away_start_time = current_time
            duration_away = current_time - self.looking_away_start_time
        else:
            self.looking_away_start_time = None
            duration_away = 0.0
        
        return is_looking, duration_away, left_iris_relative, right_iris_relative
    
    def analyze_posture(self, pose_landmarks, hand_landmarks_list, face_landmarks, 
                       image_shape) -> Optional[PostureMetrics]:
        """Analyze posture from MediaPipe landmarks"""
        if not pose_landmarks:
            return None
        
        h, w = image_shape[:2]
        
        # Extract key landmarks
        nose = pose_landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        # Convert to pixel coordinates
        nose_coords = (nose.x * w, nose.y * h)
        left_shoulder_coords = (left_shoulder.x * w, left_shoulder.y * h)
        right_shoulder_coords = (right_shoulder.x * w, right_shoulder.y * h)
        left_ear_coords = (left_ear.x * w, left_ear.y * h)
        right_ear_coords = (right_ear.x * w, right_ear.y * h)
        
        # Calculate midpoints
        shoulder_midpoint = (
            (left_shoulder_coords[0] + right_shoulder_coords[0]) / 2,
            (left_shoulder_coords[1] + right_shoulder_coords[1]) / 2
        )
        
        # Original posture metrics
        shoulder_angle_raw = self.calculate_slope_angle(left_shoulder_coords, right_shoulder_coords)
        shoulder_angle = self.smooth_metric('shoulder_angle', shoulder_angle_raw)
        
        head_tilt_raw = self.calculate_slope_angle(left_ear_coords, right_ear_coords)
        head_tilt = self.smooth_metric('head_tilt', head_tilt_raw)
        
        vertical_distance = abs(nose_coords[1] - shoulder_midpoint[1])
        horizontal_offset = abs(nose_coords[0] - shoulder_midpoint[0])
        forward_lean_raw = horizontal_offset / max(vertical_distance, 1)
        forward_lean = self.smooth_metric('forward_lean', forward_lean_raw)
        
        # New metrics
        head_motion = self.calculate_head_motion(nose_coords)
        hand_motion = self.calculate_hand_motion(hand_landmarks_list)
        is_looking, time_looking_away, left_iris_rel, right_iris_rel = self.check_eye_contact(face_landmarks, image_shape)
        
        # Store iris values
        self.last_left_iris = left_iris_rel
        self.last_right_iris = right_iris_rel
        
        # Detect issues
        issues = []
        is_slouching = False
        is_tilted = False
        is_leaning = False
        is_head_moving = False
        is_hand_fidgeting = False
        eye_contact_maintained = True
        
        if shoulder_angle > self.SHOULDER_ANGLE_MAX or shoulder_angle < self.SHOULDER_ANGLE_MIN:
            is_slouching = True
            issues.append("Shoulders Not Level")
        
        if head_tilt > self.HEAD_TILT_MAX or head_tilt < self.HEAD_TILT_MIN:
            is_tilted = True
            issues.append("Head Tilted")
        
        if forward_lean > self.FORWARD_LEAN_THRESHOLD:
            is_leaning = True
            issues.append("Leaning Forward")
        
        if head_motion > self.HEAD_MOTION_THRESHOLD:
            is_head_moving = True
            issues.append("Excessive Head Movement")
        
        if hand_motion > self.HAND_MOTION_THRESHOLD:
            is_hand_fidgeting = True
            issues.append("Hand Fidgeting")
        
        if time_looking_away > self.EYE_CONTACT_DURATION_THRESHOLD:
            eye_contact_maintained = False
            issues.append("Missing Eye Contact")
        
        return PostureMetrics(
            shoulder_angle=shoulder_angle,
            head_tilt=head_tilt,
            forward_lean=forward_lean,
            is_slouching=is_slouching,
            is_tilted=is_tilted,
            is_leaning=is_leaning,
            head_motion_score=head_motion,
            hand_motion_score=hand_motion,
            eye_contact_maintained=eye_contact_maintained,
            eye_contact_duration=time_looking_away,
            is_head_moving=is_head_moving,
            is_hand_fidgeting=is_hand_fidgeting,
            left_iris_relative=left_iris_rel,
            right_iris_relative=right_iris_rel,
            timestamp=time.time(),
            issues=issues
        )
    
    def draw_posture_info(self, image, metrics: PostureMetrics):
        """Draw posture information on the image"""
        if not metrics:
            return image
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 30
        
        # Determine primary issue for status message
        if not metrics.issues:
            status = "Good Posture"
            status_color = (0, 255, 0)
        else:
            if "Missing Eye Contact" in metrics.issues:
                status = "Missing Eye Contact"
            elif "Excessive Head Movement" in metrics.issues:
                status = "Head Movement"
            elif "Hand Fidgeting" in metrics.issues:
                status = "Hand Fidgeting"
            elif "Leaning Forward" in metrics.issues:
                status = "Leaning"
            elif "Shoulders Not Level" in metrics.issues:
                status = "Slouching"
            elif "Head Tilted" in metrics.issues:
                status = "Head Tilted"
            else:
                status = metrics.issues[0]
            status_color = (0, 0, 255)
        
        # Draw background rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (450, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Draw status
        cv2.putText(image, status, (20, y_offset), font, font_scale, status_color, thickness)
        
        # Draw metrics
        y_offset += 30
        cv2.putText(image, f"Shoulder: {metrics.shoulder_angle:.1f}° | Head Tilt: {metrics.head_tilt:.1f}°", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(image, f"Forward Lean: {metrics.forward_lean:.2f}", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        head_motion_color = (0, 255, 0) if not metrics.is_head_moving else (0, 165, 255)
        cv2.putText(image, f"Head Motion: {metrics.head_motion_score:.1f} px/frame", 
                   (20, y_offset), font, 0.5, head_motion_color, 1)
        
        y_offset += 25
        hand_motion_color = (0, 255, 0) if not metrics.is_hand_fidgeting else (0, 165, 255)
        cv2.putText(image, f"Hand Motion: {metrics.hand_motion_score:.1f} px/frame", 
                   (20, y_offset), font, 0.5, hand_motion_color, 1)
        
        y_offset += 25
        eye_color = (0, 255, 0) if metrics.eye_contact_maintained else (0, 0, 255)
        eye_status = "Maintained" if metrics.eye_contact_maintained else f"Away {metrics.eye_contact_duration:.1f}s"
        cv2.putText(image, f"Eye Contact: {eye_status}", 
                   (20, y_offset), font, 0.5, eye_color, 1)
        
        # Draw iris position debug info
        y_offset += 25
        if self.last_left_iris is not None and self.last_right_iris is not None:
            cv2.putText(image, f"L Iris: {self.last_left_iris:.3f} | R Iris: {self.last_right_iris:.3f}", 
                       (20, y_offset), font, 0.45, (255, 200, 0), 1)
        
        return image
    
    def process_frame(self, frame) -> tuple:
        """Process a single frame and return annotated frame with metrics"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        
        # Process hands
        hands_results = self.hands.process(rgb_frame)
        
        # Process face mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Analyze all metrics
        metrics = None
        if pose_results.pose_landmarks:
            metrics = self.analyze_posture(
                pose_results.pose_landmarks.landmark,
                hands_results.multi_hand_landmarks if hands_results.multi_hand_landmarks else [],
                face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None,
                frame.shape
            )
            
            frame = self.draw_posture_info(frame, metrics)
        
        return frame, metrics
    
    def release(self):
        """Clean up resources"""
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()


# ============================================================================
# STREAMING INTERFACE FOR APP.PY
# ============================================================================

def stream_camera_metrics(show_video: bool = True) -> Generator[Dict, None, None]:
    """
    Generator function that continuously yields camera metrics
    
    Args:
        show_video: Whether to display the video feed (default: True)
        
    Yields:
        Dictionary containing camera metrics
        
    Usage in app.py:
        for metrics in stream_camera_metrics():
            # metrics is a dict with all camera data
            print(metrics)
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = PostureDetector()
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            annotated_frame, metrics = detector.process_frame(frame)
            
            if show_video:
                cv2.imshow('Interview Posture Detector', annotated_frame)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Yield metrics if available
            if metrics:
                yield metrics.to_dict()
            else:
                # Yield empty metrics if no detection
                yield {
                    "shoulder_angle": 180,
                    "head_tilt": 180,
                    "forward_lean": 0.0,
                    "head_motion_score": 0.0,
                    "hand_motion_score": 0.0,
                    "eye_contact_maintained": True,
                    "left_iris_relative": None,
                    "right_iris_relative": None,
                    "timestamp": time.time()
                }
    
    finally:
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        detector.release()


def get_camera_metrics_once() -> Optional[Dict]:
    """
    Capture a single frame and return metrics
    Useful for testing or single-shot analysis
    
    Returns:
        Dictionary containing camera metrics or None if no detection
    """
    cap = cv2.VideoCapture(0,1200)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = PostureDetector()
    
    try:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            _, metrics = detector.process_frame(frame)
            return metrics.to_dict() if metrics else None
    finally:
        cap.release()
        detector.release()
    
    return None


# ============================================================================
# STANDALONE TEST
# ============================================================================

def main():
    """Test function - streams metrics and prints them"""
    print("Starting camera metrics streaming...")
    print("Press 'q' to quit")
    print("-" * 70)
    
    frame_count = 0
    for metrics in stream_camera_metrics(show_video=True):
        frame_count += 1
        
        # Print every 30 frames (~1 second at 30fps)
        if frame_count % 30 == 0:
            print(f"\nFrame {frame_count}:")
            print(f"  Eye Contact: {metrics.get('eye_contact_maintained')}")
            print(f"  Head Tilt: {metrics.get('head_tilt', 0):.1f}°")
            print(f"  Shoulder Angle: {metrics.get('shoulder_angle', 0):.1f}°")
            print(f"  Hand Fidgeting: {metrics.get('hand_motion_score', 0):.1f}")


if __name__ == "__main__":
    main()