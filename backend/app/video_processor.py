"""
Video Stream Processor with Object Tracking
Real-time fruit detection and tracking for conveyor belt systems
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import time


@dataclass
class TrackedFruit:
    """Represents a single tracked fruit"""
    track_id: int
    fruit_type: str
    ripeness: str
    quality_grade: str
    first_seen: datetime
    last_seen: datetime
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidence: float = 0.0
    frames_tracked: int = 0
    verified: bool = False  # Classified with high confidence


class SimpleTracker:
    """
    Simplified object tracker using IOU matching
    (Production would use SORT/DeepSORT)
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence)
            
        Returns:
            List of (x1, y1, x2, y2, track_id)
        """
        
        self.frame_count += 1
        
        # Match detections to existing tracks
        if len(self.tracks) == 0:
            # Initialize tracks
            tracks_out = []
            for det in detections:
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': det[:4],
                    'last_seen': self.frame_count,
                    'age': 0
                }
                
                tracks_out.append((*det[:4], track_id))
                
            return tracks_out
            
        # Compute IOU between detections and tracks
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, det in enumerate(detections):
            for j, track_bbox in enumerate(track_bboxes):
                iou_matrix[i, j] = self._compute_iou(det[:4], track_bbox)
                
        # Hungarian matching (simplified - use linear_sum_assignment in production)
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        # Greedy matching
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
                
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            matched_indices.append((i, j))
            unmatched_detections.remove(i)
            unmatched_tracks.remove(j)
            
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
            
        # Update matched tracks
        tracks_out = []
        
        for det_idx, track_idx in matched_indices:
            track_id = track_ids[track_idx]
            self.tracks[track_id]['bbox'] = detections[det_idx][:4]
            self.tracks[track_id]['last_seen'] = self.frame_count
            self.tracks[track_id]['age'] = 0
            
            tracks_out.append((*detections[det_idx][:4], track_id))
            
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'bbox': detections[det_idx][:4],
                'last_seen': self.frame_count,
                'age': 0
            }
            
            tracks_out.append((*detections[det_idx][:4], track_id))
            
        # Remove old tracks
        tracks_to_remove = []
        for track_id in self.tracks:
            age = self.frame_count - self.tracks[track_id]['last_seen']
            if age > self.max_age:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            
        return tracks_out
        
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union"""
        
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class VideoStreamProcessor:
    """
    Process video streams for real-time fruit detection and tracking
    """
    
    def __init__(
        self,
        detector_model,
        classifier_model,
        tracker: Optional[SimpleTracker] = None,
        min_confidence: float = 0.7,
        classify_every_n_frames: int = 5
    ):
        self.detector = detector_model
        self.classifier = classifier_model
        self.tracker = tracker or SimpleTracker()
        self.min_confidence = min_confidence
        self.classify_every_n_frames = classify_every_n_frames
        
        # Tracked fruits database
        self.fruits_database: Dict[int, TrackedFruit] = {}
        
        # Analytics
        self.frame_count = 0
        self.total_fruits_counted = 0
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()
        
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[TrackedFruit], Dict]:
        """
        Process single frame
        
        Returns:
            (annotated_frame, active_tracks, analytics)
        """
        
        frame_start = time.time()
        
        # 1. Detect fruits
        detections = self._detect_fruits(frame)
        
        # 2. Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # 3. Classify tracked fruits (not every frame)
        if self.frame_count % self.classify_every_n_frames == 0:
            self._classify_tracked_fruits(frame, tracked_objects)
            
        # 4. Update fruit database
        active_fruits = self._update_fruit_database(tracked_objects)
        
        # 5. Annotate frame
        annotated_frame = self._annotate_frame(frame, active_fruits)
        
        # 6. Calculate analytics
        analytics = self._calculate_analytics(time.time() - frame_start)
        
        self.frame_count += 1
        
        return annotated_frame, active_fruits, analytics
        
    def process_video_stream(
        self,
        video_source: str,
        output_path: Optional[str] = None
    ) -> Generator[Tuple[np.ndarray, List[TrackedFruit], Dict], None, None]:
        """
        Process video stream (file, RTSP, webcam)
        
        Args:
            video_source: Path to video file or camera index (0, 1, etc.) or RTSP URL
            output_path: Optional path to save annotated video
            
        Yields:
            (annotated_frame, active_tracks, analytics)
        """
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Process frame
                annotated, fruits, analytics = self.process_frame(frame)
                
                # Write to output if specified
                if writer:
                    writer.write(annotated)
                    
                yield annotated, fruits, analytics
                
        finally:
            cap.release()
            if writer:
                writer.release()
                
    def _detect_fruits(
        self,
        frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Run object detection on frame
        
        Returns:
            List of (x1, y1, x2, y2, confidence)
        """
        
        # Placeholder - actual implementation depends on detector
        # This would call YOLO, etc.
        
        results = self.detector.detect(frame)
        
        detections = []
        for result in results:
            if result['confidence'] >= self.min_confidence:
                bbox = result['bbox']  # x1, y1, x2, y2
                detections.append((*bbox, result['confidence']))
                
        return detections
        
    def _classify_tracked_fruits(
        self,
        frame: np.ndarray,
        tracked_objects: List[Tuple]
    ):
        """
        Classify fruits that are being tracked
        """
        
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            
            # Crop fruit region
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if crop.size == 0:
                continue
                
            # Classify
            classification = self.classifier.predict(crop)
            
            # Update or create fruit entry
            if track_id not in self.fruits_database:
                self.fruits_database[track_id] = TrackedFruit(
                    track_id=track_id,
                    fruit_type=classification['fruit_type'],
                    ripeness=classification['ripeness'],
                    quality_grade=classification['quality_grade'],
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    confidence=classification['confidence'],
                    frames_tracked=1,
                    verified=classification['confidence'] > 0.9
                )
                
                if self.fruits_database[track_id].verified:
                    self.total_fruits_counted += 1
            else:
                fruit = self.fruits_database[track_id]
                fruit.last_seen = datetime.now()
                fruit.frames_tracked += 1
                
                # Update classification if more confident
                if classification['confidence'] > fruit.confidence:
                    fruit.fruit_type = classification['fruit_type']
                    fruit.ripeness = classification['ripeness']
                    fruit.quality_grade = classification['quality_grade']
                    fruit.confidence = classification['confidence']
                    fruit.verified = classification['confidence'] > 0.9
                    
    def _update_fruit_database(
        self,
        tracked_objects: List[Tuple]
    ) -> List[TrackedFruit]:
        """
        Update fruit database with current tracks
        """
        
        active_track_ids = {track[4] for track in tracked_objects}
        
        # Update bbox history
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            
            if track_id in self.fruits_database:
                fruit = self.fruits_database[track_id]
                fruit.bbox_history.append((x1, y1, x2, y2))
                
                # Keep only recent history
                if len(fruit.bbox_history) > 100:
                    fruit.bbox_history.pop(0)
                    
        # Get active fruits
        active_fruits = [
            self.fruits_database[tid]
            for tid in active_track_ids
            if tid in self.fruits_database
        ]
        
        return active_fruits
        
    def _annotate_frame(
        self,
        frame: np.ndarray,
        fruits: List[TrackedFruit]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        """
        
        annotated = frame.copy()
        
        for fruit in fruits:
            if len(fruit.bbox_history) == 0:
                continue
                
            # Get latest bbox
            x1, y1, x2, y2 = fruit.bbox_history[-1]
            
            # Choose color based on quality grade
            color_map = {
                'A': (0, 255, 0),    # Green
                'B': (255, 165, 0),  # Orange
                'C': (255, 0, 0)     # Red
            }
            color = color_map.get(fruit.quality_grade, (128, 128, 128))
            
            # Draw bbox
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )
            
            # Draw label
            label = f"{fruit.fruit_type} ({fruit.ripeness}) - {fruit.quality_grade}"
            label += f" [{fruit.confidence:.2f}]"
            
            cv2.putText(
                annotated,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw track ID
            cv2.putText(
                annotated,
                f"ID:{fruit.track_id}",
                (int(x1), int(y2) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            
        return annotated
        
    def _calculate_analytics(self, frame_time: float) -> Dict:
        """
        Calculate real-time analytics
        """
        
        # FPS calculation
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        # Throughput (fruits per minute)
        elapsed_time = time.time() - self.start_time
        throughput = (self.total_fruits_counted / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Grade distribution
        grade_distribution = defaultdict(int)
        for fruit in self.fruits_database.values():
            if fruit.verified:
                grade_distribution[fruit.quality_grade] += 1
                
        return {
            'fps': float(avg_fps),
            'frame_count': self.frame_count,
            'total_fruits_counted': self.total_fruits_counted,
            'active_tracks': len(self.tracker.tracks),
            'throughput_per_minute': float(throughput),
            'grade_distribution': dict(grade_distribution),
            'processing_time_ms': float(frame_time * 1000)
        }
        
    def get_summary_report(self) -> Dict:
        """
        Get comprehensive processing summary
        """
        
        # Type distribution
        type_distribution = defaultdict(int)
        ripeness_distribution = defaultdict(int)
        grade_distribution = defaultdict(int)
        
        for fruit in self.fruits_database.values():
            if fruit.verified:
                type_distribution[fruit.fruit_type] += 1
                ripeness_distribution[fruit.ripeness] += 1
                grade_distribution[fruit.quality_grade] += 1
                
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_processing_time_seconds': float(elapsed_time),
            'total_frames_processed': self.frame_count,
            'average_fps': float(np.mean(self.fps_history)),
            'total_fruits_detected': len(self.fruits_database),
            'total_fruits_verified': self.total_fruits_counted,
            'type_distribution': dict(type_distribution),
            'ripeness_distribution': dict(ripeness_distribution),
            'grade_distribution': dict(grade_distribution),
            'quality_metrics': {
                'grade_A_percentage': grade_distribution['A'] / max(self.total_fruits_counted, 1) * 100,
                'grade_B_percentage': grade_distribution['B'] / max(self.total_fruits_counted, 1) * 100,
                'grade_C_percentage': grade_distribution['C'] / max(self.total_fruits_counted, 1) * 100
            }
        }
