"""
Active Learning Manager
Automatically flags uncertain predictions and manages annotation workflow
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from enum import Enum


class SamplingStrategy(Enum):
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"
    RANDOM = "random"


@dataclass
class AnnotationTask:
    """Represents a sample flagged for human annotation"""
    task_id: str
    image_path: str
    predictions: Dict
    uncertainty_score: float
    model_disagreement: float
    timestamp: datetime
    priority: int  # 1-5, higher = more urgent
    status: str = "pending"  # pending, in_progress, completed, rejected
    annotator_id: Optional[str] = None
    ground_truth: Optional[Dict] = None
    annotation_time_seconds: Optional[float] = None


class ActiveLearningManager:
    """
    Manages the active learning loop:
    1. Flag low-confidence predictions
    2. Select diverse samples for annotation
    3. Track annotation quality
    4. Trigger retraining
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        disagreement_threshold: float = 0.3,
        uncertainty_threshold: float = 0.4,
        annotation_budget: int = 1000,  # Max annotations per cycle
        retraining_threshold: int = 500  # Trigger retraining after N new samples
    ):
        self.confidence_threshold = confidence_threshold
        self.disagreement_threshold = disagreement_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.annotation_budget = annotation_budget
        self.retraining_threshold = retraining_threshold
        
        self.annotation_queue = deque()
        self.completed_annotations = []
        self.model_performance_history = []
        
    def should_flag_for_annotation(
        self,
        prediction: Dict,
        uncertainty: float,
        model_agreement: float
    ) -> bool:
        """
        Decision logic: should this prediction be sent for human review?
        """
        
        # Criteria 1: Low confidence
        low_confidence = prediction['confidence'] < self.confidence_threshold
        
        # Criteria 2: High disagreement between models
        high_disagreement = model_agreement < (1 - self.disagreement_threshold)
        
        # Criteria 3: High uncertainty (entropy)
        high_uncertainty = uncertainty > self.uncertainty_threshold
        
        # Criteria 4: Near decision boundary (confidence close to 0.5 for binary)
        if 'fruit_type_confidence' in prediction:
            near_boundary = 0.4 < prediction['fruit_type_confidence'] < 0.6
        else:
            near_boundary = False
            
        # Flag if ANY criteria is met
        should_flag = (
            low_confidence or 
            high_disagreement or 
            high_uncertainty or 
            near_boundary
        )
        
        return should_flag
        
    def add_to_annotation_queue(
        self,
        image_path: str,
        predictions: Dict,
        uncertainty: float,
        model_disagreement: float,
        metadata: Optional[Dict] = None
    ):
        """
        Add a sample to the annotation queue
        """
        
        # Calculate priority score
        priority = self._calculate_priority(
            predictions['confidence'],
            uncertainty,
            model_disagreement
        )
        
        task = AnnotationTask(
            task_id=self._generate_task_id(),
            image_path=image_path,
            predictions=predictions,
            uncertainty_score=uncertainty,
            model_disagreement=model_disagreement,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Insert based on priority (high priority first)
        self._insert_by_priority(task)
        
    def _insert_by_priority(self, task: AnnotationTask):
        """Insert task maintaining priority order"""
        
        for i, existing_task in enumerate(self.annotation_queue):
            if task.priority > existing_task.priority:
                self.annotation_queue.insert(i, task)
                return
                
        self.annotation_queue.append(task)
        
    def _calculate_priority(
        self,
        confidence: float,
        uncertainty: float,
        disagreement: float
    ) -> int:
        """
        Calculate priority (1-5) based on multiple factors
        5 = most urgent (very low confidence, high disagreement)
        1 = less urgent
        """
        
        # Inverse confidence (lower confidence = higher priority)
        confidence_score = (1 - confidence) * 2
        
        # Uncertainty score
        uncertainty_score = uncertainty * 2
        
        # Disagreement score
        disagreement_score = disagreement * 1.5
        
        # Combined score
        total_score = confidence_score + uncertainty_score + disagreement_score
        
        # Map to 1-5 scale
        if total_score > 4.5:
            return 5
        elif total_score > 3.5:
            return 4
        elif total_score > 2.5:
            return 3
        elif total_score > 1.5:
            return 2
        else:
            return 1
            
    def select_samples_for_annotation(
        self,
        batch_size: int = 100,
        strategy: SamplingStrategy = SamplingStrategy.HYBRID
    ) -> List[AnnotationTask]:
        """
        Select diverse, informative samples from the queue
        
        Strategies:
        - UNCERTAINTY: Select most uncertain samples
        - DIVERSITY: Select diverse samples (k-means clustering)
        - HYBRID: Combine uncertainty and diversity
        - RANDOM: Random sampling (baseline)
        """
        
        if len(self.annotation_queue) == 0:
            return []
            
        available_tasks = [
            task for task in self.annotation_queue 
            if task.status == "pending"
        ]
        
        if len(available_tasks) <= batch_size:
            return available_tasks
            
        if strategy == SamplingStrategy.UNCERTAINTY:
            return self._uncertainty_sampling(available_tasks, batch_size)
            
        elif strategy == SamplingStrategy.DIVERSITY:
            return self._diversity_sampling(available_tasks, batch_size)
            
        elif strategy == SamplingStrategy.HYBRID:
            return self._hybrid_sampling(available_tasks, batch_size)
            
        else:  # RANDOM
            import random
            return random.sample(available_tasks, batch_size)
            
    def _uncertainty_sampling(
        self,
        tasks: List[AnnotationTask],
        n: int
    ) -> List[AnnotationTask]:
        """Select n most uncertain samples"""
        
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.uncertainty_score,
            reverse=True
        )
        
        return sorted_tasks[:n]
        
    def _diversity_sampling(
        self,
        tasks: List[AnnotationTask],
        n: int
    ) -> List[AnnotationTask]:
        """
        Select diverse samples using k-means clustering on features
        """
        
        from sklearn.cluster import KMeans
        
        # Extract feature vectors (assuming stored in predictions)
        features = []
        for task in tasks:
            if 'features' in task.predictions:
                features.append(task.predictions['features'])
            else:
                # Fallback: use prediction probabilities as features
                features.append([
                    task.predictions.get('confidence', 0.5),
                    task.uncertainty_score,
                    task.model_disagreement
                ])
                
        features = np.array(features)
        
        # Cluster into n clusters
        kmeans = KMeans(n_clusters=min(n, len(tasks)), random_state=42)
        kmeans.fit(features)
        
        # Select one sample from each cluster (closest to centroid)
        selected_tasks = []
        
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Find sample closest to cluster center
            cluster_features = features[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]
            
            distances = np.linalg.norm(
                cluster_features - centroid,
                axis=1
            )
            
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_tasks.append(tasks[closest_idx])
            
        return selected_tasks
        
    def _hybrid_sampling(
        self,
        tasks: List[AnnotationTask],
        n: int
    ) -> List[AnnotationTask]:
        """
        Hybrid: 70% uncertainty, 30% diversity
        """
        
        n_uncertain = int(n * 0.7)
        n_diverse = n - n_uncertain
        
        # Get top uncertain samples
        uncertain_samples = self._uncertainty_sampling(tasks, n_uncertain)
        
        # Get diverse samples from remaining
        remaining_tasks = [t for t in tasks if t not in uncertain_samples]
        diverse_samples = self._diversity_sampling(remaining_tasks, n_diverse)
        
        return uncertain_samples + diverse_samples
        
    def record_annotation(
        self,
        task_id: str,
        annotator_id: str,
        ground_truth: Dict,
        annotation_time: float
    ):
        """
        Record completed annotation
        """
        
        # Find task in queue
        for task in self.annotation_queue:
            if task.task_id == task_id:
                task.status = "completed"
                task.annotator_id = annotator_id
                task.ground_truth = ground_truth
                task.annotation_time_seconds = annotation_time
                
                self.completed_annotations.append(task)
                break
                
    def calculate_annotation_quality(self) -> Dict:
        """
        Calculate inter-annotator agreement and quality metrics
        """
        
        if len(self.completed_annotations) < 10:
            return {"error": "Insufficient annotations"}
            
        # Extract ground truth labels
        model_predictions = []
        ground_truths = []
        
        for task in self.completed_annotations:
            model_predictions.append(task.predictions['fruit_type'])
            ground_truths.append(task.ground_truth['fruit_type'])
            
        # Calculate accuracy
        correct = sum(
            1 for p, g in zip(model_predictions, ground_truths) if p == g
        )
        accuracy = correct / len(ground_truths)
        
        # Calculate per-class precision/recall
        from sklearn.metrics import classification_report
        
        report = classification_report(
            ground_truths,
            model_predictions,
            output_dict=True
        )
        
        return {
            'overall_accuracy': accuracy,
            'num_annotations': len(self.completed_annotations),
            'avg_annotation_time': np.mean([
                t.annotation_time_seconds 
                for t in self.completed_annotations
            ]),
            'per_class_metrics': report
        }
        
    def should_trigger_retraining(self) -> bool:
        """
        Determine if we have enough new annotations to retrain
        """
        
        new_annotations_count = len([
            t for t in self.completed_annotations
            if t.status == "completed"
        ])
        
        return new_annotations_count >= self.retraining_threshold
        
    def export_training_data(self, output_path: str):
        """
        Export annotated samples for retraining
        """
        
        training_samples = []
        
        for task in self.completed_annotations:
            if task.ground_truth is not None:
                training_samples.append({
                    'image_path': task.image_path,
                    'fruit_type': task.ground_truth['fruit_type'],
                    'ripeness': task.ground_truth.get('ripeness'),
                    'defects': task.ground_truth.get('defects', []),
                    'freshness_score': task.ground_truth.get('freshness'),
                    'shelf_life_days': task.ground_truth.get('shelf_life'),
                    'metadata': {
                        'original_prediction': task.predictions,
                        'uncertainty': task.uncertainty_score,
                        'timestamp': task.timestamp.isoformat()
                    }
                })
                
        with open(output_path, 'w') as f:
            json.dump(training_samples, f, indent=2)
            
        return len(training_samples)
        
    def get_annotation_statistics(self) -> Dict:
        """
        Get comprehensive statistics about annotation workflow
        """
        
        pending_count = len([
            t for t in self.annotation_queue if t.status == "pending"
        ])
        
        in_progress_count = len([
            t for t in self.annotation_queue if t.status == "in_progress"
        ])
        
        completed_count = len(self.completed_annotations)
        
        if completed_count > 0:
            avg_uncertainty = np.mean([
                t.uncertainty_score for t in self.completed_annotations
            ])
            
            avg_priority = np.mean([
                t.priority for t in self.completed_annotations
            ])
        else:
            avg_uncertainty = 0
            avg_priority = 0
            
        return {
            'queue_status': {
                'pending': pending_count,
                'in_progress': in_progress_count,
                'completed': completed_count,
                'total': len(self.annotation_queue)
            },
            'average_metrics': {
                'uncertainty': float(avg_uncertainty),
                'priority': float(avg_priority)
            },
            'ready_for_retraining': self.should_trigger_retraining(),
            'budget_utilization': completed_count / self.annotation_budget * 100
        }
        
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return f"task_{uuid.uuid4().hex[:12]}"
