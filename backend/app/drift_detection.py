"""
Drift Detection & Continual Learning System
Monitors model performance and data distribution for degradation
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import json


@dataclass
class PerformanceMetrics:
    """Performance metrics at a point in time"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    throughput: float  # predictions per second
    latency_p95: float  # 95th percentile latency


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    alert_type: str  # PERFORMANCE_DRIFT, DATA_DRIFT, CONCEPT_DRIFT
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    metrics: Dict
    timestamp: datetime
    recommended_action: str


class DriftDetector:
    """
    Monitor model degradation and trigger retraining
    """
    
    def __init__(
        self,
        performance_alert_threshold: float = 0.03,  # 3% accuracy drop
        data_drift_p_value: float = 0.05,
        monitoring_window_days: int = 7,
        baseline_metrics_path: str = "models/baseline_metrics.json"
    ):
        self.performance_alert_threshold = performance_alert_threshold
        self.data_drift_p_value = data_drift_p_value
        self.monitoring_window_days = monitoring_window_days
        
        # Load baseline metrics
        self.baseline_metrics = self._load_baseline(baseline_metrics_path)
        
        # Recent performance history
        self.performance_history = deque(maxlen=10000)
        
        # Recent predictions  (for distribution analysis)
        self.recent_predictions = deque(maxlen=50000)
        
        # Feature distributions (baseline vs current)
        self.baseline_feature_distribution = None
        self.current_feature_distribution = deque(maxlen=10000)
        
        # Alert history
        self.alert_history = []
        
    def _load_baseline(self, path: str) -> Dict:
        """Load baseline performance metrics"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Initialize with default baseline
            return {
                'accuracy': 0.95,
                'precision': 0.94,
                'recall': 0.93,
                'f1_score': 0.935,
                'avg_confidence': 0.88,
                'timestamp': datetime.now().isoformat()
            }
            
    def record_prediction(
        self,
        prediction: Dict,
        ground_truth: Optional[Dict] = None,
        latency_ms: float = 0.0,
        features: Optional[np.ndarray] = None
    ):
        """
        Record a prediction for monitoring
        """
        
        record = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'ground_truth': ground_truth,
            'latency_ms': latency_ms,
            'confidence': prediction.get('confidence', 0.0),
            'features': features
        }
        
        self.recent_predictions.append(record)
        
        if features is not None:
            self.current_feature_distribution.append(features)
            
    def detect_performance_drift(self) -> Optional[DriftAlert]:
        """
        Detect if model performance has degraded
        """
        
        # Get recent predictions with ground truth
        recent_with_truth = [
            p for p in self.recent_predictions
            if p['ground_truth'] is not None
        ]
        
        if len(recent_with_truth) < 100:
            return None  # Not enough data
            
        # Calculate current accuracy
        correct = sum(
            1 for p in recent_with_truth
            if p['prediction']['fruit_type'] == p['ground_truth']['fruit_type']
        )
        
        current_accuracy = correct / len(recent_with_truth)
        baseline_accuracy = self.baseline_metrics['accuracy']
        
        # Check for drift
        drift_magnitude = baseline_accuracy - current_accuracy
        
        if drift_magnitude > self.performance_alert_threshold:
            # Calculate additional metrics
            current_metrics = self._calculate_detailed_metrics(recent_with_truth)
            
            severity = self._determine_severity(drift_magnitude)
            
            alert = DriftAlert(
                alert_type="PERFORMANCE_DRIFT",
                severity=severity,
                description=f"Model accuracy dropped by {drift_magnitude*100:.2f}% "
                           f"(from {baseline_accuracy:.3f} to {current_accuracy:.3f})",
                metrics={
                    'baseline_accuracy': baseline_accuracy,
                    'current_accuracy': current_accuracy,
                    'drift_magnitude': drift_magnitude,
                    'samples_evaluated': len(recent_with_truth),
                    **current_metrics
                },
                timestamp=datetime.now(),
                recommended_action="Trigger model retraining with recent data"
            )
            
            self.alert_history.append(alert)
            return alert
            
        return None
        
    def detect_data_drift(self) -> Optional[DriftAlert]:
        """
        Detect distribution shift using statistical tests
        """
        
        if self.baseline_feature_distribution is None:
            # Initialize baseline
            if len(self.current_feature_distribution) >= 1000:
                self.baseline_feature_distribution = np.array(
                    list(self.current_feature_distribution)
                )
            return None
            
        if len(self.current_feature_distribution) < 1000:
            return None  # Not enough recent data
            
        current_features = np.array(list(self.current_feature_distribution))
        
        # Kolmogorov-Smirnov test for distribution shift
        from scipy.stats import ks_2samp
        
        # Test each feature dimension
        drift_detected = False
        drift_dimensions = []
        
        for dim in range(min(
            self.baseline_feature_distribution.shape[1],
            current_features.shape[1]
        )):
            baseline_dim = self.baseline_feature_distribution[:, dim]
            current_dim = current_features[:, dim]
            
            statistic, p_value = ks_2samp(baseline_dim, current_dim)
            
            if p_value < self.data_drift_p_value:
                drift_detected = True
                drift_dimensions.append({
                    'dimension': dim,
                    'ks_statistic': float(statistic),
                    'p_value': float(p_value)
                })
                
        if drift_detected:
            alert = DriftAlert(
                alert_type="DATA_DRIFT",
                severity="MEDIUM" if len(drift_dimensions) < 5 else "HIGH",
                description=f"Significant data distribution shift detected in "
                           f"{len(drift_dimensions)} feature dimensions",
                metrics={
                    'drift_dimensions': drift_dimensions,
                    'total_dimensions': current_features.shape[1],
                    'sample_size': len(current_features)
                },
                timestamp=datetime.now(),
                recommended_action="Review data pipeline and consider model adaptation"
            )
            
            self.alert_history.append(alert)
            return alert
            
        return None
        
    def detect_concept_drift(self) -> Optional[DriftAlert]:
        """
        Detect emergence of new patterns or unseen classes
        """
        
        # Get recent low-confidence predictions
        recent_low_conf = [
            p for p in self.recent_predictions
            if p['confidence'] < 0.7
        ]
        
        if len(recent_low_conf) < 50:
            return None
            
        # Analyze patterns in low-confidence predictions
        low_conf_rate = len(recent_low_conf) / len(self.recent_predictions)
        expected_low_conf_rate = 0.1  # 10% baseline
        
        # Check for clusters of consistently low confidence
        from collections import Counter
        
        predicted_classes = [
            p['prediction']['fruit_type'] 
            for p in recent_low_conf
        ]
        
        class_counts = Counter(predicted_classes)
        
        # Find classes with consistently low confidence
        problematic_classes = [
            cls for cls, count in class_counts.items()
            if count > len(recent_low_conf) * 0.2  # >20% of low-conf predictions
        ]
        
        if low_conf_rate > expected_low_conf_rate * 2:  # 2x baseline
            alert = DriftAlert(
                alert_type="CONCEPT_DRIFT",
                severity="HIGH",
                description=f"Unusually high rate of low-confidence predictions "
                           f"({low_conf_rate*100:.1f}% vs expected {expected_low_conf_rate*100:.1f}%)",
                metrics={
                    'low_confidence_rate': low_conf_rate,
                    'expected_rate': expected_low_conf_rate,
                    'problematic_classes': problematic_classes,
                    'sample_size': len(self.recent_predictions)
                },
                timestamp=datetime.now(),
                recommended_action="Investigate for new fruit varieties or quality issues. "
                                  "Consider expanding training data."
            )
            
            self.alert_history.append(alert)
            return alert
            
        return None
        
    def run_monitoring_cycle(self) -> List[DriftAlert]:
        """
        Run complete monitoring cycle
        Returns list of all detected drifts
        """
        
        alerts = []
        
        # Check performance drift
        performance_alert = self.detect_performance_drift()
        if performance_alert:
            alerts.append(performance_alert)
            
        # Check data drift
        data_alert = self.detect_data_drift()
        if data_alert:
            alerts.append(data_alert)
            
        # Check concept drift
        concept_alert = self.detect_concept_drift()
        if concept_alert:
            alerts.append(concept_alert)
            
        return alerts
        
    def should_trigger_retraining(self) -> bool:
        """
        Determine if retraining should be triggered based on drift alerts
        """
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > datetime.now() - timedelta(days=1)
        ]
        
        # Trigger if any HIGH or CRITICAL severity alert in last 24h
        critical_alerts = [
            a for a in recent_alerts 
            if a.severity in ['HIGH', 'CRITICAL']
        ]
        
        return len(critical_alerts) > 0
        
    def _calculate_detailed_metrics(
        self,
        predictions_with_truth: List[Dict]
    ) -> Dict:
        """
        Calculate precision, recall, F1 per class
        """
        
        from sklearn.metrics import precision_recall_fscore_support
        
        y_true = [p['ground_truth']['fruit_type'] for p in predictions_with_truth]
        y_pred = [p['prediction']['fruit_type'] for p in predictions_with_truth]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
    def _determine_severity(self, drift_magnitude: float) -> str:
        """
        Determine alert severity based on drift magnitude
        """
        
        if drift_magnitude > 0.10:  # >10% drop
            return "CRITICAL"
        elif drift_magnitude > 0.05:  # 5-10% drop
            return "HIGH"
        elif drift_magnitude > 0.03:  # 3-5% drop
            return "MEDIUM"
        else:
            return "LOW"
            
    def get_monitoring_dashboard_data(self) -> Dict:
        """
        Get data for monitoring dashboard
        """
        
        # Recent performance trends
        if len(self.performance_history) > 0:
            recent_perf = list(self.performance_history)[-100:]
            avg_accuracy = np.mean([p.accuracy for p in recent_perf])
            avg_latency = np.mean([p.latency_p95 for p in recent_perf])
        else:
            avg_accuracy = 0
            avg_latency = 0
            
        # Recent alerts
        recent_alerts = [
            {
                'type': a.alert_type,
                'severity': a.severity,
                'description': a.description,
                'timestamp': a.timestamp.isoformat()
            }
            for a in self.alert_history[-10:]
        ]
        
        return {
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': {
                'accuracy': float(avg_accuracy),
                'latency_p95_ms': float(avg_latency)
            },
            'recent_alerts': recent_alerts,
            'monitoring_status': 'healthy' if len(recent_alerts) == 0 else 'degraded',
            'samples_monitored': len(self.recent_predictions),
            'retraining_recommended': self.should_trigger_retraining()
        }
