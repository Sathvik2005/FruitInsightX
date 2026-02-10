"""
Multi-Model Ensemble Orchestrator
Manages multiple detection and classification models with confidence-weighted voting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from enum import Enum


class ModelType(Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    MULTITASK = "multitask"


@dataclass
class ModelConfig:
    """Configuration for individual model"""
    name: str
    type: ModelType
    weight: float
    min_confidence: float
    latency_target_ms: float
    enabled: bool = True


@dataclass
class PredictionResult:
    """Standard prediction result format"""
    model_name: str
    fruit_type: str
    ripeness: str
    confidence: float
    defects: List[str]
    freshness_score: float
    shelf_life_days: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    features: Optional[np.ndarray] = None


class EnsembleOrchestrator:
    """
    Orchestrates multiple models with intelligent routing and aggregation
    """
    
    def __init__(self, config_path: str = "models/ensemble_config.yaml"):
        self.models = {}
        self.model_configs = {}
        self.performance_history = {}
        self.load_config(config_path)
        
    def load_config(self, config_path: str):
        """Load ensemble configuration"""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for model_cfg in config['models']:
            self.model_configs[model_cfg['name']] = ModelConfig(**model_cfg)
            
    async def predict_ensemble(
        self, 
        image: np.ndarray,
        nir_image: Optional[np.ndarray] = None,
        mode: str = 'fast'
    ) -> Dict:
        """
        Run ensemble prediction with multiple strategies
        
        Args:
            image: RGB image array (H, W, 3)
            nir_image: Optional NIR image array (H, W, 1)
            mode: 'fast' (parallel), 'accurate' (sequential), 'adaptive'
            
        Returns:
            Aggregated prediction with confidence and uncertainty
        """
        
        if mode == 'fast':
            # Run all models in parallel
            predictions = await self._predict_parallel(image, nir_image)
        elif mode == 'accurate':
            # Run high-accuracy models first, skip if confident
            predictions = await self._predict_cascade(image, nir_image)
        else:  # adaptive
            # Dynamically choose based on complexity
            predictions = await self._predict_adaptive(image, nir_image)
            
        # Aggregate results
        final = self._aggregate_predictions(predictions)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(predictions)
        
        # Check for active learning flag
        should_flag = self._should_flag_for_annotation(final, uncertainty)
        
        return {
            'prediction': final,
            'uncertainty': uncertainty,
            'model_agreement': self._calculate_agreement(predictions),
            'individual_predictions': predictions,
            'flag_for_annotation': should_flag,
            'ensemble_confidence': final['confidence']
        }
        
    async def _predict_parallel(
        self, 
        image: np.ndarray, 
        nir_image: Optional[np.ndarray]
    ) -> List[PredictionResult]:
        """Run all enabled models in parallel"""
        
        tasks = []
        
        for name, config in self.model_configs.items():
            if config.enabled:
                model = self.models[name]
                task = self._run_single_model(model, image, nir_image)
                tasks.append(task)
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed predictions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return valid_results
        
    async def _predict_cascade(
        self, 
        image: np.ndarray, 
        nir_image: Optional[np.ndarray]
    ) -> List[PredictionResult]:
        """
        Cascade prediction: run fast model first,
        only run slower models if confidence is low
        """
        
        results = []
        
        # Sort models by latency (fast first)
        sorted_models = sorted(
            self.model_configs.items(),
            key=lambda x: x[1].latency_target_ms
        )
        
        for name, config in sorted_models:
            if not config.enabled:
                continue
                
            model = self.models[name]
            result = await self._run_single_model(model, image, nir_image)
            results.append(result)
            
            # Early stopping if confident enough
            if result.confidence >= 0.95:
                break
                
        return results
        
    async def _predict_adaptive(
        self, 
        image: np.ndarray, 
        nir_image: Optional[np.ndarray]
    ) -> List[PredictionResult]:
        """
        Adaptive prediction based on image complexity
        """
        
        # Estimate image complexity
        complexity = self._estimate_complexity(image)
        
        if complexity < 0.3:  # Simple image
            # Use lightweight model only
            return await self._predict_lightweight(image, nir_image)
        elif complexity > 0.7:  # Complex image
            # Use full ensemble
            return await self._predict_parallel(image, nir_image)
        else:  # Medium complexity
            # Use cascade
            return await self._predict_cascade(image, nir_image)
            
    def _aggregate_predictions(
        self, 
        predictions: List[PredictionResult]
    ) -> Dict:
        """
        Aggregate predictions using confidence-weighted voting
        """
        
        if not predictions:
            raise ValueError("No valid predictions to aggregate")
            
        # Weighted voting for discrete outputs
        fruit_type_votes = {}
        ripeness_votes = {}
        
        total_weight = 0
        
        for pred in predictions:
            config = self.model_configs[pred.model_name]
            weight = config.weight * pred.confidence
            
            # Accumulate votes
            fruit_type_votes[pred.fruit_type] = \
                fruit_type_votes.get(pred.fruit_type, 0) + weight
            ripeness_votes[pred.ripeness] = \
                ripeness_votes.get(pred.ripeness, 0) + weight
                
            total_weight += weight
            
        # Normalize
        fruit_type_votes = {k: v/total_weight for k, v in fruit_type_votes.items()}
        ripeness_votes = {k: v/total_weight for k, v in ripeness_votes.items()}
        
        # Get final predictions
        final_fruit_type = max(fruit_type_votes, key=fruit_type_votes.get)
        final_ripeness = max(ripeness_votes, key=ripeness_votes.get)
        
        # Weighted average for continuous outputs
        avg_freshness = np.average(
            [p.freshness_score for p in predictions],
            weights=[self.model_configs[p.model_name].weight * p.confidence 
                     for p in predictions]
        )
        
        avg_shelf_life = int(np.average(
            [p.shelf_life_days for p in predictions],
            weights=[self.model_configs[p.model_name].weight * p.confidence 
                     for p in predictions]
        ))
        
        # Aggregate defects (union of all detected defects with conf > threshold)
        all_defects = set()
        for pred in predictions:
            if pred.confidence > 0.7:  # Only trust high-confidence defects
                all_defects.update(pred.defects)
                
        return {
            'fruit_type': final_fruit_type,
            'fruit_type_confidence': fruit_type_votes[final_fruit_type],
            'ripeness': final_ripeness,
            'ripeness_confidence': ripeness_votes[final_ripeness],
            'defects': list(all_defects),
            'freshness_score': float(avg_freshness),
            'shelf_life_days': avg_shelf_life,
            'confidence': np.mean([p.confidence for p in predictions]),
            'quality_grade': self._calculate_grade(
                final_ripeness, all_defects, avg_freshness
            )
        }
        
    def _calculate_uncertainty(
        self, 
        predictions: List[PredictionResult]
    ) -> float:
        """
        Calculate prediction uncertainty using entropy
        """
        
        # Collect all predictions for each output
        fruit_types = [p.fruit_type for p in predictions]
        ripeness_stages = [p.ripeness for p in predictions]
        
        # Calculate entropy
        def entropy(labels):
            from collections import Counter
            counts = Counter(labels)
            probs = [c / len(labels) for c in counts.values()]
            return -sum(p * np.log(p + 1e-10) for p in probs)
            
        fruit_entropy = entropy(fruit_types)
        ripeness_entropy = entropy(ripeness_stages)
        
        # Normalize to [0, 1]
        max_entropy = np.log(len(set(fruit_types)))
        normalized_entropy = (fruit_entropy + ripeness_entropy) / (2 * max_entropy)
        
        return float(normalized_entropy)
        
    def _calculate_agreement(
        self, 
        predictions: List[PredictionResult]
    ) -> float:
        """
        Calculate inter-model agreement score
        """
        
        if len(predictions) <= 1:
            return 1.0
            
        # Check fruit type agreement
        fruit_types = [p.fruit_type for p in predictions]
        fruit_agreement = fruit_types.count(fruit_types[0]) / len(fruit_types)
        
        # Check ripeness agreement
        ripeness_stages = [p.ripeness for p in predictions]
        ripeness_agreement = ripeness_stages.count(ripeness_stages[0]) / len(ripeness_stages)
        
        # Combined score
        return (fruit_agreement + ripeness_agreement) / 2
        
    def _should_flag_for_annotation(
        self, 
        final_prediction: Dict, 
        uncertainty: float
    ) -> bool:
        """
        Determine if prediction should be sent for human annotation
        """
        
        low_confidence = final_prediction['confidence'] < 0.85
        high_uncertainty = uncertainty > 0.4
        low_agreement = final_prediction.get('model_agreement', 1.0) < 0.7
        
        return low_confidence or high_uncertainty or low_agreement
        
    def _calculate_grade(
        self, 
        ripeness: str, 
        defects: List[str], 
        freshness: float
    ) -> str:
        """
        Calculate quality grade A/B/C based on ripeness, defects, freshness
        """
        
        if ripeness == 'ripe' and not defects and freshness >= 85:
            return 'A'
        elif ripeness in ['ripe', 'unripe'] and len(defects) <= 1 and freshness >= 70:
            return 'B'
        else:
            return 'C'
            
    def _estimate_complexity(self, image: np.ndarray) -> float:
        """
        Estimate image complexity using edge density and texture
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture complexity (std of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = laplacian.std()
        
        # Normalize and combine
        complexity = (edge_density * 0.5 + 
                     min(texture_complexity / 50, 1.0) * 0.5)
                     
        return complexity
        
    async def _run_single_model(
        self, 
        model, 
        image: np.ndarray, 
        nir_image: Optional[np.ndarray]
    ) -> PredictionResult:
        """
        Run a single model and return standardized result
        """
        
        # This would be implemented based on specific model interface
        # Placeholder for demonstration
        
        result = await model.predict(image, nir_image)
        
        return PredictionResult(
            model_name=model.name,
            fruit_type=result['fruit_type'],
            ripeness=result['ripeness'],
            confidence=result['confidence'],
            defects=result.get('defects', []),
            freshness_score=result.get('freshness', 85.0),
            shelf_life_days=result.get('shelf_life', 7),
            bbox=result.get('bbox'),
            features=result.get('features')
        )
        
    async def _predict_lightweight(
        self, 
        image: np.ndarray, 
        nir_image: Optional[np.ndarray]
    ) -> List[PredictionResult]:
        """Use only lightweight models (MobileNet, etc.)"""
        
        lightweight_models = [
            name for name, config in self.model_configs.items()
            if 'mobilenet' in name.lower() or 'inception' in name.lower()
        ]
        
        tasks = [
            self._run_single_model(self.models[name], image, nir_image)
            for name in lightweight_models
            if self.model_configs[name].enabled
        ]
        
        return await asyncio.gather(*tasks)
        
    def update_model_weights(self, performance_metrics: Dict[str, float]):
        """
        Dynamically adjust model weights based on recent performance
        """
        
        for model_name, accuracy in performance_metrics.items():
            if model_name in self.model_configs:
                # Adaptive weight: higher accuracy = higher weight
                new_weight = accuracy / max(performance_metrics.values())
                self.model_configs[model_name].weight = new_weight
                
        # Normalize weights
        total_weight = sum(cfg.weight for cfg in self.model_configs.values())
        for cfg in self.model_configs.values():
            cfg.weight /= total_weight
