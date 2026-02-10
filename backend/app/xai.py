"""
Explainable AI (XAI) Module
Generate visual explanations for model predictions
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import tensorflow as tf
from dataclasses import dataclass


@dataclass
class Explanation:
    """Container for explanation visualizations"""
    gradcam_heatmap: np.ndarray
    saliency_map: np.ndarray
    overlay_image: np.ndarray
    top_features: Dict
    confidence_breakdown: Dict


class XAIVisualizer:
    """
    Generate explainability visualizations:
    - Grad-CAM heatmaps
    - Saliency maps
    - Feature importance
    - Attention maps
    """
    
    def __init__(self, model):
        self.model = model
        self.layer_name = self._find_last_conv_layer()
        
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model"""
        
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
                
        # Fallback
        return self.model.layers[-2].name
        
    def generate_gradcam(
        self,
        image: np.ndarray,
        class_idx: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image (H, W, 3)
            class_idx: Target class index
            
        Returns:
            Heatmap array (H, W)
        """
        
        # Expand dims for batch
        img_tensor = np.expand_dims(image, axis=0)
        img_tensor = tf.cast(img_tensor, tf.float32)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_idx]
            
        # Get gradients of the loss w.r.t. conv layer outputs
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute guided gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize to match input image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        return heatmap
        
    def generate_saliency_map(
        self,
        image: np.ndarray,
        class_idx: int
    ) -> np.ndarray:
        """
        Generate saliency map showing pixel-level importance
        """
        
        img_tensor = tf.Variable(np.expand_dims(image, axis=0), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor)
            loss = predictions[:, class_idx]
            
        # Get gradients
        grads = tape.gradient(loss, img_tensor)
        grads = tf.abs(grads)
        
        # Convert to grayscale saliency
        saliency = tf.reduce_max(grads, axis=-1)[0]
        
        # Normalize
        saliency = saliency / tf.reduce_max(saliency)
        
        return saliency.numpy()
        
    def create_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create overlay of heatmap on original image
        """
        
        # Convert heatmap to color map
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Blend
        overlay = cv2.addWeighted(image_bgr, 1-alpha, heatmap_colored, alpha, 0)
        
        # Convert back to RGB
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        return overlay_rgb
        
    def analyze_feature_importance(
        self,
        image: np.ndarray,
        prediction: Dict
    ) -> Dict:
        """
        Analyze which features contributed most to prediction
        """
        
        # Get intermediate layer outputs
        feature_extractor = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[layer.output for layer in self.model.layers[:-1]]
        )
        
        features = feature_extractor(np.expand_dims(image, axis=0))
        
        # Analyze final layer activations
        final_features = features[-1][0].numpy()
        
        # Get top activating features
        top_indices = np.argsort(final_features)[-10:]
        
        return {
            'top_feature_indices': top_indices.tolist(),
            'top_feature_values': final_features[top_indices].tolist(),
            'feature_statistics': {
                'mean': float(np.mean(final_features)),
                'std': float(np.std(final_features)),
                'max': float(np.max(final_features))
            }
        }
        
    def generate_comprehensive_explanation(
        self,
        image: np.ndarray,
        prediction: Dict
    ) -> Explanation:
        """
        Generate complete explanation package
        """
        
        # Get predicted class index
        class_idx = prediction.get('class_idx', 0)
        
        # Generate visualizations
        gradcam = self.generate_gradcam(image, class_idx)
        saliency = self.generate_saliency_map(image, class_idx)
        overlay = self.create_overlay(image, gradcam)
        
        # Feature analysis
        features = self.analyze_feature_importance(image, prediction)
        
        # Confidence breakdown
        confidence_breakdown = {
            'prediction_confidence': prediction.get('confidence', 0.0),
            'entropy': self._calculate_entropy(prediction),
            'top_3_predictions': prediction.get('top_3', [])
        }
        
        return Explanation(
            gradcam_heatmap=gradcam,
            saliency_map=saliency,
            overlay_image=overlay,
            top_features=features,
            confidence_breakdown=confidence_breakdown
        )
        
    def _calculate_entropy(self, prediction: Dict) -> float:
        """Calculate prediction entropy (uncertainty measure)"""
        
        if 'all_predictions' in prediction:
            probs = np.array(list(prediction['all_predictions'].values()))
            probs = probs / np.sum(probs)  # Normalize
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return float(entropy)
            
        return 0.0
        
    def generate_attention_map(
        self,
        image: np.ndarray,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate attention map from specific layer
        """
        
        if layer_name is None:
            layer_name = self.layer_name
            
        # Create model up to target layer
        layer_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer(layer_name).output
        )
        
        # Get activations
        activations = layer_model(np.expand_dims(image, axis=0))
        
        # Average across channels
        attention = tf.reduce_mean(activations, axis=-1)[0]
        
        # Normalize
        attention = attention / tf.reduce_max(attention)
        
        # Resize to input size
        attention_resized = cv2.resize(
            attention.numpy(),
            (image.shape[1], image.shape[0])
        )
        
        return attention_resized


class SHAPExplainer:
    """
    SHAP-based explanations for model predictions
    """
    
    def __init__(self, model, background_data: np.ndarray):
        import shap
        
        self.model = model
        self.background_data = background_data
        self.explainer = shap.DeepExplainer(
            model,
            background_data[:100]  # Use subset for efficiency
        )
        
    def explain_prediction(self, image: np.ndarray) -> Dict:
        """
        Generate SHAP values for prediction
        """
        
        shap_values = self.explainer.shap_values(
            np.expand_dims(image, axis=0)
        )
        
        return {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value,
            'feature_importance': np.abs(shap_values).mean(axis=(0, 1, 2))
        }


class LIMEExplainer:
    """
    LIME-based local interpretable explanations
    """
    
    def __init__(self, model):
        from lime import lime_image
        
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()
        
    def explain_prediction(
        self,
        image: np.ndarray,
        num_samples: int = 1000
    ) -> Dict:
        """
        Generate LIME explanation
        """
        
        def predict_fn(images):
            return self.model.predict(images)
            
        explanation = self.explainer.explain_instance(
            image,
            predict_fn,
            top_labels=3,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get mask for top predicted class
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        return {
            'important_regions': mask,
            'top_features': explanation.top_labels,
            'local_prediction': explanation.local_pred
        }


def create_trust_report(
    prediction: Dict,
    explanation: Explanation
) -> Dict:
    """
    Generate trust and interpretability report for auditors
    """
    
    # Calculate trust score based on multiple factors
    confidence = prediction['confidence']
    entropy = explanation.confidence_breakdown['entropy']
    feature_consistency = np.mean([
        v for v in explanation.top_features['top_feature_values']
    ])
    
    # Trust score (0-100)
    trust_score = (
        confidence * 40 +  # 40% weight on confidence
        (1 - entropy / 3) * 30 +  # 30% weight on low entropy
        min(feature_consistency * 10, 30)  # 30% weight on features
    )
    
    report = {
        'trust_score': float(trust_score),
        'confidence_level': 'high' if confidence > 0.9 else 'medium' if confidence > 0.75 else 'low',
        'uncertainty_level': 'low' if entropy < 0.5 else 'medium' if entropy < 1.0 else 'high',
        'explanation_quality': 'good' if feature_consistency > 2.0 else 'fair',
        'recommendation': _get_recommendation(trust_score, confidence, entropy),
        'audit_trail': {
            'prediction': prediction,
            'features_analyzed': len(explanation.top_features['top_feature_indices']),
            'visualization_types': ['gradcam', 'saliency', 'overlay']
        }
    }
    
    return report


def _get_recommendation(trust_score: float, confidence: float, entropy: float) -> str:
    """Generate recommendation based on trust metrics"""
    
    if trust_score > 80 and confidence > 0.9:
        return "High confidence - accept prediction"
    elif trust_score > 60:
        return "Medium confidence - consider human review for critical applications"
    else:
        return "Low confidence - human review required"
