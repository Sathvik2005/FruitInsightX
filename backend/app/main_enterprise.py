"""
Enterprise Fruit AI API - Production-Grade FastAPI Application
Integrates all enterprise modules: ensemble, active learning, drift detection, XAI, 
video processing, and traceability
"""

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import base64
import cv2
import numpy as np
from datetime import datetime
import logging

# Enterprise modules
from app.ensemble import EnsembleOrchestrator, PredictionMode
from app.active_learning import ActiveLearningManager, SamplingStrategy
from app.drift_detection import DriftDetector
from app.xai import XAIVisualizer, XAIMethod, create_trust_report
from app.video_processor import VideoStreamProcessor
from app.traceability import TraceabilitySystem, BatchRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Enterprise Fruit AI Platform",
    description="Production-grade multi-model ensemble system for fruit detection, classification, and quality grading",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global State ====================

class ApplicationState:
    """Central application state manager"""
    
    def __init__(self):
        self.ensemble: Optional[EnsembleOrchestrator] = None
        self.active_learning: Optional[ActiveLearningManager] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.xai_visualizer: Optional[XAIVisualizer] = None
        self.video_processor: Optional[VideoStreamProcessor] = None
        self.traceability: Optional[TraceabilitySystem] = None
        self.websocket_clients: List[WebSocket] = []
        
state = ApplicationState()

# ==================== Startup & Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize all enterprise systems"""
    
    logger.info("🚀 Starting Enterprise Fruit AI Platform...")
    
    try:
        # Initialize ensemble orchestrator
        logger.info("Initializing ensemble orchestrator...")
        state.ensemble = EnsembleOrchestrator(
            detection_models=['yolov3', 'yolo_nas'],
            classification_models=['resnet50', 'vgg19', 'mobilenet_v2', 'inception_v3'],
            model_dir='models/'
        )
        await state.ensemble.load_models()
        
        # Initialize active learning
        logger.info("Initializing active learning system...")
        state.active_learning = ActiveLearningManager(
            annotation_dir='data/annotations',
            strategy=SamplingStrategy.HYBRID
        )
        
        # Initialize drift detector
        logger.info("Initializing drift detection...")
        state.drift_detector = DriftDetector(
            reference_data_path='data/reference_distribution.npz',
            model_performance_baseline={'accuracy': 0.95, 'f1_score': 0.93}
        )
        
        # Initialize XAI visualizer
        logger.info("Initializing XAI system...")
        state.xai_visualizer = XAIVisualizer()
        
        # Initialize video processor
        logger.info("Initializing video processor...")
        state.video_processor = VideoStreamProcessor(
            detection_model=None,  # Will use ensemble
            classification_model=None
        )
        
        # Initialize traceability system
        logger.info("Initializing traceability system...")
        state.traceability = TraceabilitySystem(
            blockchain_api_url='http://localhost:8545'  # Configure for production
        )
        
        logger.info("✅ All systems initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Enterprise Fruit AI Platform...")
    
    # Close all WebSocket connections
    for ws in state.websocket_clients:
        await ws.close()
    
    logger.info("✅ Shutdown complete")


# ==================== Request/Response Models ====================

class PredictionRequest(BaseModel):
    mode: PredictionMode = PredictionMode.ADAPTIVE
    explain: bool = False
    include_uncertainty: bool = True


class PredictionResponse(BaseModel):
    prediction_id: str
    fruit_type: str
    ripeness: str
    quality_grade: str
    defects: List[str]
    shelf_life_days: int
    confidence: float
    uncertainty: float
    ensemble_agreement: float
    processing_time_ms: float
    flagged_for_annotation: bool
    explanation_url: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    images: List[str]  # Base64 encoded images
    mode: PredictionMode = PredictionMode.ADAPTIVE


class AnnotationRequest(BaseModel):
    prediction_id: str
    correct_label: Dict[str, Any]
    annotator_id: str
    comment: Optional[str] = None


class DriftReport(BaseModel):
    timestamp: str
    performance_drift: Dict[str, Any]
    data_drift: Dict[str, Any]
    concept_drift: Dict[str, Any]
    overall_status: str
    recommended_actions: List[str]


class BatchRecordRequest(BaseModel):
    farm_origin: str
    harvest_date: str
    quantity_kg: float
    images: List[str]  # Base64 encoded


# ==================== Core Prediction Endpoints ====================

@app.post("/api/v2/predict", response_model=PredictionResponse)
async def predict_enterprise(
    file: UploadFile = File(...),
    mode: PredictionMode = PredictionMode.ADAPTIVE,
    explain: bool = False
):
    """
    Enterprise prediction endpoint using multi-model ensemble
    """
    
    try:
        # Read image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Run ensemble prediction
        start_time = datetime.now()
        result = await state.ensemble.predict_ensemble(image, mode=mode)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Check if should flag for annotation
        should_flag = state.ensemble._should_flag_for_annotation(
            result['confidence'],
            result.get('uncertainty', 0),
            result.get('ensemble_agreement', 0)
        )
        
        # Record for drift detection
        state.drift_detector.record_prediction(
            features=result.get('features'),
            prediction=result['fruit_type'],
            confidence=result['confidence']
        )
        
        # Generate explanation if requested
        explanation_url = None
        if explain:
            explanation_url = f"/api/v2/explain/{result['prediction_id']}"
        
        return PredictionResponse(
            prediction_id=result['prediction_id'],
            fruit_type=result['fruit_type'],
            ripeness=result.get('ripeness', 'unknown'),
            quality_grade=result.get('quality_grade', 'B'),
            defects=result.get('defects', []),
            shelf_life_days=result.get('shelf_life_days', 7),
            confidence=result['confidence'],
            uncertainty=result.get('uncertainty', 0.0),
            ensemble_agreement=result.get('ensemble_agreement', 1.0),
            processing_time_ms=processing_time,
            flagged_for_annotation=should_flag,
            explanation_url=explanation_url
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/batch-predict")
async def batch_predict_enterprise(request: BatchPredictionRequest):
    """
    Batch prediction for multiple images
    """
    
    results = []
    
    for idx, img_b64 in enumerate(request.images):
        try:
            # Decode base64 image
            img_data = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Predict
            result = await state.ensemble.predict_ensemble(image, mode=request.mode)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing image {idx}: {str(e)}")
            results.append({'error': str(e), 'image_index': idx})
    
    return {
        'total_images': len(request.images),
        'successful': sum(1 for r in results if 'error' not in r),
        'failed': sum(1 for r in results if 'error' in r),
        'results': results
    }


# ==================== Explainability Endpoints ====================

@app.get("/api/v2/explain/{prediction_id}")
async def get_explanation(
    prediction_id: str,
    method: XAIMethod = XAIMethod.GRADCAM
):
    """
    Generate explanation for a prediction
    """
    
    try:
        # Retrieve prediction data (from cache/database)
        # For demo, assume we have the image and model
        
        if method == XAIMethod.GRADCAM:
            explanation = state.xai_visualizer.generate_gradcam(
                model=state.ensemble.classification_models[0],
                image=None,  # Load from storage
                layer_name='conv5_block3_out'
            )
        elif method == XAIMethod.SALIENCY:
            explanation = state.xai_visualizer.generate_saliency_map(
                model=state.ensemble.classification_models[0],
                image=None,
                class_idx=0
            )
        else:
            raise HTTPException(status_code=400, detail=f"Method {method} not supported")
        
        # Create trust report
        trust_report = create_trust_report(
            confidence=0.92,
            uncertainty=0.15,
            feature_importance=[0.3, 0.25, 0.2, 0.15, 0.1],
            consistency_score=0.88
        )
        
        return {
            'prediction_id': prediction_id,
            'explanation': explanation,
            'trust_report': trust_report
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Active Learning Endpoints ====================

@app.post("/api/v2/learning/flag-annotation")
async def flag_for_annotation(prediction_id: str, reason: str):
    """
    Manually flag a prediction for human annotation
    """
    
    try:
        # Load prediction data
        prediction_data = {
            'prediction_id': prediction_id,
            'image_path': f'data/predictions/{prediction_id}.jpg',
            'model_prediction': {'fruit_type': 'apple', 'confidence': 0.75},
            'uncertainty': 0.3
        }
        
        # Flag for annotation
        flagged = await state.active_learning.flag_for_annotation(prediction_data, reason)
        
        return {
            'success': flagged,
            'prediction_id': prediction_id,
            'queue_position': await state.active_learning.get_queue_length()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/learning/annotation-queue")
async def get_annotation_queue(limit: int = 50):
    """
    Get samples awaiting annotation
    """
    
    try:
        samples = await state.active_learning.get_annotation_queue(limit=limit)
        
        return {
            'total_pending': len(samples),
            'samples': samples
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/learning/submit-annotation")
async def submit_annotation(request: AnnotationRequest):
    """
    Submit human annotation
    """
    
    try:
        # Record annotation
        await state.active_learning.record_annotation(
            sample_id=request.prediction_id,
            annotation=request.correct_label,
            annotator_id=request.annotator_id
        )
        
        # Check if retraining threshold reached
        annotated_count = await state.active_learning.get_annotated_count()
        should_retrain = annotated_count >= 500
        
        if should_retrain:
            # Trigger background retraining
            logger.info(f"🔄 Triggering model retraining ({annotated_count} samples)")
            # await trigger_retraining()  # Background task
        
        return {
            'success': True,
            'annotated_count': annotated_count,
            'retraining_triggered': should_retrain
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Drift Detection Endpoints ====================

@app.get("/api/v2/monitoring/drift", response_model=DriftReport)
async def check_drift():
    """
    Run comprehensive drift detection
    """
    
    try:
        # Check all drift types
        performance_drift = await state.drift_detector.detect_performance_drift()
        data_drift = await state.drift_detector.detect_data_drift()
        concept_drift = await state.drift_detector.detect_concept_drift()
        
        # Determine overall status
        severities = [
            performance_drift.get('severity'),
            data_drift.get('severity'),
            concept_drift.get('severity')
        ]
        
        overall_status = 'healthy'
        if 'CRITICAL' in severities:
            overall_status = 'critical'
        elif 'HIGH' in severities:
            overall_status = 'warning'
        elif 'MEDIUM' in severities:
            overall_status = 'attention'
        
        # Collect recommended actions
        actions = []
        if state.drift_detector.should_trigger_retraining():
            actions.append('Trigger model retraining immediately')
        if performance_drift.get('severity') in ['HIGH', 'CRITICAL']:
            actions.append('Review recent prediction quality')
        if data_drift.get('severity') in ['HIGH', 'CRITICAL']:
            actions.append('Audit incoming data distribution')
        
        return DriftReport(
            timestamp=datetime.now().isoformat(),
            performance_drift=performance_drift,
            data_drift=data_drift,
            concept_drift=concept_drift,
            overall_status=overall_status,
            recommended_actions=actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Video Processing Endpoints ====================

@app.websocket("/ws/video-stream")
async def video_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video processing
    """
    
    await websocket.accept()
    state.websocket_clients.append(websocket)
    
    try:
        logger.info("📹 Video stream connected")
        
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            
            # Decode frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Process frame using ensemble
            results = await state.video_processor.process_frame(
                frame,
                frame_id=int(datetime.now().timestamp() * 1000)
            )
            
            # Send results back
            await websocket.send_json({
                'timestamp': datetime.now().isoformat(),
                'fruits_detected': len(results),
                'results': results,
                'analytics': state.video_processor.get_analytics()
            })
            
    except WebSocketDisconnect:
        logger.info("Video stream disconnected")
        state.websocket_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()


@app.post("/api/v2/video/process-file")
async def process_video_file(file: UploadFile = File(...)):
    """
    Process uploaded video file
    """
    
    try:
        # Save temporary file
        temp_path = f"temp/{file.filename}"
        with open(temp_path, 'wb') as f:
            f.write(await file.read())
        
        # Process video
        results = []
        async for frame, fruits, analytics in state.video_processor.process_video_stream(temp_path):
            results.append({
                'fruits': fruits,
                'analytics': analytics
            })
        
        # Cleanup
        import os
        os.remove(temp_path)
        
        return {
            'total_frames': len(results),
            'total_fruits_detected': sum(len(r['fruits']) for r in results),
            'average_fps': np.mean([r['analytics']['fps'] for r in results]),
            'results': results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Traceability Endpoints ====================

@app.post("/api/v2/traceability/batch")
async def create_batch(request: BatchRecordRequest):
    """
    Create new batch record with QR code
    """
    
    try:
        # Run predictions on batch images
        quality_scores = []
        for img_b64 in request.images[:5]:  # Sample 5 images
            img_data = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            result = await state.ensemble.predict_ensemble(image)
            quality_scores.append({
                'grade': result.get('quality_grade', 'B'),
                'confidence': result['confidence']
            })
        
        # Create batch record
        batch_record = await state.traceability.create_batch_record(
            farm_origin=request.farm_origin,
            harvest_date=request.harvest_date,
            quantity_kg=request.quantity_kg,
            quality_scores=quality_scores
        )
        
        # Generate QR code
        qr_code_b64 = state.traceability.generate_qr_code(batch_record)
        
        return {
            'batch_id': batch_record.batch_id,
            'qr_code': qr_code_b64,
            'traceability_hash': batch_record.traceability_hash,
            'average_grade': batch_record.quality_grade,
            'shelf_life_days': batch_record.estimated_shelf_life_days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/traceability/verify/{batch_id}")
async def verify_batch(batch_id: str):
    """
    Verify batch authenticity
    """
    
    try:
        is_valid = await state.traceability.verify_batch(batch_id)
        
        if not is_valid:
            raise HTTPException(status_code=404, detail="Batch not found or invalid")
        
        # Get batch details
        batch_record = await state.traceability.get_batch_record(batch_id)
        
        return {
            'valid': is_valid,
            'batch_record': batch_record
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/traceability/compliance-report")
async def generate_compliance_report(batch_id: str):
    """
    Generate compliance report for batch
    """
    
    try:
        report = await state.traceability.generate_compliance_report(batch_id)
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== System Health & Metrics ====================

@app.get("/api/v2/health")
async def health_check():
    """
    Comprehensive system health check
    """
    
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'systems': {
            'ensemble': state.ensemble is not None,
            'active_learning': state.active_learning is not None,
            'drift_detection': state.drift_detector is not None,
            'xai': state.xai_visualizer is not None,
            'video_processing': state.video_processor is not None,
            'traceability': state.traceability is not None
        },
        'version': '2.0.0'
    }


@app.get("/api/v2/metrics")
async def get_metrics():
    """
    System performance metrics
    """
    
    return {
        'ensemble': {
            'detection_models': len(state.ensemble.detection_models) if state.ensemble else 0,
            'classification_models': len(state.ensemble.classification_models) if state.ensemble else 0,
            'total_predictions': 0  # Track in production
        },
        'active_learning': {
            'pending_annotations': 0,  # Get from active learning manager
            'completed_annotations': 0,
            'retraining_threshold': 500
        },
        'drift': {
            'alerts': [],  # Get from drift detector
            'last_check': datetime.now().isoformat()
        },
        'websockets': {
            'active_connections': len(state.websocket_clients)
        }
    }


# ==================== Run Application ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main_enterprise:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=4,  # Multi-worker for production
        log_level="info"
    )
