"""
Fruit Classifier API
FastAPI backend for fruit image classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import Dict, List
import uvicorn

app = FastAPI(
    title="Fruit Classifier API",
    description="AI-powered fruit classification system",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
IMAGE_SIZE = 100
MODEL_PATH = "models/fruit_classifier.h5"

# Class names for fruits (matches trained model)
CLASS_NAMES = [
    "Apple", "Banana", "Cherry", "Grape", "Guava",
    "Kiwi", "Mango", "Orange", "Peach", "Pear", "Strawberry"
]

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Warning: Could not load model - {e}")
    model = None


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🍎 Fruit Classifier API - AI-Powered Fruit Recognition",
        "version": "2.0.0",
        "status": "operational",
        "model": {
            "loaded": model is not None,
            "type": "TensorFlow/Keras CNN",
            "input_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}x3",
            "classes": len(CLASS_NAMES)
        },
        "api": {
            "endpoints": {
                "predict": {
                    "path": "/predict",
                    "method": "POST",
                    "description": "Classify a single fruit image"
                },
                "batch_predict": {
                    "path": "/batch-predict",
                    "method": "POST",
                    "description": "Classify multiple images (max 10)"
                },
                "health": {
                    "path": "/health",
                    "method": "GET",
                    "description": "Check API health status"
                },
                "classes": {
                    "path": "/classes",
                    "method": "GET",
                    "description": "Get list of supported fruits"
                }
            }
        },
        "documentation": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed system information"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "timestamp": str(np.datetime64('now')),
        "model": {
            "loaded": model is not None,
            "path": MODEL_PATH,
            "input_shape": f"{IMAGE_SIZE}x{IMAGE_SIZE}x3",
            "output_classes": len(CLASS_NAMES)
        },
        "system": {
            "tensorflow_version": tf.__version__,
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
        },
        "capabilities": {
            "single_prediction": True,
            "batch_prediction": True,
            "max_batch_size": 10
        }
    }


@app.get("/classes")
async def get_classes():
    """Get list of supported fruit classes with additional metadata"""
    return {
        "success": True,
        "count": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "categories": {
            "all": CLASS_NAMES,
            "berries": ["Cherry", "Raspberry", "Redcurrant"],
            "tropical": ["Pineapple", "Guava", "Rambutan", "Salak", "Lychee"],
            "common": ["Apple", "Banana", "Grape"]
        },
        "model_info": {
            "input_size": IMAGE_SIZE,
            "supported_formats": ["jpg", "jpeg", "png", "webp"]
        }
    }


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed numpy array
    """
    # Resize image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict fruit class from uploaded image with detailed analysis
    
    Args:
        file: Uploaded image file
        
    Returns:
        Comprehensive prediction results with confidence scores
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Model not available",
                "message": "The classification model is not loaded. Please contact the administrator.",
                "code": "MODEL_NOT_LOADED"
            }
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "message": f"Expected image file, got {file.content_type}",
                "accepted_types": ["image/jpeg", "image/png", "image/jpg", "image/webp"],
                "code": "INVALID_FILE_TYPE"
            }
        )
    
    try:
        # Read and process image
        import time
        start_time = time.time()
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get original image info
        original_size = image.size
        image_format = image.format
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_image, verbose=0)[0]
        
        processing_time = time.time() - start_time
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        top_results = []
        for idx in top_indices:
            top_results.append({
                "rank": len(top_results) + 1,
                "class": CLASS_NAMES[idx],
                "confidence": round(float(predictions[idx]), 4),
                "percentage": round(float(predictions[idx] * 100), 2)
            })
        
        # Get top prediction
        top_class = CLASS_NAMES[np.argmax(predictions)]
        top_confidence = float(np.max(predictions))
        
        # Confidence level description
        if top_confidence >= 0.9:
            confidence_level = "Very High"
        elif top_confidence >= 0.75:
            confidence_level = "High"
        elif top_confidence >= 0.5:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        return {
            "success": True,
            "prediction": {
                "class": top_class,
                "confidence": round(top_confidence, 4),
                "percentage": round(top_confidence * 100, 2),
                "confidence_level": confidence_level
            },
            "top_5_predictions": top_results,
            "all_predictions": {
                CLASS_NAMES[i]: round(float(predictions[i]), 4)
                for i in range(len(CLASS_NAMES))
            },
            "metadata": {
                "original_size": original_size,
                "format": image_format,
                "processed_size": [IMAGE_SIZE, IMAGE_SIZE],
                "processing_time_ms": round(processing_time * 1000, 2),
                "model_version": "1.0.0"
            }
        }
    
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "message": str(e),
                "trace": traceback.format_exc() if __debug__ else "Enable debug mode for trace",
                "code": "PROCESSING_ERROR"
            }
        )


@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict multiple fruit images with comprehensive analysis
    
    Args:
        files: List of uploaded image files (max 10)
        
    Returns:
        Detailed predictions for each image
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not available",
                "message": "Classification model is not loaded",
                "code": "MODEL_NOT_LOADED"
            }
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Too many files",
                "message": f"Maximum 10 images allowed per batch. You uploaded {len(files)}",
                "max_allowed": 10,
                "received": len(files),
                "code": "BATCH_SIZE_EXCEEDED"
            }
        )
    
    import time
    batch_start_time = time.time()
    
    results = []
    success_count = 0
    error_count = 0
    
    for idx, file in enumerate(files, 1):
        try:
            start_time = time.time()
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image, verbose=0)[0]
            
            processing_time = time.time() - start_time
            
            top_class = CLASS_NAMES[np.argmax(predictions)]
            top_confidence = float(np.max(predictions))
            
            # Get top 3 for batch
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_3 = [
                {
                    "class": CLASS_NAMES[i],
                    "confidence": round(float(predictions[i]), 4),
                    "percentage": round(float(predictions[i] * 100), 2)
                }
                for i in top_indices
            ]
            
            results.append({
                "index": idx,
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "class": top_class,
                    "confidence": round(top_confidence, 4),
                    "percentage": round(top_confidence * 100, 2)
                },
                "top_3": top_3,
                "processing_time_ms": round(processing_time * 1000, 2)
            })
            success_count += 1
            
        except Exception as e:
            results.append({
                "index": idx,
                "filename": file.filename,
                "success": False,
                "error": str(e),
                "code": "PROCESSING_ERROR"
            })
            error_count += 1
    
    total_time = time.time() - batch_start_time
    
    return {
        "success": True,
        "batch_summary": {
            "total_files": len(files),
            "successful": success_count,
            "failed": error_count,
            "success_rate": round(success_count / len(files) * 100, 2) if files else 0,
            "total_processing_time_ms": round(total_time * 1000, 2),
            "average_time_per_image_ms": round(total_time / len(files) * 1000, 2) if files else 0
        },
        "results": results
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
