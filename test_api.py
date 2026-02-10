"""
Comprehensive Testing Suite for Fruit Classifier
Tests backend API, model inference, and system integration
"""

import requests
import os
import json
import time
from pathlib import Path
from PIL import Image
import io
import numpy as np


class FruitClassifierTester:
    """Test the Fruit Classifier API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            passed = response.status_code == 200
            data = response.json()
            message = f"Status: {data.get('status')}, Model: {data.get('model_loaded')}"
            self.log_test("Health Endpoint", passed, message)
            return passed
        except Exception as e:
            self.log_test("Health Endpoint", False, str(e))
            return False
            
    def test_classes_endpoint(self):
        """Test classes endpoint"""
        try:
            response = requests.get(f"{self.base_url}/classes", timeout=5)
            passed = response.status_code == 200
            data = response.json()
            classes = data.get('classes', [])
            message = f"Found {len(classes)} classes"
            self.log_test("Classes Endpoint", passed, message)
            return passed
        except Exception as e:
            self.log_test("Classes Endpoint", False, str(e))
            return False
            
    def create_test_image(self, color=(255, 0, 0), size=(100, 100)):
        """Create a test image"""
        img = Image.new('RGB', size, color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
        
    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        try:
            # Create test image
            test_img = self.create_test_image()
            
            files = {'file': ('test.jpg', test_img, 'image/jpeg')}
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                timeout=30
            )
            
            passed = response.status_code == 200
            if passed:
                data = response.json()
                pred = data.get('prediction', 'N/A')
                conf = data.get('confidence', 0)
                message = f"Prediction: {pred}, Confidence: {conf:.2f}"
            else:
                message = f"Status code: {response.status_code}"
                
            self.log_test("Predict Endpoint", passed, message)
            return passed
        except Exception as e:
            self.log_test("Predict Endpoint", False, str(e))
            return False
            
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        try:
            # Create multiple test images
            files = []
            for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                test_img = self.create_test_image(color=color)
                files.append(('files', (f'test_{i}.jpg', test_img, 'image/jpeg')))
            
            response = requests.post(
                f"{self.base_url}/batch-predict",
                files=files,
                timeout=60
            )
            
            passed = response.status_code == 200
            if passed:
                data = response.json()
                results = data.get('results', [])
                message = f"Processed {len(results)} images"
            else:
                message = f"Status code: {response.status_code}"
                
            self.log_test("Batch Predict Endpoint", passed, message)
            return passed
        except Exception as e:
            self.log_test("Batch Predict Endpoint", False, str(e))
            return False
            
    def test_response_time(self):
        """Test API response time"""
        try:
            test_img = self.create_test_image()
            files = {'file': ('test.jpg', test_img, 'image/jpeg')}
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                timeout=30
            )
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            passed = elapsed < 1000  # Should be under 1 second
            message = f"Response time: {elapsed:.0f}ms"
            self.log_test("Response Time", passed, message)
            return passed
        except Exception as e:
            self.log_test("Response Time", False, str(e))
            return False
            
    def test_invalid_file(self):
        """Test with invalid file"""
        try:
            # Send text file as image
            files = {'file': ('test.txt', io.BytesIO(b'not an image'), 'text/plain')}
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                timeout=30
            )
            
            # Should return error (400 or 422)
            passed = response.status_code in [400, 422, 500]
            message = f"Status code: {response.status_code}"
            self.log_test("Invalid File Handling", passed, message)
            return passed
        except Exception as e:
            self.log_test("Invalid File Handling", False, str(e))
            return False
            
    def test_cors_headers(self):
        """Test CORS headers"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            cors_header = response.headers.get('access-control-allow-origin')
            passed = cors_header is not None
            message = f"CORS header: {cors_header}"
            self.log_test("CORS Headers", passed, message)
            return passed
        except Exception as e:
            self.log_test("CORS Headers", False, str(e))
            return False
            
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("🧪 Running Fruit Classifier Tests")
        print("="*60 + "\n")
        
        print("🔍 Testing Basic Endpoints...")
        self.test_health_endpoint()
        self.test_classes_endpoint()
        
        print("\n🤖 Testing Prediction...")
        self.test_predict_endpoint()
        self.test_batch_predict_endpoint()
        
        print("\n⚡ Testing Performance...")
        self.test_response_time()
        
        print("\n🛡️ Testing Error Handling...")
        self.test_invalid_file()
        
        print("\n🌐 Testing CORS...")
        self.test_cors_headers()
        
        # Summary
        print("\n" + "="*60)
        print("📊 Test Summary")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(1 for t in self.test_results if t['passed'])
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print("\n❌ Failed Tests:")
            for test in self.test_results:
                if not test['passed']:
                    print(f"  - {test['test']}: {test['message']}")
        
        print("\n" + "="*60)
        
        return failed == 0


def test_model_file():
    """Test if model file exists"""
    print("\n📦 Checking Model File...")
    model_path = Path("backend/models/fruit_classifier.h5")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model found: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Model not found: {model_path}")
        print("  Please train a model first:")
        print("  python backend/train_model.py --data_dir data/organized")
        return False


def test_frontend():
    """Test if frontend is accessible"""
    print("\n🌐 Checking Frontend...")
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        if response.status_code == 200:
            print("✓ Frontend is running at http://localhost:5173")
            return True
        else:
            print(f"✗ Frontend returned status code: {response.status_code}")
            return False
    except Exception as e:
        print("✗ Frontend is not accessible")
        print("  Start with: cd frontend && npm run dev")
        return False


def test_dependencies():
    """Test if all dependencies are installed"""
    print("\n📚 Checking Dependencies...")
    
    dependencies = {
        'fastapi': 'FastAPI',
        'tensorflow': 'TensorFlow',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'sklearn': 'scikit-learn'
    }
    
    all_installed = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} not installed")
            all_installed = False
    
    if not all_installed:
        print("\nInstall missing dependencies:")
        print("  cd backend")
        print("  pip install -r requirements.txt")
    
    return all_installed


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("🍎 Fruit Classifier - Test Suite")
    print("="*60)
    
    # Pre-flight checks
    print("\n🔍 Pre-flight Checks...")
    
    # Check backend is running
    try:
        requests.get("http://localhost:8000/health", timeout=2)
        print("✓ Backend is running")
    except:
        print("✗ Backend is not running")
        print("\nPlease start the backend first:")
        print("  cd backend")
        print("  python -m uvicorn app.main:app --reload")
        print("\nThen run tests again.")
        sys.exit(1)
    
    # Check dependencies
    # test_dependencies()
    
    # Check model
    # test_model_file()
    
    # Check frontend (optional)
    # test_frontend()
    
    # Run API tests
    tester = FruitClassifierTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
