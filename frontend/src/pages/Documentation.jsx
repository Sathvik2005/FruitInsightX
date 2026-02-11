import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import './Documentation.css'

const Documentation = () => {
  const containerRef = useRef(null)

  useEffect(() => {
    if (containerRef.current) {
      gsap.from(containerRef.current.querySelectorAll('.doc-section'), {
        y: 30,
        opacity: 0,
        duration: 0.6,
        stagger: 0.15,
        ease: 'power2.out'
      })
    }
  }, [])

  return (
    <div className="documentation-page" ref={containerRef}>
      <div className="doc-hero">
        <h1 className="page-title">Documentation</h1>
        <p className="page-subtitle">
          Complete guide to using FruitInsightX API and web interface
        </p>
      </div>

      <div className="doc-container">
        <aside className="doc-sidebar">
          <nav className="doc-nav">
            <a href="#quick-start">Quick Start</a>
            <a href="#installation">Installation</a>
            <a href="#api-endpoints">API Endpoints</a>
            <a href="#authentication">Authentication</a>
            <a href="#examples">Code Examples</a>
            <a href="#error-handling">Error Handling</a>
            <a href="#rate-limits">Rate Limits</a>
          </nav>
        </aside>

        <main className="doc-content">
          <section className="doc-section" id="quick-start">
            <h2>Quick Start</h2>
            <p>Get started with FruitInsightX in minutes:</p>
            <div className="code-block">
              <pre><code>{`# Install required packages
pip install requests pillow

# Import libraries
import requests
from PIL import Image

# Classify an image
url = "http://localhost:8000/predict"
files = {"file": open("apple.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())`}</code></pre>
            </div>
          </section>

          <section className="doc-section" id="installation">
            <h2>Installation</h2>
            <h3>Backend Setup</h3>
            <div className="code-block">
              <pre><code>{`cd backend
python -m venv venv
venv\\Scripts\\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`}</code></pre>
            </div>
            <h3>Frontend Setup</h3>
            <div className="code-block">
              <pre><code>{`cd frontend
npm install
npm run dev`}</code></pre>
            </div>
          </section>

          <section className="doc-section" id="api-endpoints">
            <h2>API Endpoints</h2>
            
            <div className="endpoint-card">
              <div className="endpoint-header">
                <span className="http-method get">GET</span>
                <span className="endpoint-path">/</span>
              </div>
              <p>Returns API information and version</p>
              <div className="code-block">
                <pre><code>{`curl http://localhost:8000/`}</code></pre>
              </div>
            </div>

            <div className="endpoint-card">
              <div className="endpoint-header">
                <span className="http-method post">POST</span>
                <span className="endpoint-path">/predict</span>
              </div>
              <p>Classify a single fruit image</p>
              <div className="code-block">
                <pre><code>{`curl -X POST http://localhost:8000/predict \\
  -F "file=@apple.jpg"`}</code></pre>
              </div>
              <h4>Response</h4>
              <div className="code-block">
                <pre><code>{`{
  "success": true,
  "prediction": {
    "class": "Apple",
    "confidence": 0.9856,
    "confidence_level": "Very High",
    "top_5_predictions": [...]
  },
  "metadata": {
    "processing_time_ms": 45.23,
    "model_version": "1.0.0",
    "timestamp": "2026-02-11T..."
  }
}`}</code></pre>
              </div>
            </div>

            <div className="endpoint-card">
              <div className="endpoint-header">
                <span className="http-method post">POST</span>
                <span className="endpoint-path">/predict/batch</span>
              </div>
              <p>Classify multiple images in one request</p>
              <div className="code-block">
                <pre><code>{`curl -X POST http://localhost:8000/predict/batch \\
  -F "files=@apple.jpg" \\
  -F "files=@banana.jpg"`}</code></pre>
              </div>
            </div>

            <div className="endpoint-card">
              <div className="endpoint-header">
                <span className="http-method get">GET</span>
                <span className="endpoint-path">/health</span>
              </div>
              <p>Check system health and model status</p>
            </div>

            <div className="endpoint-card">
              <div className="endpoint-header">
                <span className="http-method get">GET</span>
                <span className="endpoint-path">/classes</span>
              </div>
              <p>Get list of supported fruit classes</p>
            </div>
          </section>

          <section className="doc-section" id="authentication">
            <h2>Authentication</h2>
            <p>
              Currently, the API does not require authentication. For production deployments,
              consider implementing API keys or OAuth2 authentication.
            </p>
          </section>

          <section className="doc-section" id="examples">
            <h2>Code Examples</h2>
            
            <h3>Python Example</h3>
            <div className="code-block">
              <pre><code>{`import requests
from PIL import Image

def classify_fruit(image_path):
    url = "http://localhost:8000/predict"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Fruit: {result['prediction']['class']}")
        print(f"Confidence: {result['prediction']['confidence']:.2%}")
        return result
    else:
        print(f"Error: {response.status_code}")
        return None

# Usage
classify_fruit("apple.jpg")`}</code></pre>
            </div>

            <h3>JavaScript Example</h3>
            <div className="code-block">
              <pre><code>{`async function classifyFruit(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    console.log('Fruit:', result.prediction.class);
    console.log('Confidence:', result.prediction.confidence);
    return result;
  } catch (error) {
    console.error('Error:', error);
  }
}

// Usage with file input
const fileInput = document.querySelector('input[type="file"]');
fileInput.addEventListener('change', (e) => {
  classifyFruit(e.target.files[0]);
});`}</code></pre>
            </div>
          </section>

          <section className="doc-section" id="error-handling">
            <h2>Error Handling</h2>
            <p>The API returns structured error responses:</p>
            <div className="code-block">
              <pre><code>{`{
  "success": false,
  "detail": {
    "message": "Invalid image format",
    "code": "INVALID_FORMAT",
    "suggestion": "Upload JPEG or PNG image"
  }
}`}</code></pre>
            </div>
            <h3>Common Error Codes</h3>
            <ul>
              <li><code>INVALID_FORMAT</code> - Unsupported image format</li>
              <li><code>FILE_TOO_LARGE</code> - Image exceeds size limit</li>
              <li><code>MODEL_ERROR</code> - Model processing failed</li>
              <li><code>NO_FILE</code> - No file provided in request</li>
            </ul>
          </section>

          <section className="doc-section" id="rate-limits">
            <h2>Rate Limits</h2>
            <p>
              Current implementation does not enforce rate limits. For production use,
              implement rate limiting based on your infrastructure requirements.
            </p>
            <p>Recommended limits:</p>
            <ul>
              <li>Single predictions: 60 requests per minute</li>
              <li>Batch predictions: 10 requests per minute</li>
              <li>Maximum batch size: 10 images</li>
            </ul>
          </section>
        </main>
      </div>
    </div>
  )
}

export default Documentation
