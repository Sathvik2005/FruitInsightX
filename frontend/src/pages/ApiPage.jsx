import { useState, useEffect } from 'react'
import axios from 'axios'
import './ApiPage.css'

const ApiPage = () => {
  const [apiStatus, setApiStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [testResult, setTestResult] = useState(null)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  useEffect(() => {
    checkApiHealth()
  }, [])

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`)
      setApiStatus(response.data)
      setLoading(false)
    } catch (error) {
      setApiStatus({ status: 'unavailable', error: error.message })
      setLoading(false)
    }
  }

  const testEndpoint = async (endpoint) => {
    setTestResult({ loading: true, endpoint })
    try {
      const response = await axios.get(`${API_URL}${endpoint}`)
      setTestResult({
        success: true,
        endpoint,
        status: response.status,
        data: response.data
      })
    } catch (error) {
      setTestResult({
        success: false,
        endpoint,
        error: error.message
      })
    }
  }

  return (
    <div className="api-page">
      <div className="api-hero">
        <h1 className="page-title">API Testing Console</h1>
        <p className="page-subtitle">
          Interactive API testing and monitoring dashboard
        </p>
      </div>

      <div className="api-container">
        <section className="status-section">
          <h2>API Status</h2>
          {loading ? (
            <div className="status-loading">Checking API status...</div>
          ) : (
            <div className={`status-card ${apiStatus?.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
              <div className="status-indicator">
                <div className={`indicator-dot ${apiStatus?.status === 'healthy' ? 'green' : 'red'}`}></div>
                <span className="status-text">
                  {apiStatus?.status === 'healthy' ? 'API Online' : 'API Offline'}
                </span>
              </div>
              
              {apiStatus?.status === 'healthy' && (
                <div className="status-details">
                  <div className="detail-row">
                    <span className="detail-label">Model Status:</span>
                    <span className="detail-value">{apiStatus.model?.loaded ? 'Loaded' : 'Not Loaded'}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Model Path:</span>
                    <span className="detail-value">{apiStatus.model?.path}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Input Shape:</span>
                    <span className="detail-value">{apiStatus.model?.input_shape}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Classes:</span>
                    <span className="detail-value">{apiStatus.model?.output_classes}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">TensorFlow:</span>
                    <span className="detail-value">{apiStatus.system?.tensorflow_version}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">GPU Available:</span>
                    <span className="detail-value">{apiStatus.system?.gpu_available ? 'Yes' : 'No'}</span>
                  </div>
                </div>
              )}
              
              <button onClick={checkApiHealth} className="refresh-btn">
                Refresh Status
              </button>
            </div>
          )}
        </section>

        <section className="test-section">
          <h2>Test Endpoints</h2>
          <div className="endpoint-tests">
            <div className="test-card">
              <div className="test-header">
                <span className="method get">GET</span>
                <span className="endpoint">/</span>
              </div>
              <p className="test-description">Get API information and version</p>
              <button onClick={() => testEndpoint('/')} className="test-btn">
                Test Endpoint
              </button>
            </div>

            <div className="test-card">
              <div className="test-header">
                <span className="method get">GET</span>
                <span className="endpoint">/health</span>
              </div>
              <p className="test-description">Check system health and model status</p>
              <button onClick={() => testEndpoint('/health')} className="test-btn">
                Test Endpoint
              </button>
            </div>

            <div className="test-card">
              <div className="test-header">
                <span className="method get">GET</span>
                <span className="endpoint">/classes</span>
              </div>
              <p className="test-description">Get list of supported fruit classes</p>
              <button onClick={() => testEndpoint('/classes')} className="test-btn">
                Test Endpoint
              </button>
            </div>
          </div>
        </section>

        {testResult && (
          <section className="result-section">
            <h2>Test Results</h2>
            <div className="result-card">
              <div className="result-header">
                <span className="result-endpoint">{testResult.endpoint}</span>
                {testResult.success ? (
                  <span className="result-status success">Success ({testResult.status})</span>
                ) : (
                  <span className="result-status error">Error</span>
                )}
              </div>
              <div className="result-body">
                <pre>{JSON.stringify(testResult.data || testResult.error, null, 2)}</pre>
              </div>
            </div>
          </section>
        )}

        <section className="info-section">
          <h2>Quick Integration</h2>
          <div className="code-example">
            <h3>cURL Example</h3>
            <div className="code-block">
              <pre><code>{`curl -X POST ${API_URL}/predict \\
  -F "file=@fruit.jpg"`}</code></pre>
            </div>
          </div>
          
          <div className="code-example">
            <h3>JavaScript/Axios Example</h3>
            <div className="code-block">
              <pre><code>{`const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await axios.post('${API_URL}/predict', formData);
console.log(response.data);`}</code></pre>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default ApiPage
