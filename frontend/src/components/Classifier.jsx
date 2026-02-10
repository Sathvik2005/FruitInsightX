import { useState, useRef, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { gsap } from 'gsap'
import { FaUpload, FaSpinner, FaCheckCircle } from 'react-icons/fa'
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js'
import { Bar } from 'react-chartjs-2'
import axios from 'axios'
import './Classifier.css'

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement)

const Classifier = () => {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const dropzoneRef = useRef(null)
  const resultRef = useRef(null)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  useEffect(() => {
    // Animate on scroll
    if (dropzoneRef.current) {
      gsap.from(dropzoneRef.current, {
        scrollTrigger: {
          trigger: dropzoneRef.current,
          start: 'top 80%'
        },
        y: 100,
        opacity: 0,
        duration: 1,
        ease: 'power3.out'
      })
    }
  }, [])

  useEffect(() => {
    if (result && resultRef.current) {
      gsap.from(resultRef.current, {
        scale: 0.8,
        opacity: 0,
        duration: 0.5,
        ease: 'back.out(1.7)'
      })
    }
  }, [result])

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (file) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    multiple: false
  })

  const handlePredict = async () => {
    if (!image) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', image)

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      // Handle both old and new response formats
      const data = response.data
      const formattedResult = {
        success: data.success,
        prediction: data.prediction,
        top_predictions: data.top_5_predictions || data.top_predictions || [],
        all_predictions: data.all_predictions,
        metadata: data.metadata
      }
      
      setResult(formattedResult)
      
      // Success animation
      gsap.from('.result-card', {
        scale: 0.5,
        rotation: 360,
        duration: 0.6,
        ease: 'back.out(1.7)'
      })
    } catch (err) {
      const errorMsg = err.response?.data?.detail?.message || 
                       err.response?.data?.detail || 
                       err.response?.data?.error ||
                       'Failed to classify image. Please try again.'
      setError(errorMsg)
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getChartData = () => {
    if (!result?.top_predictions) return null

    return {
      labels: result.top_predictions.map(p => p.class),
      datasets: [{
        label: 'Confidence (%)',
        data: result.top_predictions.map(p => p.percentage),
        backgroundColor: [
          'rgba(76, 175, 80, 0.8)',
          'rgba(33, 150, 243, 0.8)',
          'rgba(255, 193, 7, 0.8)',
          'rgba(156, 39, 176, 0.8)',
          'rgba(244, 67, 54, 0.8)'
        ],
        borderColor: [
          'rgb(76, 175, 80)',
          'rgb(33, 150, 243)',
          'rgb(255, 193, 7)',
          'rgb(156, 39, 176)',
          'rgb(244, 67, 54)'
        ],
        borderWidth: 2,
        borderRadius: 8
      }]
    }
  }

  return (
    <section className="classifier" id="classify">
      <div className="classifier-container">
        <h2 className="section-title">Classify Your Fruit</h2>
        <p className="section-subtitle">
          Upload an image of a fruit and our AI will identify it instantly
        </p>

        <div className="classifier-grid">
          {/* Upload Section */}
          <div className="upload-section" ref={dropzoneRef}>
            <div 
              {...getRootProps()} 
              className={`dropzone ${isDragActive ? 'active' : ''}`}
            >
              <input {...getInputProps()} />
              
              {preview ? (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-image" />
                  <div className="preview-overlay">
                    <p>Click or drag to change image</p>
                  </div>
                </div>
              ) : (
                <div className="dropzone-content">
                  <FaUpload className="upload-icon" />
                  <h3>Drop your image here</h3>
                  <p>or click to browse</p>
                  <span className="file-types">Supports: JPG, PNG, GIF</span>
                </div>
              )}
            </div>

            <button
              className={`predict-button ${!image || loading ? 'disabled' : ''}`}
              onClick={handlePredict}
              disabled={!image || loading}
            >
              {loading ? (
                <>
                  <FaSpinner className="spinner" />
                  Analyzing...
                </>
              ) : (
                'Classify Fruit'
              )}
            </button>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="results-section" ref={resultRef}>
            {result ? (
              <div className="result-card">
                <div className="result-header">
                  <FaCheckCircle className="success-icon" />
                  <h3>Classification Result</h3>
                </div>

                <div className="main-prediction">
                  <div className="prediction-fruit">
                    <span className="fruit-emoji">🍎</span>
                    <span className="fruit-name">{result.prediction.class}</span>
                    {result.prediction.confidence_level && (
                      <span className="confidence-badge">{result.prediction.confidence_level}</span>
                    )}
                  </div>
                  <div className="confidence-meter">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${result.prediction.percentage}%` }}
                    />
                  </div>
                  <p className="confidence-text">
                    {result.prediction.percentage.toFixed(2)}% Confidence
                  </p>
                  {result.metadata && (
                    <p className="processing-time">
                      ⚡ Processed in {result.metadata.processing_time_ms}ms
                    </p>
                  )}
                </div>

                {result.top_predictions && (
                  <div className="chart-container">
                    <h4>Top 5 Predictions</h4>
                    <Bar 
                      data={getChartData()} 
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: { display: false }
                        },
                        scales: {
                          y: {
                            beginAtZero: true,
                            max: 100
                          }
                        }
                      }}
                    />
                  </div>
                )}

                <div className="predictions-list">
                  {result.top_predictions?.slice(0, 5).map((pred, index) => (
                    <div key={index} className="prediction-item">
                      <span className="rank">#{index + 1}</span>
                      <span className="class-name">{pred.class}</span>
                      <span className="percentage">{pred.percentage.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="placeholder">
                <div className="placeholder-icon">🔍</div>
                <p>Upload an image to see results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

export default Classifier
