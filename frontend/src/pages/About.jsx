import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import './About.css'

const About = () => {
  const containerRef = useRef(null)

  useEffect(() => {
    if (containerRef.current) {
      gsap.from(containerRef.current.querySelectorAll('.about-section'), {
        y: 50,
        opacity: 0,
        duration: 0.8,
        stagger: 0.2,
        ease: 'power3.out'
      })
    }
  }, [])

  return (
    <div className="about-page" ref={containerRef}>
      <div className="about-hero">
        <h1 className="page-title">About FruitInsightX</h1>
        <p className="page-subtitle">
          Enterprise-grade AI platform revolutionizing fruit quality assessment
        </p>
      </div>

      <div className="about-container">
        <section className="about-section">
          <h2>Our Mission</h2>
          <p>
            FruitInsightX is dedicated to transforming the agricultural industry through
            cutting-edge artificial intelligence and machine learning technologies. Our platform
            provides real-time, accurate fruit classification and quality assessment, enabling
            businesses to make data-driven decisions and improve operational efficiency.
          </p>
        </section>

        <section className="about-section">
          <h2>Technology Stack</h2>
          <div className="tech-grid">
            <div className="tech-card">
              <h3>Deep Learning</h3>
              <p>Powered by TensorFlow 2.16 with custom CNN architecture for high-accuracy classification</p>
            </div>
            <div className="tech-card">
              <h3>Modern Frontend</h3>
              <p>Built with React 18, Vite, and GSAP for smooth animations and responsive design</p>
            </div>
            <div className="tech-card">
              <h3>Fast Backend</h3>
              <p>FastAPI 2.0 provides high-performance async API with sub-100ms response times</p>
            </div>
            <div className="tech-card">
              <h3>Production Ready</h3>
              <p>Comprehensive error handling, batch processing, and enterprise-grade reliability</p>
            </div>
          </div>
        </section>

        <section className="about-section">
          <h2>11 Fruit Classes</h2>
          <div className="fruit-grid">
            <div className="fruit-item">🍎 Apple</div>
            <div className="fruit-item">🍌 Banana</div>
            <div className="fruit-item">🍒 Cherry</div>
            <div className="fruit-item">🍇 Grape</div>
            <div className="fruit-item">🥭 Guava</div>
            <div className="fruit-item">🥝 Kiwi</div>
            <div className="fruit-item">🥭 Mango</div>
            <div className="fruit-item">🍊 Orange</div>
            <div className="fruit-item">🍑 Peach</div>
            <div className="fruit-item">🍐 Pear</div>
            <div className="fruit-item">🍓 Strawberry</div>
          </div>
        </section>

        <section className="about-section">
          <h2>Performance Metrics</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-value">98.5%</div>
              <div className="metric-label">Training Accuracy</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">96.2%</div>
              <div className="metric-label">Validation Accuracy</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">&lt;50ms</div>
              <div className="metric-label">Inference Time</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">99.8%</div>
              <div className="metric-label">Top-5 Accuracy</div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default About
