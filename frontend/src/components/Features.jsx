import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { FaBrain, FaRocket, FaMobileAlt, FaChartLine } from 'react-icons/fa'
import './Features.css'

gsap.registerPlugin(ScrollTrigger)

const features = [
  {
    icon: <FaBrain />,
    title: 'Deep Learning',
    description: 'Powered by advanced CNN architecture trained on thousands of fruit images'
  },
  {
    icon: <FaRocket />,
    title: 'Lightning Fast',
    description: 'Get results in milliseconds with our optimized inference engine'
  },
  {
    icon: <FaMobileAlt />,
    title: 'Responsive Design',
    description: 'Works seamlessly on all devices - desktop, tablet, and mobile'
  },
  {
    icon: <FaChartLine />,
    title: 'High Accuracy',
    description: '95%+ accuracy across 11 different fruit categories'
  }
]

const Features = () => {
  const featuresRef = useRef([])
  const sectionRef = useRef(null)

  useEffect(() => {
    // Stagger animation for features
    gsap.from(featuresRef.current, {
      scrollTrigger: {
        trigger: sectionRef.current,
        start: 'top 70%'
      },
      y: 100,
      opacity: 0,
      duration: 0.8,
      stagger: 0.2,
      ease: 'power3.out'
    })

    // Hover animations for each feature card
    featuresRef.current.forEach(card => {
      if (!card) return
      
      card.addEventListener('mouseenter', () => {
        gsap.to(card, {
          y: -10,
          scale: 1.05,
          duration: 0.3,
          ease: 'power2.out'
        })
      })

      card.addEventListener('mouseleave', () => {
        gsap.to(card, {
          y: 0,
          scale: 1,
          duration: 0.3,
          ease: 'power2.out'
        })
      })
    })
  }, [])

  return (
    <section className="features" id="features" ref={sectionRef}>
      <div className="features-container">
        <h2 className="section-title">Why Choose Our Classifier?</h2>
        <p className="section-subtitle">
          State-of-the-art technology meets user-friendly design
        </p>

        <div className="features-grid">
          {features.map((feature, index) => (
            <div
              key={index}
              className="feature-card"
              ref={el => featuresRef.current[index] = el}
            >
              <div className="feature-icon">{feature.icon}</div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Stats Section */}
        <div className="stats-section">
          <div className="stat-item">
            <span className="stat-number">11</span>
            <span className="stat-label">Fruit Types</span>
          </div>
          <div className="stat-item">
            <span className="stat-number">95%</span>
            <span className="stat-label">Accuracy</span>
          </div>
          <div className="stat-item">
            <span className="stat-number">&lt;100ms</span>
            <span className="stat-label">Response Time</span>
          </div>
          <div className="stat-item">
            <span className="stat-number">10K+</span>
            <span className="stat-label">Training Images</span>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Features
