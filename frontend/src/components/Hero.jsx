import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { FaArrowDown } from 'react-icons/fa'
import './Hero.css'

const Hero = () => {
  const heroRef = useRef(null)
  const titleRef = useRef(null)
  const subtitleRef = useRef(null)
  const ctaRef = useRef(null)
  const fruitsRef = useRef([])

  useEffect(() => {
    const tl = gsap.timeline()

    // Title animation
    tl.from(titleRef.current, {
      y: 100,
      opacity: 0,
      duration: 1,
      ease: 'power4.out'
    })

    // Subtitle animation
    tl.from(subtitleRef.current, {
      y: 50,
      opacity: 0,
      duration: 0.8,
      ease: 'power3.out'
    }, '-=0.5')

    // CTA button animation
    tl.from(ctaRef.current, {
      scale: 0,
      opacity: 0,
      duration: 0.5,
      ease: 'back.out(1.7)'
    }, '-=0.3')

    // Floating fruits animation
    fruitsRef.current.forEach((fruit, index) => {
      gsap.to(fruit, {
        y: -30,
        rotation: 360,
        duration: 2 + index * 0.5,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
        delay: index * 0.2
      })
    })

    // Particle effect
    createParticles()
  }, [])

  const createParticles = () => {
    const particles = document.querySelectorAll('.particle')
    particles.forEach(particle => {
      gsap.to(particle, {
        y: -window.innerHeight,
        x: `random(-100, 100)`,
        opacity: 0,
        duration: `random(3, 6)`,
        repeat: -1,
        ease: 'none',
        delay: `random(0, 3)`
      })
    })
  }

  const scrollToClassifier = () => {
    document.querySelector('#classify').scrollIntoView({
      behavior: 'smooth'
    })
  }

  return (
    <section className="hero" id="home" ref={heroRef}>
      <div className="hero-content">
        <h1 className="hero-title" ref={titleRef}>
          Identify Fruits with
          <span className="gradient-text"> AI Power</span>
        </h1>
        
        <p className="hero-subtitle" ref={subtitleRef}>
          Upload an image and let our advanced deep learning model
          classify your fruits with incredible accuracy
        </p>

        <button 
          className="cta-button" 
          ref={ctaRef}
          onClick={scrollToClassifier}
        >
          Try It Now
          <FaArrowDown className="cta-icon" />
        </button>

        {/* Floating fruit emojis */}
        <div className="floating-fruits">
          <span 
            className="fruit-emoji" 
            ref={el => fruitsRef.current[0] = el}
            style={{ left: '10%', top: '20%' }}
          >
            🍎
          </span>
          <span 
            className="fruit-emoji" 
            ref={el => fruitsRef.current[1] = el}
            style={{ right: '15%', top: '30%' }}
          >
            🍌
          </span>
          <span 
            className="fruit-emoji" 
            ref={el => fruitsRef.current[2] = el}
            style={{ left: '20%', bottom: '25%' }}
          >
            🍇
          </span>
          <span 
            className="fruit-emoji" 
            ref={el => fruitsRef.current[3] = el}
            style={{ right: '10%', bottom: '20%' }}
          >
            🍍
          </span>
          <span 
            className="fruit-emoji" 
            ref={el => fruitsRef.current[4] = el}
            style={{ left: '50%', top: '15%' }}
          >
            🍒
          </span>
        </div>

        {/* Animated particles */}
        <div className="particles">
          {[...Array(20)].map((_, i) => (
            <div 
              key={i} 
              className="particle"
              style={{
                left: `${Math.random() * 100}%`,
                bottom: 0
              }}
            />
          ))}
        </div>
      </div>

      {/* Wave animation */}
      <div className="wave-container">
        <svg
          className="wave"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 1200 120"
          preserveAspectRatio="none"
        >
          <path
            d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z"
            fill="currentColor"
          />
        </svg>
      </div>
    </section>
  )
}

export default Hero
