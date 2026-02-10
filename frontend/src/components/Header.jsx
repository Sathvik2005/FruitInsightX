import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { FaSun, FaMoon, FaGithub } from 'react-icons/fa'
import './Header.css'

const Header = ({ theme, toggleTheme }) => {
  const headerRef = useRef(null)
  const logoRef = useRef(null)

  useEffect(() => {
    // Animate header on load
    gsap.from(headerRef.current, {
      y: -100,
      opacity: 0,
      duration: 0.8,
      ease: 'power3.out'
    })

    // Logo pulse animation
    gsap.to(logoRef.current, {
      scale: 1.1,
      duration: 1.5,
      repeat: -1,
      yoyo: true,
      ease: 'sine.inOut'
    })

    // Scroll effect
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const handleScroll = () => {
    const header = headerRef.current
    if (window.scrollY > 50) {
      header.classList.add('scrolled')
    } else {
      header.classList.remove('scrolled')
    }
  }

  return (
    <header className="header" ref={headerRef}>
      <div className="header-content">
        <div className="logo" ref={logoRef}>
          <span className="logo-icon">🍎</span>
          <span className="logo-text">Fruit Classifier</span>
        </div>
        
        <nav className="nav">
          <a href="#home" className="nav-link">Home</a>
          <a href="#classify" className="nav-link">Classify</a>
          <a href="#features" className="nav-link">Features</a>
          <a href="#about" className="nav-link">About</a>
        </nav>

        <div className="header-actions">
          <button 
            className="theme-toggle" 
            onClick={toggleTheme}
            aria-label="Toggle theme"
          >
            {theme === 'light' ? <FaMoon /> : <FaSun />}
          </button>
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noopener noreferrer"
            className="github-link"
            aria-label="GitHub"
          >
            <FaGithub />
          </a>
        </div>
      </div>
    </header>
  )
}

export default Header
