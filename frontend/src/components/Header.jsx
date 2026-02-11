import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { gsap } from 'gsap'
import { FaSun, FaMoon, FaGithub, FaHome, FaInfoCircle, FaBook, FaCode, FaEnvelope, FaBars, FaTimes } from 'react-icons/fa'
import './Header.css'

const Header = ({ theme, toggleTheme }) => {
  const headerRef = useRef(null)
  const logoRef = useRef(null)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const location = useLocation()

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

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen)
  }

  const closeMobileMenu = () => {
    setMobileMenuOpen(false)
  }

  return (
    <header className="header" ref={headerRef}>
      <div className="header-content">
        <Link to="/" className="logo" ref={logoRef}>
          <span className="logo-icon">🍎</span>
          <span className="logo-text">FruitInsightX</span>
        </Link>
        
        <button className="mobile-menu-toggle" onClick={toggleMobileMenu} aria-label="Toggle menu">
          {mobileMenuOpen ? <FaTimes /> : <FaBars />}
        </button>

        <nav className={`nav ${mobileMenuOpen ? 'mobile-active' : ''}`}>
          <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`} onClick={closeMobileMenu}>
            <FaHome className="nav-icon" />
            <span>Home</span>
          </Link>
          <Link to="/about" className={`nav-link ${location.pathname === '/about' ? 'active' : ''}`} onClick={closeMobileMenu}>
            <FaInfoCircle className="nav-icon" />
            <span>About</span>
          </Link>
          <Link to="/documentation" className={`nav-link ${location.pathname === '/documentation' ? 'active' : ''}`} onClick={closeMobileMenu}>
            <FaBook className="nav-icon" />
            <span>Docs</span>
          </Link>
          <Link to="/api" className={`nav-link ${location.pathname === '/api' ? 'active' : ''}`} onClick={closeMobileMenu}>
            <FaCode className="nav-icon" />
            <span>API</span>
          </Link>
          <Link to="/contact" className={`nav-link ${location.pathname === '/contact' ? 'active' : ''}`} onClick={closeMobileMenu}>
            <FaEnvelope className="nav-icon" />
            <span>Contact</span>
          </Link>
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
            href="https://github.com/Sathvik2005/FruitInsightX" 
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
