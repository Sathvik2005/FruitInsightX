import { useState, useEffect, useRef } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import About from './pages/About'
import Documentation from './pages/Documentation'
import ApiPage from './pages/ApiPage'
import Contact from './pages/Contact'
import './App.css'

gsap.registerPlugin(ScrollTrigger)

function App() {
  const [theme, setTheme] = useState('light')
  const appRef = useRef(null)

  useEffect(() => {
    // Smooth scroll animation
    gsap.to(appRef.current, {
      duration: 0.3,
      ease: 'power2.out'
    })

    // Parallax effect for background
    gsap.to('.app-background', {
      scrollTrigger: {
        trigger: appRef.current,
        start: 'top top',
        end: 'bottom bottom',
        scrub: true
      },
      y: '50%',
      ease: 'none'
    })
  }, [])

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light')
  }

  return (
    <Router>
      <div className={`app ${theme}`} ref={appRef}>
        <div className="app-background"></div>
        <Header theme={theme} toggleTheme={toggleTheme} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/documentation" element={<Documentation />} />
            <Route path="/api" element={<ApiPage />} />
            <Route path="/contact" element={<Contact />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  )
}

export default App
