import { useState, useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Header from './components/Header'
import Hero from './components/Hero'
import Classifier from './components/Classifier'
import Features from './components/Features'
import Footer from './components/Footer'
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
    <div className={`app ${theme}`} ref={appRef}>
      <div className="app-background"></div>
      <Header theme={theme} toggleTheme={toggleTheme} />
      <Hero />
      <Classifier />
      <Features />
      <Footer />
    </div>
  )
}

export default App
