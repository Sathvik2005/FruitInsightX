import { Link } from 'react-router-dom'
import { FaGithub, FaLinkedin, FaTwitter, FaHeart, FaRocket } from 'react-icons/fa'
import './Footer.css'

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-section">
          <h3 className="footer-title">🍎 FruitInsightX</h3>
          <p className="footer-description">
            AI-powered fruit classification system using advanced deep learning and computer vision technology. Classify 11+ fruit varieties with high accuracy.
          </p>
          <div className="social-links">
            <a href="https://github.com/Sathvik2005/FruitInsightX" target="_blank" rel="noopener noreferrer" aria-label="GitHub" title="View on GitHub">
              <FaGithub />
            </a>
            <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" title="Connect on LinkedIn">
              <FaLinkedin />
            </a>
            <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" aria-label="Twitter" title="Follow on Twitter">
              <FaTwitter />
            </a>
          </div>
        </div>

        <div className="footer-section">
          <h4>Quick Links</h4>
          <ul className="footer-links">
            <li><Link to="/">Home</Link></li>
            <li><Link to="/about">About Us</Link></li>
            <li><Link to="/documentation">Documentation</Link></li>
            <li><Link to="/api">API Console</Link></li>
            <li><Link to="/contact">Contact</Link></li>
          </ul>
        </div>

        <div className="footer-section">
          <h4>Technology Stack</h4>
          <ul className="footer-links">
            <li><a href="https://www.tensorflow.org/" target="_blank" rel="noopener noreferrer">TensorFlow 2.16</a></li>
            <li><a href="https://fastapi.tiangolo.com/" target="_blank" rel="noopener noreferrer">FastAPI</a></li>
            <li><a href="https://react.dev/" target="_blank" rel="noopener noreferrer">React 18</a></li>
            <li><a href="https://gsap.com/" target="_blank" rel="noopener noreferrer">GSAP Animations</a></li>
            <li><a href="https://vitejs.dev/" target="_blank" rel="noopener noreferrer">Vite</a></li>
          </ul>
        </div>

        <div className="footer-section">
          <h4>Resources</h4>
          <ul className="footer-links">
            <li><Link to="/documentation">API Docs</Link></li>
            <li><Link to="/api">Live Testing</Link></li>
            <li><a href="https://github.com/Sathvik2005/FruitInsightX#readme" target="_blank" rel="noopener noreferrer">Usage Guide</a></li>
            <li><Link to="/about">Model Info</Link></li>
          </ul>
        </div>
      </div>

      <div className="footer-bottom">
        <p className="copyright">
          © 2026 FruitInsightX. Made with <FaHeart className="heart-icon" /> using React & FastAPI | <FaRocket className="rocket-icon" /> Powered by AI
        </p>
      </div>
    </footer>
  )
}

export default Footer
