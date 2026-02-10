import { FaGithub, FaLinkedin, FaTwitter, FaHeart } from 'react-icons/fa'
import './Footer.css'

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-section">
          <h3 className="footer-title">Fruit Classifier</h3>
          <p className="footer-description">
            AI-powered fruit recognition system built with cutting-edge deep learning technology.
          </p>
          <div className="social-links">
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
              <FaGithub />
            </a>
            <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
              <FaLinkedin />
            </a>
            <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" aria-label="Twitter">
              <FaTwitter />
            </a>
          </div>
        </div>

        <div className="footer-section">
          <h4>Quick Links</h4>
          <ul className="footer-links">
            <li><a href="#home">Home</a></li>
            <li><a href="#classify">Classify</a></li>
            <li><a href="#features">Features</a></li>
            <li><a href="#about">About</a></li>
          </ul>
        </div>

        <div className="footer-section">
          <h4>Technology</h4>
          <ul className="footer-links">
            <li><a href="https://www.tensorflow.org/" target="_blank" rel="noopener noreferrer">TensorFlow</a></li>
            <li><a href="https://fastapi.tiangolo.com/" target="_blank" rel="noopener noreferrer">FastAPI</a></li>
            <li><a href="https://react.dev/" target="_blank" rel="noopener noreferrer">React</a></li>
            <li><a href="https://gsap.com/" target="_blank" rel="noopener noreferrer">GSAP</a></li>
          </ul>
        </div>

        <div className="footer-section">
          <h4>Resources</h4>
          <ul className="footer-links">
            <li><a href="#docs">Documentation</a></li>
            <li><a href="#api">API Reference</a></li>
            <li><a href="#contact">Contact</a></li>
            <li><a href="#privacy">Privacy Policy</a></li>
          </ul>
        </div>
      </div>

      <div className="footer-bottom">
        <p className="copyright">
          © 2026 Fruit Classifier. Made with <FaHeart className="heart-icon" /> using React & FastAPI
        </p>
      </div>
    </footer>
  )
}

export default Footer
