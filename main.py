from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import pandas as pd
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from fpdf import FPDF
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from huggingface_hub import login
import polars as pl
import google.generativeai as genai
from markdown import markdown
from itertools import islice
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Constants
GEMINI_MODEL = "gemini-2.0-flash"
# API_KEY = "AIzaSyCR9VnCOiSdxWbsQD-7E53nWJ4gzDBXMF8"
# HF_TOKEN = "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"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YouTube Sentiment Analyzer Pro</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --danger-color: #ef4444;
                --glass-bg: rgba(255, 255, 255, 0.1);
                --glass-border: rgba(255, 255, 255, 0.2);
                --text-primary: #1f2937;
                --text-secondary: #6b7280;
                --shadow-light: 0 4px 6px rgba(0, 0, 0, 0.1);
                --shadow-medium: 0 10px 25px rgba(0, 0, 0, 0.15);
                --shadow-heavy: 0 20px 40px rgba(0, 0, 0, 0.2);
            }

            body {
                font-family: 'Inter', sans-serif;
                background: var(--primary-gradient);
                min-height: 100vh;
                overflow-x: hidden;
                position: relative;
            }

            /* Animated Background Elements */
            .bg-animation {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
                overflow: hidden;
            }

            .floating-shapes {
                position: absolute;
                width: 100%;
                height: 100%;
            }

            .shape {
                position: absolute;
                opacity: 0.1;
                animation: float 20s infinite ease-in-out;
            }

            .shape:nth-child(1) {
                width: 80px;
                height: 80px;
                background: white;
                border-radius: 50%;
                top: 10%;
                left: 10%;
                animation-delay: 0s;
            }

            .shape:nth-child(2) {
                width: 60px;
                height: 60px;
                background: white;
                border-radius: 50%;
                top: 20%;
                right: 20%;
                animation-delay: 5s;
            }

            .shape:nth-child(3) {
                width: 100px;
                height: 100px;
                background: white;
                border-radius: 20px;
                bottom: 20%;
                left: 15%;
                animation-delay: 10s;
            }

            .shape:nth-child(4) {
                width: 120px;
                height: 120px;
                background: white;
                border-radius: 50%;
                bottom: 10%;
                right: 10%;
                animation-delay: 15s;
            }

            @keyframes float {
                0%, 100% { 
                    transform: translateY(0px) rotate(0deg) scale(1); 
                }
                25% { 
                    transform: translateY(-20px) rotate(5deg) scale(1.1); 
                }
                50% { 
                    transform: translateY(-40px) rotate(-5deg) scale(0.9); 
                }
                75% { 
                    transform: translateY(-20px) rotate(3deg) scale(1.05); 
                }
            }

            /* Particle System */
            .particles {
                position: absolute;
                width: 100%;
                height: 100%;
            }

            .particle {
                position: absolute;
                width: 3px;
                height: 3px;
                background: rgba(255, 255, 255, 0.6);
                border-radius: 50%;
                animation: particleFloat 8s infinite linear;
            }

            @keyframes particleFloat {
                0% {
                    transform: translateY(100vh) translateX(0px) rotate(0deg);
                    opacity: 0;
                }
                10% {
                    opacity: 1;
                }
                90% {
                    opacity: 1;
                }
                100% {
                    transform: translateY(-100px) translateX(100px) rotate(360deg);
                    opacity: 0;
                }
            }

            /* Main Container */
            .main-container {
                position: relative;
                z-index: 1;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 2rem;
            }

            /* Header Section */
            .header {
                text-align: center;
                margin-bottom: 4rem;
                animation: slideInDown 1s ease-out;
            }

            @keyframes slideInDown {
                from {
                    transform: translateY(-100px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }

            .header h1 {
                font-size: clamp(2.5rem, 8vw, 4rem);
                font-weight: 800;
                background: linear-gradient(45deg, #ffffff, #f0f0f0, #ffffff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1rem;
                text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                position: relative;
            }

            .header h1::after {
                content: '';
                position: absolute;
                bottom: -10px;
                left: 50%;
                transform: translateX(-50%);
                width: 100px;
                height: 4px;
                background: var(--secondary-gradient);
                border-radius: 2px;
                animation: expandWidth 1s ease-out 0.5s both;
            }

            @keyframes expandWidth {
                from { width: 0; }
                to { width: 100px; }
            }

            .header p {
                font-size: 1.25rem;
                color: rgba(255, 255, 255, 0.9);
                font-weight: 300;
                max-width: 600px;
                margin: 0 auto;
                line-height: 1.6;
            }

            .header .subtitle {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 1rem;
                margin-top: 1rem;
                animation: fadeInUp 1s ease-out 0.3s both;
            }

            @keyframes fadeInUp {
                from {
                    transform: translateY(30px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }

            /* Form Card */
            .form-container {
                width: 100%;
                max-width: 800px;
                animation: slideInUp 1s ease-out 0.6s both;
            }

            @keyframes slideInUp {
                from {
                    transform: translateY(100px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }

            .form-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border: 1px solid var(--glass-border);
                border-radius: 24px;
                padding: 3rem;
                box-shadow: var(--shadow-heavy);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .form-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 2px;
                background: var(--secondary-gradient);
                transition: left 0.5s ease;
            }

            .form-card:hover::before {
                left: 0;
            }

            .form-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3);
            }

            /* Form Groups */
            .form-group {
                margin-bottom: 2.5rem;
                animation: fadeIn 0.6s ease-out;
                animation-fill-mode: both;
            }

            .form-group:nth-child(1) { animation-delay: 0.8s; }
            .form-group:nth-child(2) { animation-delay: 1s; }
            .form-group:nth-child(3) { animation-delay: 1.2s; }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .form-label {
                display: flex;
                align-items: center;
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 1rem;
                font-size: 1.1rem;
                transition: color 0.3s ease;
            }

            .form-label i {
                margin-right: 0.75rem;
                font-size: 1.2rem;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .form-control {
                width: 100%;
                padding: 1.25rem 1.5rem;
                border: 2px solid #e5e7eb;
                border-radius: 16px;
                font-size: 1rem;
                font-family: 'Inter', sans-serif;
                background: #f9fafb;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
            }

            .form-control:focus {
                outline: none;
                border-color: #667eea;
                background: white;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }

            .form-control:hover {
                border-color: #9ca3af;
                background: white;
                transform: translateY(-1px);
            }

            textarea.form-control {
                resize: vertical;
                min-height: 140px;
                font-family: 'Inter', sans-serif;
            }

            /* Submit Button */
            .btn-submit {
                width: 100%;
                padding: 1.5rem 2rem;
                background: var(--primary-gradient);
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 1.2rem;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-top: 1rem;
                animation: fadeIn 0.6s ease-out 1.4s both;
            }

            .btn-submit::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                transition: left 0.6s ease;
            }

            .btn-submit:hover::before {
                left: 100%;
            }

            .btn-submit:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
                background: var(--secondary-gradient);
            }

            .btn-submit:active {
                transform: translateY(-2px);
            }

            .btn-submit:disabled {
                opacity: 0.7;
                cursor: not-allowed;
                transform: none;
            }

            /* Loading State */
            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                display: none;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                backdrop-filter: blur(5px);
            }

            .loading-overlay.show {
                display: flex;
                animation: fadeIn 0.3s ease-out;
            }

            .loading-content {
                background: white;
                padding: 3rem;
                border-radius: 20px;
                text-align: center;
                box-shadow: var(--shadow-heavy);
                animation: scaleIn 0.3s ease-out;
            }

            @keyframes scaleIn {
                from { transform: scale(0.8); opacity: 0; }
                to { transform: scale(1); opacity: 1; }
            }

            .spinner {
                width: 60px;
                height: 60px;
                border: 4px solid #f3f4f6;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 1.5rem;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .loading-text {
                color: var(--text-primary);
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }

            .loading-subtext {
                color: var(--text-secondary);
                font-size: 0.9rem;
            }

            /* Features Grid */
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 2rem;
                margin-top: 4rem;
                max-width: 1000px;
                width: 100%;
                animation: fadeIn 1s ease-out 1.6s both;
            }

            .feature-card {
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border: 1px solid var(--glass-border);
                border-radius: 20px;
                padding: 2rem;
                text-align: center;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }

            .feature-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
                transform: translateX(-100%);
                transition: transform 0.6s ease;
            }

            .feature-card:hover::before {
                transform: translateX(100%);
            }

            .feature-card:hover {
                transform: translateY(-10px) scale(1.02);
                background: rgba(255, 255, 255, 0.15);
                box-shadow: var(--shadow-heavy);
            }

            .feature-icon {
                font-size: 3rem;
                margin-bottom: 1.5rem;
                background: var(--secondary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: bounce 2s infinite;
            }

            @keyframes bounce {
                0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                40% { transform: translateY(-10px); }
                60% { transform: translateY(-5px); }
            }

            .feature-title {
                color: white;
                font-weight: 700;
                font-size: 1.3rem;
                margin-bottom: 1rem;
            }

            .feature-description {
                color: rgba(255, 255, 255, 0.8);
                line-height: 1.6;
                font-size: 0.95rem;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .main-container {
                    padding: 1rem;
                }

                .form-card {
                    padding: 2rem;
                    margin: 1rem;
                }

                .features-grid {
                    grid-template-columns: 1fr;
                    margin-top: 3rem;
                }

                .header h1 {
                    font-size: 2.5rem;
                }
            }

            @media (max-width: 480px) {
                .form-card {
                    padding: 1.5rem;
                    border-radius: 16px;
                }

                .btn-submit {
                    padding: 1.25rem 1.5rem;
                    font-size: 1.1rem;
                }
            }

            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }

            ::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 255, 255, 0.5);
            }

            /* Progress Bar */
            .progress-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: rgba(255, 255, 255, 0.1);
                z-index: 1001;
            }

            .progress-bar {
                height: 100%;
                background: var(--secondary-gradient);
                width: 0%;
                transition: width 0.3s ease;
            }
        </style>
    </head>
    <body>
        <!-- Progress Bar -->
        <div class="progress-container">
            <div class="progress-bar" id="progressBar"></div>
        </div>

        <!-- Background Animation -->
        <div class="bg-animation">
            <div class="floating-shapes">
                <div class="shape"></div>
                <div class="shape"></div>
                <div class="shape"></div>
                <div class="shape"></div>
            </div>
            <div class="particles" id="particlesContainer"></div>
        </div>

        <!-- Loading Overlay -->
        <div class="loading-overlay" id="loadingOverlay">
            <div class="loading-content">
                <div class="spinner"></div>
                <div class="loading-text">Analyzing Comments...</div>
                <div class="loading-subtext">This may take a few moments while we process your data</div>
            </div>
        </div>

        <!-- Main Container -->
        <div class="main-container">
            <!-- Header -->
            <header class="header">
                <h1>
                    <i class="fab fa-youtube"></i>
                    YouTube Sentiment Analyzer Pro
                </h1>
                <p>Transform YouTube comments into actionable business insights with AI-powered sentiment analysis</p>
                <div class="subtitle">
                    <span><i class="fas fa-robot"></i> AI-Powered</span>
                    <span><i class="fas fa-chart-line"></i> Real-time Analysis</span>
                    <span><i class="fas fa-download"></i> PDF Reports</span>
                </div>
            </header>

            <!-- Form Container -->
            <div class="form-container">
                <div class="form-card">
                    <form id="sentimentForm" action="/analyze" method="post">
                        <div class="form-group">
                            <label class="form-label" for="youtube_url">
                                <i class="fas fa-link"></i>
                                YouTube Video URL
                            </label>
                            <input 
                                type="url" 
                                class="form-control" 
                                id="youtube_url" 
                                name="youtube_url" 
                                placeholder="https://www.youtube.com/watch?v=example" 
                                required
                            >
                        </div>

                        <div class="form-group">
                            <label class="form-label" for="custom_stopwords">
                                <i class="fas fa-filter"></i>
                                Custom Stopwords (Optional)
                            </label>
                            <input 
                                type="text" 
                                class="form-control" 
                                id="custom_stopwords" 
                                name="custom_stopwords" 
                                placeholder="Enter words to exclude: spam, bot, fake..."
                            >
                        </div>

                        <div class="form-group">
                            <label class="form-label" for="custom_question">
                                <i class="fas fa-question-circle"></i>
                                Analysis Focus Question
                            </label>
                            <textarea 
                                class="form-control" 
                                id="custom_question" 
                                name="custom_question" 
                                rows="4"
                                placeholder="What specific insights would you like to discover from the sentiment analysis?"
                            >Please provide insights based on the sentiment analysis:</textarea>
                        </div>

                        <button type="submit" class="btn-submit">
                            <i class="fas fa-chart-pie"></i>
                            Analyze Sentiment
                        </button>
                    </form>
                </div>
            </div>

            <!-- Features Grid -->
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="feature-title">AI-Powered Analysis</div>
                    <div class="feature-description">
                        Advanced machine learning models analyze sentiment with high accuracy and context understanding
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-cloud-word"></i>
                    </div>
                    <div class="feature-title">Visual Word Clouds</div>
                    <div class="feature-description">
                        Interactive word clouds reveal key themes and trending topics in positive, negative, and neutral comments
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-chart-bar"></i>
                    </div>
                    <div class="feature-title">Comprehensive Reports</div>
                    <div class="feature-description">
                        Detailed analytics with charts, insights, and actionable recommendations for business decisions
                    </div>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-mobile-alt"></i>
                    </div>
                    <div class="feature-title">Responsive Design</div>
                    <div class="feature-description">
                        Seamless experience across all devices - desktop, tablet, and mobile with touch-optimized interface
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Initialize page animations and interactions
            class SentimentAnalyzer {
                constructor() {
                    this.init();
                }

                init() {
                    this.createParticles();
                    this.setupFormHandlers();
                    this.setupScrollProgress();
                    this.setupInputAnimations();
                    this.setupFeatureAnimations();
                }

                // Create floating particles
                createParticles() {
                    const container = document.getElementById('particlesContainer');
                    const particleCount = 30;

                    for (let i = 0; i < particleCount; i++) {
                        const particle = document.createElement('div');
                        particle.className = 'particle';
                        
                        // Random positioning and timing
                        particle.style.left = Math.random() * 100 + '%';
                        particle.style.animationDelay = Math.random() * 8 + 's';
                        particle.style.animationDuration = (8 + Math.random() * 4) + 's';
                        
                        container.appendChild(particle);
                    }
                }

                // Setup form submission handling
                setupFormHandlers() {
                    const form = document.getElementById('sentimentForm');
                    const loadingOverlay = document.getElementById('loadingOverlay');

                    form.addEventListener('submit', (e) => {
                        // Show loading overlay
                        loadingOverlay.classList.add('show');
                        
                        // Disable form
                        const submitBtn = form.querySelector('.btn-submit');
                        submitBtn.disabled = true;
                        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                    });
                }

                // Setup scroll progress bar
                setupScrollProgress() {
                    const progressBar = document.getElementById('progressBar');
                    
                    window.addEventListener('scroll', () => {
                        const scrollTop = window.pageYOffset;
                        const docHeight = document.body.offsetHeight - window.innerHeight;
                        const scrollPercent = (scrollTop / docHeight) * 100;
                        
                        progressBar.style.width = scrollPercent + '%';
                    });
                }

                // Setup input animations
                setupInputAnimations() {
                    const inputs = document.querySelectorAll('.form-control');
                    
                    inputs.forEach(input => {
                        input.addEventListener('focus', () => {
                            input.parentElement.style.transform = 'scale(1.02)';
                            input.parentElement.style.transition = 'transform 0.3s ease';
                        });

                        input.addEventListener('blur', () => {
                            input.parentElement.style.transform = 'scale(1)';
                        });

                        // Real-time validation feedback
                        input.addEventListener('input', () => {
                            if (input.checkValidity()) {
                                input.style.borderColor = '#10b981';
                            } else {
                                input.style.borderColor = '#ef4444';
                            }
                        });
                    });
                }

                // Setup feature card animations
                setupFeatureAnimations() {
                    const features = document.querySelectorAll('.feature-card');
                    
                    const observer = new IntersectionObserver((entries) => {
                        entries.forEach((entry, index) => {
                            if (entry.isIntersecting) {
                                setTimeout(() => {
                                    entry.target.style.opacity = '1';
                                    entry.target.style.transform = 'translateY(0)';
                                }, index * 100);
                            }
                        });
                    }, { threshold: 0.1 });

                    features.forEach(feature => {
                        feature.style.opacity = '0';
                        feature.style.transform = 'translateY(50px)';
                        feature.style.transition = 'all 0.6s ease';
                        observer.observe(feature);
                    });
                }
            }

            // Add ripple effect to button
            function addRippleEffect() {
                const button = document.querySelector('.btn-submit');
                
                button.addEventListener('click', function(e) {
                    const ripple = document.createElement('div');
                    const rect = this.getBoundingClientRect();
                    const size = Math.max(rect.width, rect.height);
                    const x = e.clientX - rect.left - size / 2;
                    const y = e.clientY - rect.top - size / 2;
                    
                    ripple.style.cssText = `
                        position: absolute;
                        width: ${size}px;
                        height: ${size}px;
                        left: ${x}px;
                        top: ${y}px;
                        background: rgba(255,255,255,0.4);
                        border-radius: 50%;
                        transform: scale(0);
                        animation: ripple 0.8s ease-out;
                        pointer-events: none;
                    `;
                    
                    this.appendChild(ripple);
                    
                    setTimeout(() => ripple.remove(), 800);
                });
            }

            // Add CSS for ripple animation
            const rippleStyle = document.createElement('style');
            rippleStyle.textContent = `
                @keyframes ripple {
                    to {
                        transform: scale(2.5);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(rippleStyle);

            // Initialize everything when DOM is loaded
            document.addEventListener('DOMContentLoaded', () => {
                new SentimentAnalyzer();
                addRippleEffect();
            });
        </script>
    </body>
    </html>
    """

def safe_encode(text):
    """Safely encode text for PDF generation"""
    try:
        # Replace problematic characters
        text = str(text)
        # Remove or replace non-Latin characters
        text = text.encode('ascii', errors='ignore').decode('ascii')
        return text
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        return "Error encoding text"

def clean_text_data(df: pd.DataFrame, target_variable: str, all_stopwords: list) -> pd.DataFrame:
    """Clean text data using Polars for better performance"""
    try:
        logger.info("Starting text cleaning process with Polars")
        
        # Convert Pandas DataFrame to Polars DataFrame
        pl_df = pl.from_pandas(df)
        
        # Define regex patterns
        hyperlink_pattern = r"https?://\S+|www\.\S+"
        emoticon_pattern = r"[:;=X8B][-oO^']?[\)\(DPp\[\]{}@/\|\\<>*~]"
        emoji_pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]"
        number_pattern = r"\b\d+\b"
        special_char_pattern = r"[^a-zA-Z\s]"

        # Ensure target_variable is of type string
        pl_df = pl_df.with_columns(
            pl.col(target_variable).cast(pl.Utf8).alias(target_variable)
        )

        # Apply text cleaning steps
        pl_df = pl_df.with_columns(
            pl.col(target_variable)
            .str.replace(hyperlink_pattern, "", literal=False)
            .str.replace(emoticon_pattern, "", literal=False)
            .str.replace(emoji_pattern, "", literal=False)
            .str.replace(number_pattern, "", literal=False)
            .str.replace(special_char_pattern, "", literal=False)
            .str.replace(r"\s+", " ", literal=False)
            .str.strip_chars()
            .alias("cleaned_text")
        )

        # Remove stopwords
        stopwords_set = pl.Series("stopwords", all_stopwords)

        pl_df = pl_df.with_columns(
            pl.col("cleaned_text")
            .str.split(" ")
            .list.eval(
                pl.when(pl.element().str.to_lowercase().is_in(stopwords_set))
                .then(None)
                .otherwise(pl.element())
            )
            .list.drop_nulls()
            .list.join(" ")
            .alias("cleaned_text")
        )

        # Drop rows where cleaned_text length is greater than 512
        pl_df = pl_df.filter(pl.col("cleaned_text").str.len_chars() <= 512)

        logger.info(f"Text cleaning complete. Final dataframe shape: {pl_df.shape}")
        return pl_df.to_pandas()
    
    except Exception as e:
        logger.error(f"Error in text cleaning: {e}")
        # Return original dataframe with a simple cleaned_text column
        df['cleaned_text'] = df[target_variable].astype(str)
        return df

def analyze_sentiment(text, sentiment_analysis, label_index):
    """Analyze sentiment with error handling"""
    try:
        if not text or text.strip() == "":
            return pd.Series({"sentiment_label": "neutral", "sentiment_score": 0.5})
        
        result = sentiment_analysis(text)
        label = label_index.get(result[0]["label"], "neutral")
        score = result[0]["score"]
        return pd.Series({"sentiment_label": label, "sentiment_score": score})
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return pd.Series({"sentiment_label": "neutral", "sentiment_score": 0.5})

def create_modern_wordcloud(text, filename, colormap):
    """Create wordcloud with error handling"""
    try:
        if not text or text.strip() == "":
            return None
            
        wordcloud = WordCloud(
            min_font_size=8,
            max_words=150,
            width=1200,
            height=600,
            colormap=colormap,
            background_color="white",
            margin=20,
            relative_scaling=0.5,
            font_path=None
        ).generate(text)

        wordcloud_path = f"static/{filename}"
        wordcloud.to_file(wordcloud_path)
        return wordcloud_path
    except Exception as e:
        logger.error(f"Error creating wordcloud {filename}: {e}")
        return None

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_youtube(
    youtube_url: str = Form(...),
    custom_stopwords: str = Form(""),
    custom_question: str = Form("Please provide insights based on the sentiment analysis:"),
):
    try:
        logger.info(f"Starting analysis for URL: {youtube_url}")
        
        # Validate inputs
        if not youtube_url:
            raise HTTPException(status_code=400, detail="YouTube URL is required")
        
        # Login to Hugging Face
        try:
            login(HF_TOKEN)
            logger.info("Successfully logged into Hugging Face")
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {e}")
            raise HTTPException(status_code=500, detail="Authentication failed")

        # Initialize the downloader
        downloader = YoutubeCommentDownloader()

        # Fetch comments from a YouTube URL
        try:
            comments = downloader.get_comments_from_url(
                youtube_url,
                sort_by=SORT_BY_POPULAR
            )
            # Collect all "text" fields
            comment_texts = [comment['text'] for comment in islice(comments, 1000)]
            logger.info(f"Fetched {len(comment_texts)} comments")
        except Exception as e:
            logger.error(f"Error fetching comments: {e}")
            raise HTTPException(status_code=400, detail="Failed to fetch comments. Please check the YouTube URL.")

        # Create a DataFrame
        df = pd.DataFrame(comment_texts, columns=["comment"])

        if df.empty:
            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>No Comments Found</title>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
                <style>
                    body {
                        font-family: 'Inter', sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin: 0;
                    }
                    .error-container {
                        background: white;
                        padding: 3rem;
                        border-radius: 24px;
                        text-align: center;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
                        max-width: 500px;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 4rem;
                        color: #f59e0b;
                        margin-bottom: 1.5rem;
                        animation: bounce 1s infinite;
                    }
                    @keyframes bounce {
                        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                        40% { transform: translateY(-10px); }
                        60% { transform: translateY(-5px); }
                    }
                    .error-title {
                        color: #dc2626;
                        font-size: 1.8rem;
                        font-weight: 700;
                        margin-bottom: 1rem;
                    }
                    .error-message {
                        color: #6b7280;
                        margin-bottom: 2rem;
                        line-height: 1.6;
                    }
                    .btn-back {
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        color: white;
                        padding: 1rem 2rem;
                        text-decoration: none;
                        border-radius: 12px;
                        display: inline-block;
                        transition: all 0.3s ease;
                        font-weight: 600;
                    }
                    .btn-back:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
                        text-decoration: none;
                        color: white;
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <h2 class="error-title">No Comments Found</h2>
                    <p class="error-message">
                        We couldn't find any comments for this video. This might happen if:
                        <br>• Comments are disabled on the video
                        <br>• The video is private or doesn't exist
                        <br>• There are no comments yet
                    </p>
                    <a href="/" class="btn-back">
                        <i class="fas fa-arrow-left"></i> Try Another Video
                    </a>
                </div>
            </body>
            </html>
            """

        # Process stopwords
        add_stopwords = [
            "the", "of", "is", "a", "in", "https", "yg", "gua", "gue", "lo", "lu", "gw",
        ]
        custom_stopword_list = [word.strip() for word in custom_stopwords.split(",") if word.strip()]
        all_stopwords = add_stopwords + custom_stopword_list

        # Clean text data
        df = clean_text_data(df, "comment", all_stopwords)

        # Perform Sentiment Analysis
        try:
            pretrained = "mdhugol/indonesia-bert-sentiment-classification"
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained, token=HF_TOKEN
            )
            tokenizer = AutoTokenizer.from_pretrained(pretrained, token=HF_TOKEN)
            sentiment_analysis = pipeline(
                "sentiment-analysis", model=model, tokenizer=tokenizer
            )
            label_index = {
                "LABEL_0": "positive",
                "LABEL_1": "neutral",
                "LABEL_2": "negative",
            }
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise HTTPException(status_code=500, detail="Failed to load sentiment analysis model")

        # Apply sentiment analysis
        try:
            df[["sentiment_label", "sentiment_score"]] = df["cleaned_text"].apply(
                lambda text: analyze_sentiment(text, sentiment_analysis, label_index)
            )
            logger.info("Sentiment analysis completed")
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            # Fallback to random sentiment assignment
            df["sentiment_label"] = "neutral"
            df["sentiment_score"] = 0.5

        # Count the occurrences of each sentiment label
        sentiment_counts = df["sentiment_label"].value_counts()

        # Create modern sentiment distribution plot
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['#10B981', '#F59E0B', '#EF4444']  # Green, Yellow, Red
            bars = ax.bar(sentiment_counts.index, sentiment_counts.values, 
                          color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=14)
            
            ax.set_title('Sentiment Distribution Analysis', fontsize=20, fontweight='bold', pad=20)
            ax.set_xlabel('Sentiment Category', fontsize=14, fontweight='600')
            ax.set_ylabel('Number of Comments', fontsize=14, fontweight='600')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Style the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dee2e6')
            ax.spines['bottom'].set_color('#dee2e6')
            
            plt.tight_layout()
            sentiment_plot_path = "static/sentiment_distribution.png"
            plt.savefig(sentiment_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info("Sentiment distribution plot created")
        except Exception as e:
            logger.error(f"Error creating sentiment plot: {e}")
            sentiment_plot_path = None

        # Concatenate Cleaned text
        positive_text = " ".join(df[df["sentiment_label"] == "positive"]["cleaned_text"])
        negative_text = " ".join(df[df["sentiment_label"] == "negative"]["cleaned_text"])
        neutral_text = " ".join(df[df["sentiment_label"] == "neutral"]["cleaned_text"])

        # Generate wordclouds with different color schemes
        wordcloud_positive = create_modern_wordcloud(positive_text, "wordcloud_positive.png", "Greens")
        wordcloud_negative = create_modern_wordcloud(negative_text, "wordcloud_negative.png", "Reds")
        wordcloud_neutral = create_modern_wordcloud(neutral_text, "wordcloud_neutral.png", "Blues")

        # Generate Gemini responses for each sentiment
        gemini_responses = {}
        try:
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)

            sentiment_types = [
                ("positive", wordcloud_positive, "positive"),
                ("negative", wordcloud_negative, "negative"),
                ("neutral", wordcloud_neutral, "neutral")
            ]

            for sentiment_type, wordcloud_path, label in sentiment_types:
                if wordcloud_path:
                    try:
                        img = Image.open(wordcloud_path)
                        response = model.generate_content([
                            f"{custom_question} As a marketing consultant, I aim to analyze consumer insights derived from the {label} sentiment wordcloud. Please provide actionable insights and recommendations based on this {label} sentiment analysis in a structured format with bullet points.",
                            img,
                        ])
                        response.resolve()
                        gemini_responses[sentiment_type] = response.text
                        logger.info(f"Generated Gemini response for {sentiment_type}")
                    except Exception as e:
                        logger.error(f"Error generating content with Gemini for {sentiment_type}: {e}")
                        gemini_responses[sentiment_type] = f"Analysis for {sentiment_type} sentiment: Unable to generate detailed insights due to processing limitations."
            
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            gemini_responses = {
                "positive": "Positive sentiment analysis: Processing error occurred.",
                "negative": "Negative sentiment analysis: Processing error occurred.",
                "neutral": "Neutral sentiment analysis: Processing error occurred."
            }

        # Generate comprehensive summary
        response_result = None
        try:
            if gemini_responses:
                summary_prompt = f"{custom_question} Based on the overall sentiment analysis of YouTube comments, please provide a comprehensive business strategy summary with key insights and actionable recommendations."
                response = model.generate_content(summary_prompt)
                response.resolve()
                response_result = response.text
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            response_result = "Executive summary: Analysis completed with limited processing capabilities."

        # Generate enhanced PDF report
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(200, 15, txt="YouTube Sentiment Analysis Report", ln=True, align="C")
            pdf.ln(5)

            # Add sentiment distribution
            if sentiment_plot_path and os.path.exists(sentiment_plot_path):
                pdf.image(sentiment_plot_path, x=10, y=40, w=190)
                pdf.ln(120)

            # Add wordclouds and analysis
            for sentiment_type, wordcloud_path, label in [
                ("positive", wordcloud_positive, "positive"),
                ("negative", wordcloud_negative, "negative"), 
                ("neutral", wordcloud_neutral, "neutral")
            ]:
                if wordcloud_path and os.path.exists(wordcloud_path) and sentiment_type in gemini_responses:
                    pdf.add_page()
                    pdf.set_font("Arial", size=14)
                    pdf.cell(200, 10, txt=f"{label.title()} Sentiment Analysis", ln=True, align="C")
                    pdf.image(wordcloud_path, x=10, y=30, w=190, h=95)
                    pdf.ln(105)
                    pdf.set_font("Arial", size=10)
                    pdf.multi_cell(0, 6, safe_encode(gemini_responses[sentiment_type]))

            if response_result:
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt="Executive Summary & Recommendations", ln=True, align="C")
                pdf.ln(10)
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 6, safe_encode(response_result))

            pdf_file_path = "static/sentiment_analysis_report.pdf"
            pdf.output(pdf_file_path)
            logger.info("PDF report generated successfully")
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")

        # Create modern results HTML
        results_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sentiment Analysis Results</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #1f2937;
                }}

                .results-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                    animation: fadeIn 1s ease-out;
                }}

                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(30px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}

                .results-header {{
                    text-align: center;
                    margin-bottom: 3rem;
                    color: white;
                }}

                .results-header h1 {{
                    font-size: 3rem;
                    font-weight: 800;
                    margin-bottom: 1rem;
                    background: linear-gradient(45deg, #ffffff, #f0f0f0);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}

                .results-header p {{
                    font-size: 1.2rem;
                    opacity: 0.9;
                }}

                .result-card {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 20px;
                    margin-bottom: 2rem;
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    overflow: hidden;
                    transition: all 0.3s ease;
                    animation: slideUp 0.8s ease-out;
                    animation-fill-mode: both;
                }}

                .result-card:nth-child(even) {{ animation-delay: 0.2s; }}
                .result-card:nth-child(odd) {{ animation-delay: 0.1s; }}

                @keyframes slideUp {{
                    from {{ transform: translateY(50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}

                .result-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
                }}

                .card-header {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 1.5rem 2rem;
                    font-size: 1.3rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }}

                .card-body {{
                    padding: 2rem;
                }}

                .card-body img {{
                    width: 100%;
                    height: auto;
                    border-radius: 12px;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                    margin-bottom: 1.5rem;
                    transition: transform 0.3s ease;
                }}

                .card-body img:hover {{
                    transform: scale(1.02);
                }}

                .markdown-content {{
                    font-size: 1rem;
                    line-height: 1.7;
                    color: #374151;
                }}

                .markdown-content h1, .markdown-content h2, .markdown-content h3 {{
                    color: #1f2937;
                    margin-bottom: 1rem;
                    font-weight: 600;
                }}

                .markdown-content ul {{
                    margin: 1rem 0;
                    padding-left: 1.5rem;
                }}

                .markdown-content li {{
                    margin-bottom: 0.5rem;
                }}

                .download-section {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    text-align: center;
                    padding: 3rem 2rem;
                    border-radius: 20px;
                    margin: 2rem 0;
                    animation: pulse 2s infinite;
                }}

                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.02); }}
                }}

                .download-btn {{
                    display: inline-block;
                    background: white;
                    color: #f5576c;
                    padding: 1rem 2rem;
                    border-radius: 12px;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 1.1rem;
                    transition: all 0.3s ease;
                    margin: 1rem 0.5rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                }}

                .download-btn:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
                    text-decoration: none;
                    color: #f5576c;
                }}

                .back-btn {{
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 12px;
                    text-decoration: none;
                    font-weight: 600;
                    display: inline-block;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    transition: all 0.3s ease;
                    margin-top: 2rem;
                }}

                .back-btn:hover {{
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                    text-decoration: none;
                    color: white;
                }}

                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                    margin-bottom: 2rem;
                }}

                .stat-item {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    text-align: center;
                    animation: fadeInUp 0.8s ease-out;
                }}

                @keyframes fadeInUp {{
                    from {{ transform: translateY(30px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}

                .stat-number {{
                    font-size: 2rem;
                    font-weight: 800;
                    display: block;
                }}

                .stat-label {{
                    font-size: 0.9rem;
                    opacity: 0.9;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}

                @media (max-width: 768px) {{
                    .results-container {{
                        padding: 1rem;
                    }}

                    .results-header h1 {{
                        font-size: 2rem;
                    }}

                    .card-body {{
                        padding: 1.5rem;
                    }}

                    .download-section {{
                        padding: 2rem 1rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="results-container">
                <div class="results-header">
                    <h1><i class="fas fa-chart-line"></i> Analysis Complete!</h1>
                    <p>Here are your comprehensive sentiment analysis results</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-number">{len(df)}</span>
                        <span class="stat-label">Total Comments</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">{sentiment_counts.get('positive', 0)}</span>
                        <span class="stat-label">Positive</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">{sentiment_counts.get('neutral', 0)}</span>
                        <span class="stat-label">Neutral</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">{sentiment_counts.get('negative', 0)}</span>
                        <span class="stat-label">Negative</span>
                    </div>
                </div>

                <div class="result-card">
                    <div class="card-header">
                        <i class="fas fa-chart-bar"></i>
                        Sentiment Distribution Overview
                    </div>
                    <div class="card-body">
                        <img src="/static/sentiment_distribution.png" alt="Sentiment Distribution Chart">
                        <p>This chart shows the overall distribution of sentiments across all analyzed comments.</p>
                    </div>
                </div>
        """

        # Add sentiment-specific cards
        if "positive" in gemini_responses and wordcloud_positive:
            results_html += f"""
                <div class="result-card">
                    <div class="card-header">
                        <i class="fas fa-smile" style="color: #10b981;"></i>
                        Positive Sentiment Analysis
                    </div>
                    <div class="card-body">
                        <img src="/static/wordcloud_positive.png" alt="Positive Sentiment Word Cloud">
                        <div class="markdown-content">
                            {markdown(gemini_responses["positive"])}
                        </div>
                    </div>
                </div>
            """

        if "negative" in gemini_responses and wordcloud_negative:
            results_html += f"""
                <div class="result-card">
                    <div class="card-header">
                        <i class="fas fa-frown" style="color: #ef4444;"></i>
                        Negative Sentiment Analysis
                    </div>
                    <div class="card-body">
                        <img src="/static/wordcloud_negative.png" alt="Negative Sentiment Word Cloud">
                        <div class="markdown-content">
                            {markdown(gemini_responses["negative"])}
                        </div>
                    </div>
                </div>
            """

        if "neutral" in gemini_responses and wordcloud_neutral:
            results_html += f"""
                <div class="result-card">
                    <div class="card-header">
                        <i class="fas fa-meh" style="color: #f59e0b;"></i>
                        Neutral Sentiment Analysis
                    </div>
                    <div class="card-body">
                        <img src="/static/wordcloud_neutral.png" alt="Neutral Sentiment Word Cloud">
                        <div class="markdown-content">
                            {markdown(gemini_responses["neutral"])}
                        </div>
                    </div>
                </div>
            """

        if response_result:
            results_html += f"""
                <div class="result-card">
                    <div class="card-header">
                        <i class="fas fa-lightbulb"></i>
                        Executive Summary & Strategic Recommendations
                    </div>
                    <div class="card-body">
                        <div class="markdown-content">
                            {markdown(response_result)}
                        </div>
                    </div>
                </div>
            """

        results_html += f"""
                <div class="download-section">
                    <h2><i class="fas fa-download"></i> Download Your Report</h2>
                    <p>Get a comprehensive PDF report with all insights and recommendations</p>
                    <a href="/static/sentiment_analysis_report.pdf" class="download-btn" target="_blank">
                        <i class="fas fa-file-pdf"></i> Download PDF Report
                    </a>
                </div>

                <div style="text-align: center;">
                    <a href="/" class="back-btn">
                        <i class="fas fa-arrow-left"></i> Analyze Another Video
                    </a>
                </div>
            </div>

            <script>
                // Add smooth scrolling and animations
                document.addEventListener('DOMContentLoaded', function() {{
                    // Stagger card animations
                    const cards = document.querySelectorAll('.result-card');
                    cards.forEach((card, index) => {{
                        card.style.animationDelay = (index * 0.1) + 's';
                    }});

                    // Add click-to-zoom for images
                    const images = document.querySelectorAll('.card-body img');
                    images.forEach(img => {{
                        img.addEventListener('click', function() {{
                            if (this.style.position === 'fixed') {{
                                // Close zoom
                                this.style.position = '';
                                this.style.top = '';
                                this.style.left = '';
                                this.style.width = '';
                                this.style.height = '';
                                this.style.zIndex = '';
                                this.style.cursor = '';
                                document.body.style.overflow = '';
                            }} else {{
                                // Open zoom
                                this.style.position = 'fixed';
                                this.style.top = '50%';
                                this.style.left = '50%';
                                this.style.width = '90%';
                                this.style.height = 'auto';
                                this.style.transform = 'translate(-50%, -50%)';
                                this.style.zIndex = '1000';
                                this.style.cursor = 'zoom-out';
                                document.body.style.overflow = 'hidden';
                            }}
                        }});
                    }});
                }});
            </script>
        </body>
        </html>
        """

        logger.info("Analysis completed successfully")
        return results_html

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_youtube: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a friendly error page
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analysis Error</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 0;
                }}
                .error-container {{
                    background: white;
                    padding: 3rem;
                    border-radius: 24px;
                    text-align: center;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
                    max-width: 600px;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .error-icon {{
                    font-size: 4rem;
                    color: #ef4444;
                    margin-bottom: 1.5rem;
                    animation: shake 1s infinite;
                }}
                @keyframes shake {{
                    0%, 100% {{ transform: translateX(0); }}
                    25% {{ transform: translateX(-5px); }}
                    75% {{ transform: translateX(5px); }}
                }}
                .error-title {{
                    color: #dc2626;
                    font-size: 1.8rem;
                    font-weight: 700;
                    margin-bottom: 1rem;
                }}
                .error-message {{
                    color: #6b7280;
                    margin-bottom: 2rem;
                    line-height: 1.6;
                }}
                .btn-back {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 1rem 2rem;
                    text-decoration: none;
                    border-radius: 12px;
                    display: inline-block;
                    transition: all 0.3s ease;
                    font-weight: 600;
                }}
                .btn-back:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
                    text-decoration: none;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <div class="error-icon">
                    <i class="fas fa-exclamation-circle"></i>
                </div>
                <h2 class="error-title">Processing Error</h2>
                <p class="error-message">
                    We encountered an error while processing your request. This could be due to:
                    <br>• Network connectivity issues
                    <br>• High server load
                    <br>• Invalid video URL format
                    <br>• Temporary service disruption
                    <br><br>
                    <strong>Error:</strong> {str(e)[:200]}...
                </p>
                <a href="/" class="btn-back">
                    <i class="fas fa-arrow-left"></i> Try Again
                </a>
            </div>
        </body>
        </html>
        """

# Add error handler for development
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return HTMLResponse(
        content="""
        <html>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #dc2626;">Internal Server Error</h1>
                <p>Something went wrong on our end. Please try again later.</p>
                <a href="/" style="background: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Go Home</a>
            </body>
        </html>
        """,
        status_code=500
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
        timeout_keep_alive=600,
        log_level="info",
        access_log=True,
    )