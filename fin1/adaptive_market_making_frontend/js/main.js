// Main JavaScript file for Adaptive Market Making Frontend

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize animations
    initAnimations();
    
    // Initialize header scroll effect
    initHeaderScroll();
    
    // Initialize interactive elements
    initInteractiveElements();
    
    // Initialize creator photo effects
    initCreatorPhotoEffects();
    
    // Initialize charts and visualizations
    initCharts();
});

// Header scroll effect
function initHeaderScroll() {
    const header = document.querySelector('.header');
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
}

// Initialize animations
function initAnimations() {
    // Check if IntersectionObserver is supported
    if ('IntersectionObserver' in window) {
        // Create observer for fade-in animations
        const fadeObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    fadeObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1
        });
        
        // Observe all elements with animation classes
        document.querySelectorAll('.fade-in, .fade-in-delay-1, .fade-in-delay-2, .fade-in-delay-3').forEach(el => {
            fadeObserver.observe(el);
        });
        
        // Create observer for feature cards
        const featureObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    // Add staggered animation delay
                    setTimeout(() => {
                        entry.target.classList.add('visible');
                    }, index * 150);
                    featureObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1
        });
        
        // Observe all feature cards
        document.querySelectorAll('.feature-card').forEach(el => {
            featureObserver.observe(el);
        });
    } else {
        // Fallback for browsers that don't support IntersectionObserver
        document.querySelectorAll('.fade-in, .fade-in-delay-1, .fade-in-delay-2, .fade-in-delay-3, .feature-card').forEach(el => {
            el.classList.add('visible');
        });
    }
    
    // Animate floating elements
    animateFloatingElements();
}

// Animate floating elements in hero section
function animateFloatingElements() {
    const floatingElements = document.querySelectorAll('.floating-element');
    
    floatingElements.forEach((el, index) => {
        // Set random initial positions
        const randomX = Math.random() * 20 - 10;
        const randomY = Math.random() * 20 - 10;
        
        el.style.transform = `translate(${randomX}px, ${randomY}px)`;
        
        // Animate with GSAP if available
        if (typeof gsap !== 'undefined') {
            gsap.to(el, {
                x: randomX * -1,
                y: randomY * -1,
                duration: 10 + index * 2,
                repeat: -1,
                yoyo: true,
                ease: "sine.inOut"
            });
        }
    });
}

// Initialize interactive elements
function initInteractiveElements() {
    // Mobile menu toggle
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', function() {
            navLinks.classList.toggle('active');
        });
    }
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                // Close mobile menu if open
                if (navLinks && navLinks.classList.contains('active')) {
                    navLinks.classList.remove('active');
                }
                
                // Scroll to target
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Initialize interactive feature cards
    initFeatureCards();
}

// Initialize feature cards interaction
function initFeatureCards() {
    const featureCards = document.querySelectorAll('.feature-card');
    
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            // Add hover effect
            this.classList.add('hover');
            
            // Animate icon if GSAP is available
            const icon = this.querySelector('.feature-icon');
            if (icon && typeof gsap !== 'undefined') {
                gsap.to(icon, {
                    y: -10,
                    scale: 1.1,
                    duration: 0.3,
                    ease: "back.out(1.7)"
                });
            }
        });
        
        card.addEventListener('mouseleave', function() {
            // Remove hover effect
            this.classList.remove('hover');
            
            // Reset icon animation if GSAP is available
            const icon = this.querySelector('.feature-icon');
            if (icon && typeof gsap !== 'undefined') {
                gsap.to(icon, {
                    y: 0,
                    scale: 1,
                    duration: 0.3,
                    ease: "back.out(1.7)"
                });
            }
        });
    });
}

// Initialize creator photo effects
function initCreatorPhotoEffects() {
    const photoContainer = document.querySelector('.creator-photo-container');
    const photo = document.querySelector('.creator-photo');
    
    if (photoContainer && photo) {
        // 3D rotation effect on mouse move
        photoContainer.addEventListener('mousemove', function(e) {
            const rect = photoContainer.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            
            const rotateX = y / 10;
            const rotateY = -x / 10;
            
            photo.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        });
        
        // Reset on mouse leave
        photoContainer.addEventListener('mouseleave', function() {
            photo.style.transform = 'rotateX(0) rotateY(0)';
        });
        
        // Pulse effect on click
        photoContainer.addEventListener('click', function() {
            photo.classList.add('pulse');
            
            setTimeout(() => {
                photo.classList.remove('pulse');
            }, 1000);
        });
    }
}

// Initialize charts and visualizations
function initCharts() {
    // Check if Chart.js is available
    if (typeof Chart !== 'undefined') {
        // Market regime chart
        const regimeCtx = document.getElementById('regime-chart');
        if (regimeCtx) {
            new Chart(regimeCtx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{
                        label: 'Market Volatility',
                        data: [12, 19, 3, 5, 2, 3, 20, 25, 15, 10, 5, 2],
                        borderColor: '#00c4cc',
                        backgroundColor: 'rgba(0, 196, 204, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        // Performance chart
        const perfCtx = document.getElementById('performance-chart');
        if (perfCtx) {
            new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{
                        label: 'Strategy Returns',
                        data: [1, 3, 7, 12, 15, 18, 22, 25, 30, 35, 40, 45],
                        borderColor: '#0a2540',
                        backgroundColor: 'rgba(10, 37, 64, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Benchmark',
                        data: [1, 2, 5, 8, 10, 12, 15, 17, 20, 22, 25, 27],
                        borderColor: '#ff5e5b',
                        backgroundColor: 'rgba(255, 94, 91, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
    }
    
    // Initialize 3D visualization if Three.js is available
    if (typeof THREE !== 'undefined') {
        initMarketVisualization();
    }
}

// Initialize 3D market visualization
function initMarketVisualization() {
    const container = document.getElementById('market-visualization');
    
    if (!container) return;
    
    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8f9fa);
    
    // Create camera
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 30;
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Create market data points
    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
    const material = new THREE.MeshBasicMaterial({ color: 0x00c4cc });
    
    const dataPoints = [];
    for (let i = 0; i < 100; i++) {
        const point = new THREE.Mesh(geometry, material);
        
        // Position points in a 3D space representing market features
        point.position.x = (Math.random() - 0.5) * 40;
        point.position.y = (Math.random() - 0.5) * 40;
        point.position.z = (Math.random() - 0.5) * 40;
        
        scene.add(point);
        dataPoints.push(point);
    }
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        
        // Rotate camera around the scene
        camera.position.x = 30 * Math.sin(Date.now() * 0.0005);
        camera.position.z = 30 * Math.cos(Date.now() * 0.0005);
        camera.lookAt(scene.position);
        
        // Animate data points
        dataPoints.forEach((point, index) => {
            point.position.y += Math.sin(Date.now() * 0.001 + index) * 0.02;
        });
        
        renderer.render(scene, camera);
    }
    
    // Handle window resize
    window.addEventListener('resize', function() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
    
    // Start animation
    animate();
}

// Additional animations for creator section
function initCreatorAnimations() {
    const creatorSection = document.querySelector('.creator-section');
    const creatorPhoto = document.querySelector('.creator-photo');
    const creatorName = document.querySelector('.creator-name');
    const creatorTitle = document.querySelector('.creator-title');
    const creatorSocial = document.querySelector('.creator-social');
    
    if (creatorSection && typeof gsap !== 'undefined') {
        // Create timeline
        const tl = gsap.timeline({
            scrollTrigger: {
                trigger: creatorSection,
                start: "top 80%",
                end: "bottom 20%",
                toggleActions: "play none none reverse"
            }
        });
        
        // Add animations
        tl.from(creatorPhoto, {
            scale: 0,
            opacity: 0,
            rotation: -180,
            duration: 1,
            ease: "back.out(1.7)"
        })
        .from(creatorName, {
            y: 30,
            opacity: 0,
            duration: 0.6
        }, "-=0.3")
        .from(creatorTitle, {
            y: 30,
            opacity: 0,
            duration: 0.6
        }, "-=0.3")
        .from(creatorSocial.children, {
            y: 30,
            opacity: 0,
            stagger: 0.1,
            duration: 0.6
        }, "-=0.3");
    }
}

// Load external libraries if needed
function loadExternalLibraries() {
    // Load GSAP
    if (typeof gsap === 'undefined') {
        const gsapScript = document.createElement('script');
        gsapScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.11.4/gsap.min.js';
        document.head.appendChild(gsapScript);
        
        // Load ScrollTrigger plugin
        const scrollTriggerScript = document.createElement('script');
        scrollTriggerScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.11.4/ScrollTrigger.min.js';
        document.head.appendChild(scrollTriggerScript);
    }
    
    // Load Chart.js
    if (typeof Chart === 'undefined') {
        const chartScript = document.createElement('script');
        chartScript.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        document.head.appendChild(chartScript);
    }
    
    // Load Three.js
    if (typeof THREE === 'undefined') {
        const threeScript = document.createElement('script');
        threeScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
        document.head.appendChild(threeScript);
    }
}

// Call to load external libraries
loadExternalLibraries();
