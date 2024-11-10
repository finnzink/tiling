import * as THREE from 'three';
import { setupControls, moveState, moveSpeed, getCameraRotation, colemakMode } from './controls.js';
import { createDebugText, updateDebugText } from './utils.js';
import { initViewer, loadCells, setDebugMode } from './viewer.js';

// Global state
let scene, camera, renderer;
let debugMode = false;
let debugText;

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Camera setup
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Setup controls
    setupControls(camera, renderer);

    // Add lighting
    const pointLight = new THREE.PointLight(0xffaa00, 0.5);
    pointLight.position.set(-5, -5, 5);
    scene.add(pointLight);

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Create debug text BEFORE calling updateDebugText
    debugText = createDebugText();

    // Initialize viewer components
    initViewer(scene, camera, renderer);

    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    
    // Handle movement
    const { yaw } = getCameraRotation();
    const direction = new THREE.Vector3();
    
    if (moveState.forward) {
        direction.z -= Math.cos(yaw);
        direction.x -= Math.sin(yaw);
    }
    if (moveState.backward) {
        direction.z += Math.cos(yaw);
        direction.x += Math.sin(yaw);
    }
    if (moveState.left) {
        direction.x -= Math.cos(yaw);
        direction.z += Math.sin(yaw);
    }
    if (moveState.right) {
        direction.x += Math.cos(yaw);
        direction.z -= Math.sin(yaw);
    }
    if (moveState.up) direction.y += 1;
    if (moveState.down) direction.y -= 1;
    
    // Only move if there's input
    if (direction.lengthSq() > 0) {
        direction.normalize();
        camera.position.addScaledVector(direction, moveSpeed);
    }
    
    renderer.render(scene, camera);
}

// Initialize everything
init();

// Replace the JSON file loading with API call
fetch('http://localhost:9000/lambda-url/dualgrid/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        center_point: [10.0, -10.0, 10.0],
        k_range: 2,
        cube_size: 8.0
    })
})
.then(response => response.json())
.then(jsonData => {
    loadCells(jsonData);
})
.catch(error => {
    console.error('Error fetching cell data:', error);
});

// Only call updateDebugText after debugText is created
updateDebugText(debugText, debugMode, colemakMode);

// Add debug mode toggle
export function toggleDebugMode() {
    debugMode = !debugMode;
    setDebugMode(debugMode);
    updateDebugText(debugText, debugMode, colemakMode);
}
