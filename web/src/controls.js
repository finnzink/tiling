import { updateDebugText } from './utils.js';

export let moveState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false
};

export let colemakMode = true;
export const moveSpeed = 0.1;
export const rotationSpeed = 0.002;

let pitch = 0;
let yaw = 0;
let camera;
let renderer;
let debugMode = false;
let debugText;

export function setupControls(cam, rendererRef) {
    camera = cam;
    renderer = rendererRef;
    
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mouseup', onMouseUp);
    document.addEventListener('mousemove', onMouseMove);

    renderer.domElement.addEventListener('click', () => {
        renderer.domElement.requestPointerLock();
    });
}

function onKeyDown(event) {
    switch (event.code) {
        case colemakMode ? 'KeyF' : 'KeyW': moveState.forward = true; break;
        case colemakMode ? 'KeyS' : 'KeyS': moveState.backward = true; break;
        case colemakMode ? 'KeyR' : 'KeyA': moveState.left = true; break;
        case colemakMode ? 'KeyT' : 'KeyD': moveState.right = true; break;
        case colemakMode ? 'ShiftLeft' : 'Space': moveState.up = true; break;
        case colemakMode ? 'Backspace' : 'ShiftLeft': moveState.down = true; break;
        case 'KeyB': 
            debugMode = !debugMode;
            updateDebugText(debugText, debugMode, colemakMode);
            break;
        case 'KeyC': 
            colemakMode = !colemakMode;
            updateDebugText(debugText, debugMode, colemakMode);
            break;
    }
}

function onKeyUp(event) {
    switch (event.code) {
        case colemakMode ? 'KeyF' : 'KeyW': moveState.forward = false; break;
        case colemakMode ? 'KeyS' : 'KeyS': moveState.backward = false; break;
        case colemakMode ? 'KeyR' : 'KeyA': moveState.left = false; break;
        case colemakMode ? 'KeyT' : 'KeyD': moveState.right = false; break;
        case colemakMode ? 'ShiftLeft' : 'Space': moveState.up = false; break;
        case colemakMode ? 'Backspace' : 'ShiftLeft': moveState.down = false; break;
    }
}

function onMouseMove(event) {
    if (document.pointerLockElement === renderer.domElement) {
        yaw -= event.movementX * rotationSpeed;
        pitch -= event.movementY * rotationSpeed;
        pitch = Math.max(-Math.PI/2, Math.min(Math.PI/2, pitch));
        
        camera.rotation.set(0, 0, 0);
        camera.rotateY(yaw);
        camera.rotateX(pitch);
    }
}

function onMouseDown(event) {
    // Mouse down handling if needed
}

function onMouseUp(event) {
    // Mouse up handling if needed
}

export function getCameraRotation() {
    return { pitch, yaw };
}
