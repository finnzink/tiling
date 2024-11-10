import * as THREE from 'three';
import { createRhomb } from './utils.js';

let scene, camera, renderer;
let raycaster, mouse;
let hoverSphere, ghostCell, coiSphere, coiBox;
let jsonData;
let debugMode = false;
let isMouseDragging = false;
let mouseDownTime = 0;

export function initViewer(sceneRef, cameraRef, rendererRef) {
    scene = sceneRef;
    camera = cameraRef;
    renderer = rendererRef;
    
    // Setup raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Create hover sphere
    const sphereGeometry = new THREE.SphereGeometry(0.05);
    const sphereMaterial = new THREE.MeshBasicMaterial({ 
        color: 0xff0000,
        transparent: true,
        opacity: 0.8,
        depthTest: false,
        side: THREE.DoubleSide
    });
    hoverSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    hoverSphere.visible = false;
    hoverSphere.renderOrder = 999;
    scene.add(hoverSphere);

    // Create COI sphere
    const coiGeometry = new THREE.SphereGeometry(0.1);
    const coiMaterial = new THREE.MeshBasicMaterial({ 
        color: 0x0000ff,
        transparent: true,
        opacity: 0.8,
        depthTest: false,
        side: THREE.DoubleSide
    });
    coiSphere = new THREE.Mesh(coiGeometry, coiMaterial);
    coiSphere.visible = false;
    coiSphere.renderOrder = 999;
    scene.add(coiSphere);

    // Add mouse event listeners
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mouseup', onMouseUp);
    document.addEventListener('mousemove', onMouseMove);
}

export function loadCells(data) {
    jsonData = data;
    
    // Clear existing cells
    scene.children = scene.children.filter(child => 
        !(child instanceof THREE.Mesh) || 
        child === hoverSphere || 
        child === coiSphere
    );

    const cells = data.cells;
    const colorMap = new THREE.Color();
    
    Object.entries(cells).forEach(([uuid, cell], index) => {
        if (cell.filled) {
            colorMap.setHSL(index / Object.keys(cells).length, 0.5, 0.5);
            const rhomb = createRhomb(cell.vertices, cell.face_indices, colorMap.getHex());
            rhomb.userData.uuid = uuid;
            scene.add(rhomb);
        }
    });

    // Handle center of interest
    if (data.center_of_interest) {
        const coi = data.center_of_interest;
        coiSphere.position.set(coi[0], coi[1], coi[2]);
        coiSphere.visible = debugMode;

        // Create COI box
        if (coiBox) scene.remove(coiBox);
        const boxGeometry = new THREE.BoxGeometry(4, 4, 4);
        const boxMaterial = new THREE.LineBasicMaterial({ 
            color: 0x0000ff,
            transparent: true,
            opacity: 0.5
        });
        const boxWireframe = new THREE.WireframeGeometry(boxGeometry);
        coiBox = new THREE.LineSegments(boxWireframe, boxMaterial);
        coiBox.position.set(coi[0], coi[1], coi[2]);
        coiBox.visible = debugMode;
        scene.add(coiBox);
    }

    // Reset camera to view all cells
    const bbox = new THREE.Box3().setFromObject(scene);
    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(center.x + maxDim, center.y + maxDim, center.z + maxDim);
    camera.lookAt(center);
}

function onMouseDown(event) {
    // Reset drag state
    isMouseDragging = false;
    mouseDownTime = Date.now();
}

function onMouseUp(event) {
    if (!isMouseDragging && (Date.now() - mouseDownTime) < 200) {
        handleCellClick();
    }
}

function handleCellClick() {
    if (ghostCell) {
        const uuid = ghostCell.userData.uuid;
        const cellData = jsonData.cells[uuid];
        
        if (cellData && !cellData.filled) {
            cellData.filled = true;
            scene.remove(ghostCell);

            const filledRhomb = createRhomb(cellData.vertices, cellData.face_indices, 0x808080, false);
            filledRhomb.userData.uuid = uuid;
            scene.add(filledRhomb);

            ghostCell = null;
        }
    }
}

function onMouseMove(event) {
    if (document.pointerLockElement === renderer.domElement) {
        mouse.x = 0;
        mouse.y = 0;

        raycaster.setFromCamera(mouse, camera);
        const intersectableObjects = scene.children.filter(child => 
            child instanceof THREE.Mesh && 
            child !== hoverSphere && 
            child !== ghostCell
        );
        const intersects = raycaster.intersectObjects(intersectableObjects);

        // Reset previous state
        scene.children.forEach(child => {
            if (child instanceof THREE.Mesh && child !== hoverSphere) {
                if (child.material.emissive) {
                    child.material.emissive.setHex(0x000000);
                }
            }
        });
        if (ghostCell) {
            scene.remove(ghostCell);
            ghostCell = null;
        }

        // Handle intersections
        if (intersects.length > 0) {
            handleIntersection(intersects[0]);
        } else {
            hoverSphere.visible = false;
        }
    }
}

function handleIntersection(intersect) {
    const intersected = intersect.object;
    if (intersected !== hoverSphere) {
        if (intersected.material.emissive) {
            intersected.material.emissive.setHex(0x333333);
        }
        
        const face = intersect.face;
        const geometry = intersected.geometry;
        const positionAttribute = geometry.getAttribute('position');
        
        const triCenter = calculateTriangleCenter(positionAttribute, face);
        triCenter.applyMatrix4(intersected.matrixWorld);
        
        const triCenterKey = triCenter.toArray()
            .map(x => x.toFixed(2))
            .join(',');
        
        const triangleData = jsonData.triangles[triCenterKey];
        
        if (triangleData && triangleData.cells.length > 1) {
            createGhostCell(triangleData, intersected.userData.uuid);
        }

        updateHoverSphere(triCenter);
    }
}

export function setDebugMode(mode) {
    debugMode = mode;
    if (coiSphere) coiSphere.visible = mode;
    if (coiBox) coiBox.visible = mode;
}

function calculateTriangleCenter(positionAttribute, face) {
    const a = new THREE.Vector3().fromBufferAttribute(positionAttribute, face.a);
    const b = new THREE.Vector3().fromBufferAttribute(positionAttribute, face.b);
    const c = new THREE.Vector3().fromBufferAttribute(positionAttribute, face.c);
    
    return new THREE.Vector3()
        .add(a)
        .add(b)
        .add(c)
        .multiplyScalar(1/3);
}

function updateHoverSphere(position) {
    const directionToCamera = new THREE.Vector3()
        .subVectors(camera.position, position)
        .normalize();
    position.add(directionToCamera.multiplyScalar(0.01));
    
    hoverSphere.position.copy(position);
    hoverSphere.visible = debugMode;
}

function createGhostCell(triangleData, intersectedUUID) {
    const otherUUID = triangleData.cells.find(uuid => uuid !== intersectedUUID);
    const otherCell = jsonData.cells[otherUUID];
    
    if (otherCell && !otherCell.filled) {
        ghostCell = createRhomb(otherCell.vertices, otherCell.face_indices, 0x808080, true);
        ghostCell.userData.uuid = otherUUID;
        scene.add(ghostCell);
    }
}
