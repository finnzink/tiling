import * as THREE from 'three';

export function createDebugText() {
    const div = document.createElement('div');
    div.style.position = 'absolute';
    div.style.top = '10px';
    div.style.right = '10px';
    div.style.color = 'white';
    div.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    div.style.padding = '5px';
    div.style.borderRadius = '5px';
    div.style.display = 'none';
    div.textContent = 'Debug Mode: ON';
    document.body.appendChild(div);
    return div;
}

export function updateDebugText(debugText, debugMode, colemakMode) {
    if (!debugText) {
        console.warn('Debug text element not initialized');
        return;
    }
    
    let text = [];
    if (debugMode) text.push('Debug Mode: ON');
    if (colemakMode) text.push('Colemak Mode: ON');
    
    debugText.textContent = text.join(' | ');
    debugText.style.display = text.length > 0 ? 'block' : 'none';
}

export function createRhomb(vertices, faceIndices, color = 0x808080, isGhost = false) {
    const geometry = new THREE.BufferGeometry();
    
    const positions = vertices.flat();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    
    const indices = [];
    faceIndices.forEach(face => {
        indices.push(face[0], face[1], face[2]);
        indices.push(face[0], face[2], face[3]);
    });
    geometry.setIndex(indices);
    
    geometry.computeVertexNormals();
    const flatGeometry = geometry.toNonIndexed();
    flatGeometry.computeVertexNormals();

    const normals = flatGeometry.getAttribute('normal');
    const colors = [];
    for (let i = 0; i < normals.count; i++) {
        const normal = new THREE.Vector3().fromBufferAttribute(normals, i);
        const color = new THREE.Color()
            .setRGB(normal.x, normal.y, normal.z)
            .multiplyScalar(0.5)
            .addScalar(0.5);
        colors.push(color.r, color.g, color.b);
    }
    flatGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: isGhost,
        opacity: isGhost ? 0.3 : 1,
        flatShading: true
    });

    return new THREE.Mesh(flatGeometry, material);
}
