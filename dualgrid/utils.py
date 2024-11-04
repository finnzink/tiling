import numpy as np
import dualgrid as dg
import time
from multiprocessing import Pool
from functools import partial
import json
import uuid  # Add this import at the top

""" OFFSET generation
"""
def offsets_fixed_around_centre(n):
    """
    Gives offsets that give a tiling around centre of rotation.
    """
    return np.array([1/n for _i in range(n)])

def generate_offsets(num, random=True, below_one=False, sum_zero=False, centred=False):
    """
    Generates offsets for use in the dualgrid method.
    num: Number of offsets to generate. Usually the same as the number of basis vectors.
    random: Generate random offsets? Or the same random ones each time (fixed seed).
    below_one: Keep all of the offsets below one - divides by num. Only makes a difference when sum_zero=True
    sum_zero: Make the offsets sum to 0. For example Penrose tiling offsets must sum to 0.
    centred: Give a tiling centred around the centre of rotation. (NOTE: Not sure if this is working in 3D)
    """
    if centred:
        return offsets_fixed_around_centre(num)

    if random:
        offsets = np.random.random(num)
    else:
        # Fixed offsets for deterministic results
        offsets = np.array([0.27, 0.37, 0.47, 0.57, 0.67, 0.77][:num])

    if below_one:
        offsets /= num

    if sum_zero:
        offsets[-1] = -np.sum(offsets[:-1])

    return offsets


""" BASES
    Various pre-defined bases to play around with
"""
def icosahedral_basis(random_offsets=True, **kwargs):
    offsets = generate_offsets(6, random_offsets, **kwargs)

    # From: https://physics.princeton.edu//~steinh/QuasiPartII.pdf
    sqrt5 = np.sqrt(5)
    icos = [
        np.array([(2.0 / sqrt5) * np.cos(2 * np.pi * n / 5),
                  (2.0 / sqrt5) * np.sin(2 * np.pi * n / 5),
                  1.0 / sqrt5])
        for n in range(5)
    ]
    icos.append(np.array([0.0, 0.0, 1.0]))
    
    # Debug print
    print("Python icosahedral basis:")
    for i, vec in enumerate(icos):
        print(f"v{i}: [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")
    print("Offsets:", offsets)
    
    return dg.Basis(np.array(icos), offsets)


""" Filtering functions. Must take form (point, filter_centre, param_1, param_2, ..., param_N)
"""
def is_point_within_radius(r, filter_centre, radius):
    """
    Used to retain cells within a certain radius of the filter_centre.
    """
    return np.linalg.norm(r - filter_centre) < radius

def is_point_within_cube(r, filter_centre, size):
    """
    Used to retain cells within a cube centred at filter_centre.
    N dimensional cube
    """
    diff = r - filter_centre
    sizediv2 = size/2.0

    return np.sum([abs(d) > sizediv2 for d in diff]) == 0

def contains_value(r, filter_centre, value):
    """
    Checks for index "r" within indices. Not for use with real space.
    """
    return value in list(r)

def elements_are_below(r, filter_centre, value):
    """
    Checks if all elements are below the given value
    """
    for element in r:
        if element > value:
            return False

    return True



def get_centre_of_interest(cells):
    """
    Used to centre the camera/filter on the densest part of the generated crystal.
    """
    all_verts = []
    for c in cells:
        for v in c.verts:
            all_verts.append(v)

    return np.mean(all_verts, axis=0)  # centre of interest is mean of all the vertices

""" Graph & filtering
"""

def filter_cells(cells, filter, filter_args=[], filter_centre=None, fast_filter=False, filter_indices=False, invert_filter=False):
    """
    The position and indices of the vertex is embedded in each node.
    Takes a function to filter out points with, along with it's arguments.
    """

    if type(filter_centre) == type(None):
        print("FINDING COI")
        # Find centre of interest
        filter_centre = get_centre_of_interest(cells)
        print("COI:", filter_centre)

    filter_results = [c.is_in_filter(filter, filter_centre, filter_args, fast=fast_filter, filter_indices=filter_indices, invert_filter=invert_filter) for c in cells]
    
    return [cell for cell, keep in zip(cells, filter_results) if keep]


def to_native(obj):
    """Convert numpy types to native Python types recursively."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [to_native(item) for item in obj]
    elif isinstance(obj, list):
        return [to_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_native(value) for key, value in obj.items()}
    return obj

def cells_to_dict(cells, center_point=None):
    """
    Converts a list of Cell objects to a dictionary format suitable for JSON serialization.
    Includes cells (indexed by UUID), triangles (indexed by triangle center position), and COI.
    """
    # Define face indices for 3D rhombohedron (initial configuration)
    FACE_INDICES = np.array([
        [0, 2, 3, 1],  # front
        [0, 1, 5, 4],  # right
        [5, 7, 6, 4],  # back
        [2, 6, 7, 3],  # left
        [0, 4, 6, 2],  # top
        [3, 7, 5, 1]   # bottom
    ])

    # Initialize output structure
    result = {
        'cells': {},    # Will be indexed by UUID
        'triangles': {},  # Will be indexed by triangle center position string
        'center_of_interest': center_point   # Add COI to output
    }
    
    def calculate_face_normal(vertices):
        """Calculate normal vector for a face using first three vertices."""
        v0, v1, v2 = vertices[:3]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        return normal / np.linalg.norm(normal)

    def face_needs_flip(vertices, cell_center):
        """Check if face normal points outward from cell center."""
        normal = calculate_face_normal(vertices)
        face_center = np.mean(vertices, axis=0)
        direction_from_center = face_center - cell_center
        # If dot product is negative, normal points inward
        return np.dot(normal, direction_from_center) < 0

    def analyze_and_adjust_triangulation(vertices, face_indices, cell_center):
        """Determine triangulation type and adjust orientation if needed."""
        v0, v1, v2, v3 = vertices
        diag1 = np.linalg.norm(v2 - v0)
        diag2 = np.linalg.norm(v3 - v1)
        
        # First determine which diagonal to use
        if diag1 > diag2:
            # Use short diagonal (v1-v3)
            face_indices = [face_indices[1], face_indices[2], face_indices[3], face_indices[0]]
            vertices = [vertices[1], vertices[2], vertices[3], vertices[0]]
        
        # Check if face needs to be flipped
        if face_needs_flip(vertices, cell_center):
            # Reverse vertex order to flip face orientation
            return [face_indices[0], face_indices[3], face_indices[2], face_indices[1]]
        
        return face_indices

    # Process each cell
    for i, cell in enumerate(cells):
        cell_uuid = str(uuid.uuid4())
        cell_center = np.mean(cell.verts, axis=0)
        
        # Add cell to cells dict
        result['cells'][cell_uuid] = {
            'vertices': to_native(cell.verts),
            'indices': to_native(cell.indices),
            'intersection': to_native(cell.intersection),
            'filled': True,
            'face_indices': []
        }
        
        # Analyze and adjust triangulation for all cells
        adjusted_face_indices = []
        if i == 0:
            print(f"Triangulation analysis and adjustment for first cell (UUID: {cell_uuid}):")
        
        for face_index, face_indices in enumerate(FACE_INDICES):
            face_vertices = [cell.verts[idx] for idx in face_indices]
            adjusted_indices = analyze_and_adjust_triangulation(
                face_vertices, 
                face_indices, 
                cell_center
            )
            adjusted_face_indices.append(adjusted_indices)
            
            # Add the adjusted face indices to the cell data
            result['cells'][cell_uuid]['face_indices'].append(to_native(adjusted_indices))
        
        # Use the adjusted face indices for triangle generation
        for face_indices in adjusted_face_indices:
            # Each face is split into two triangles
            triangles = [
                [face_indices[0], face_indices[1], face_indices[2]],  # First triangle
                [face_indices[0], face_indices[2], face_indices[3]]   # Second triangle
            ]
            
            for triangle in triangles:
                # Get vertices for this triangle
                tri_verts = [cell.verts[idx] for idx in triangle]
                
                # Calculate triangle center
                tri_center = np.mean(tri_verts, axis=0)
                # Round the values before creating the key
                tri_center_key = ','.join(f"{x:.2f}" for x in to_native(tri_center))
                
                # Add or update triangle entry
                if tri_center_key not in result['triangles']:
                    result['triangles'][tri_center_key] = {
                        'center': to_native(tri_center),
                        'cells': [cell_uuid]
                    }
                else:
                    # Add this cell to existing triangle if it's not already there
                    if cell_uuid not in result['triangles'][tri_center_key]['cells']:
                        result['triangles'][tri_center_key]['cells'].append(cell_uuid)

    return to_native(result)

def export_cells_to_json(cells, filepath, center_point=None):
    """
    Saves the cells to a JSON file.
    
    Args:
        cells: List of Cell objects to export
        filepath: Path to save the JSON file
    """
    # Debug prints for only cells 2 and 3
    print("Pre-export cell vertices:")
    for i, cell in enumerate(cells):
        if i in [1, 2]:  # Only show cells at index 2 and 3
            print(f"Cell {i}:")
            for j, vert in enumerate(cell.verts):
                print(f"  Vertex {j}: [{vert[0]}, {vert[1]}, {vert[2]}]")
    
    data = cells_to_dict(cells, center_point)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

