import numpy as np
import itertools
from multiprocessing import Pool
from functools import partial

def _get_k_combos(k_range, dimensions):
    print(f"[PY] k_range input: {k_range}")
    """
    Returns all possible comparison between two sets of lines for dimension number "dimensions" and max k_range (index range)
    If center_point is provided, the k_range will be centered around that point in real space.
    """

    range_list = [k for k in range(-k_range, k_range + 1)]
    print(f"[PY] k_combo range: {range_list}")
    return np.array(list(itertools.product(*[
        range_list
        for d in range(dimensions)
    ])))

    print(f"Python k_combos shape: {combos.shape}")  # Debug print
    return combos

class ConstructionSet:
    """
    A class to represent a set of parallel lines / planes / n dimensional parallel structure.
    It implements a single method to return all intersections with another ConstructionSet.
    """
    def __init__(self, normal, offset):
        """
        normal: Normal vector to this construction set.
        offset: Offset of these lines from the origin.
        """
        self.normal = normal
        self.offset = offset

    def get_intersections_with(self, k_range, others, center_point=None):
        """
        Calculates all intersections between this set of lines/planes and another.
        center_point: Point in real space to center the k_range around
        """
        print(f"[PY] Computing intersections with k_range: {k_range}")
        
        dimensions = len(self.normal)
        coef_matrix = np.array([self.normal, *[o.normal for o in others]])

        if np.linalg.det(coef_matrix) == 0:
            print("WARNING: Unit vectors form singular matrices.")
            return [], []

        coef_inv = np.linalg.inv(coef_matrix)
        k_combos = _get_k_combos(k_range, dimensions)
        print(f"[PY] Generated {len(k_combos)} k_combos")

        # Get the base offsets
        base_offsets = np.array([self.offset, *[o.offset for o in others]])
        
        if center_point is not None:
            # Calculate the plane indices for the center point
            # For each normal vector (including this one and others), calculate which plane the point lies on
            center_planes = np.array([
                np.dot(center_point, self.normal),
                *[np.dot(center_point, o.normal) for o in others]
            ])
            # Round to get the nearest plane indices
            center_indices = np.floor(center_planes - base_offsets)
            # Shift k_combos to be centered around these indices
            k_combos = k_combos + center_indices

        ds = k_combos + base_offsets
        intersections = np.asarray((coef_inv * np.asmatrix(ds).T).T)

        # print("Coefficient Matrix:\n", coef_matrix)
        # print("Inverse Coefficient Matrix:\n", coef_inv)
        # print("k_combos after adjustment:\n", k_combos)
        # print("Intersections:\n", intersections)

        print(f"[PY] ds matrix shape: {ds.shape}")
        print(f"[PY] First few ds values: {ds[:5]}")
        print(f"[PY] First intersection: {intersections[0]}")

        print("[PY] ds matrix:\n", np.asmatrix(ds).T)
        print("[PY] coef_inv:\n", coef_inv)
        print("[PY] Result before transpose:\n", coef_inv * np.asmatrix(ds).T)

        return intersections, k_combos

def _get_neighbours(intersection, js, ks, basis):
    """
    For a given intersection, this function returns the grid-space indices of the spaces surrounding the intersection.
    A "grid-space index" is an N dimensional vector of integer values where N is the number of basis vectors. Each element
    corresponds to an integer multiple of a basis vector, which gives the final location of the tile vertex.

    There will always be a set number of neighbours depending on the number of dimensions. For 2D this is 4 (to form a tile),
    for 3D this is 8 (to form a cube), etc...
    """
    print("[PY] get_neighbours input:")
    print(f"[PY]   js: {js}")
    print(f"[PY]   ks: {ks}")
    
    # Get initial indices
    indices = basis.gridspace(intersection)
    print(f"[PY] Gridspace returned indices: {indices}")
    
    # Load known indices
    for i, j in enumerate(js):
        indices[j] = ks[i]
    print(f"[PY] After loading known indices: {indices}")
    print(f"[PY] Initial indices: {indices}")

    # Each possible neighbour of intersection. See eq. 4.5 in de Bruijn paper
    # For example:
    # [0, 0], [0, 1], [1, 0], [1, 1] for 2D
    directions = np.array(list(itertools.product(*[[0, 1] for _i in range(basis.dimensions)])))

    # Copy the intersection indices. This is then incremented for the remaining indices depending on what neighbour it is.
    neighbours = [ np.array([ v for v in indices ]) for _i in range(len(directions)) ]

    # Quick note: Kronecker delta function -> (i == j) = False (0) or True (1) in python. Multiplication of bool is allowed
    # Also from de Bruijn paper 1.
    deltas = [np.array([(j == js[i]) * 1 for j in range(len(basis.vecs))]) for i in range(basis.dimensions)]

    # Apply equation 4.5 in de Bruijn's paper 1, expanded for any basis len and extra third dimension
    for i, e in enumerate(directions): # e Corresponds to epsilon in paper
        neighbours[i] += np.dot(e, deltas)
    
    print(f"[PY] Initial indices: {indices}")
    print(f"[PY] Deltas: {deltas}")
    print(f"[PY] First neighbour: {directions[0]}")

    return neighbours


class Basis:
    """
    Utility class for defining a set of basis vectors. Has conversion functions between different spaces.
    """
    def __init__(self, vecs, offsets):
        self.vecs = vecs
        self.dimensions = len(self.vecs[0])
        self.offsets = offsets

    def realspace(self, indices):
        """
        Gives position of given indices in real space.
        """
        out = np.zeros(self.dimensions, dtype=float)
        for j, e in enumerate(self.vecs):
            out += e * indices[j]

        return out

    def gridspace(self, r):
        """
        Returns where a "real" point lies in grid space.
        """
        out = np.zeros(len(self.vecs), dtype=int)
        for j, e in enumerate(self.vecs):
            print(f"[PY] Vec {j}: {e}")
            dot = np.dot(r, e)
            print(f"[PY] Dot product {j}: {dot}")
            print(f"[PY] Offset {j}: {self.offsets[j]}")
            value = dot - self.offsets[j]
            print(f"[PY] Pre-ceil value {j}: {value}")
            ceil_value = int(np.ceil(value))
            print(f"[PY] Ceil value {j}: {ceil_value}")
            out[j] = ceil_value
        print(f"[PY] Final gridspace indices: {out}")
        return out

    def get_possible_cells(self, decimals):
        """
        Function that finds all possible cell shapes in the final mesh.
        Number of decimal places required for finite hash keys (floats are hard to == )
        Returns a dictionary of volume : [all possible combinations of basis vector to get that volume]
        """
        shapes = {}  # volume : set indices

        for inds in itertools.combinations(range(len(self.vecs)), self.dimensions):
            vol = abs(np.linalg.det(np.matrix([self.vecs[j] for j in inds]))) # Determinant ~ volume

            if vol != 0:
                vol = np.around(vol, decimals=decimals)
                if vol not in shapes.keys():
                    shapes[vol] = [inds]
                else:
                    shapes[vol].append(inds)

        return shapes


class Cell:
    """
    Class to hold a set of four vertices, along with additional information
    """
    def __init__(self, vertices, indices, intersection):
        """
        verts: Corner vertices of the real tile/cell.
        indices: The "grid space" indices of each vertex.
        
        """
        self.verts = vertices
        self.indices = indices
        self.intersection = intersection # The intersection which caused this cell's existance. Used for plotting

    def __repr__(self):
        return "Cell(%s)" % (self.indices[0])

    def __eq__(self, other):
        return self.indices == other.indices

    def is_in_filter(self, *args, **kwargs):
        """
        Utility function for checking whether the cell's center is in rendering distance.
        Uses average of all vertices as the cell's center point.
        """
        def run_filter(filter, filter_centre, filter_args=[], filter_indices=False, fast=False, invert_filter=False):
            if filter_indices:
                # For index-based filtering, use average of indices
                center = np.mean(self.indices, axis=0)
                zero_centre = np.zeros_like(center)
                return filter(center, zero_centre, *filter_args)
            else:
                # For position-based filtering, use average of vertices
                center = np.mean(self.verts, axis=0)
                return filter(center, filter_centre, *filter_args)
        
        result = run_filter(*args, **kwargs)
        if kwargs["invert_filter"]:
            return not result
        else:
            return result
    
@classmethod
def get_edges_from_indices(indices):
    """
    Gets the edges from vertices given.
    Edges will be found when the indices difference
    has a length of 1. I.e sum(index1 - index2)) = 1
    NOTE: Should be the same for all cells for the particular dimension.
            Therefore it only needs to be run once.
    """
    edges = []
    # Compare every index set with every other index set
    for ind1 in range(len(indices)-1):
        for ind2 in range(ind1+1, len(indices)):
            if abs(np.sum( indices[ind1] - indices[ind2] )) == 1:
                edges.append([ind1, ind2])

    return np.array(edges)

def _get_cells_from_construction_sets(construction_sets, k_range, basis, shape_accuracy, js, center_point=None):
    print(f"[PY] Getting cells for js: {js}")
    
    intersections, k_combos = construction_sets[js[0]].get_intersections_with(
        k_range, 
        [construction_sets[j] for j in js[1:]], 
        center_point=center_point
    )
    
    print(f"[PY] Found {len(intersections)} intersections for js {js}")
    
    cells = []
    for i, intersection in enumerate(intersections):
        print(f"[PY] Creating cell from intersection: {intersection}")
        print(f"[PY] Using k_combo: {k_combos[i]}")
        
        # Calculate neighbours for this intersection
        indices_set = _get_neighbours(intersection, js, k_combos[i], basis)
        print(f"[PY] Generated indices set: {indices_set}")
        
        vertices_set = []
        for indices in indices_set:
            vertex = basis.realspace(indices)
            print(f"[PY] Generated vertex: {vertex} from indices: {indices}")
            vertices_set.append(vertex)

        vertices_set = np.array(vertices_set)
        print(f"[PY] Final vertices set: {vertices_set}")
        c = Cell(vertices_set, indices_set, intersection)
        cells.append(c)

    return cells

def construction_sets_from_basis(basis):
    return [ ConstructionSet(e, basis.offsets[i]) for (i, e) in enumerate(basis.vecs) ]

def dualgrid_method(basis, k_range, center_point=None, shape_accuracy=4, single_threaded=False):
    print("[PY] Starting dualgrid_method")
    
    if center_point is not None:
        center_point = np.array(center_point) / 2
        print(f"[PY] Adjusted center_point: {center_point}")
    
    construction_sets = construction_sets_from_basis(basis)
    print(f"[PY] Created {len(construction_sets)} construction sets")
    
    j_combos = list(itertools.combinations(range(len(construction_sets)), basis.dimensions))
    print(f"[PY] Generated {len(j_combos)} j_combos")
    
    cells = []
    if single_threaded:
        print("[PY] Running single-threaded")
        for js in j_combos:
            new_cells = _get_cells_from_construction_sets(
                construction_sets, k_range, basis, shape_accuracy, js, center_point
            )
            print(f"[PY] Found {len(new_cells)} cells for combo {js}")
            cells.extend(new_cells)
    else:
        # Use a `Pool` to distribute work between CPU cores.
        p = Pool()
        work_func = partial(_get_cells_from_construction_sets, 
                          construction_sets, k_range, basis, shape_accuracy, 
                          center_point=center_point)
        # Flatten the results from Pool.map
        cell_lists = p.map(work_func, j_combos)
        cells = [cell for sublist in cell_lists for cell in sublist]
        p.close()

    print(f"[PY] Total cells found: {len(cells)}")
    return cells

