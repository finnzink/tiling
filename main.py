import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def main():
    # Make a Basis object. There are some presets available in the `utils`.
    #basis = dg.utils.surface_with_n_rotsym(7, centred=True)   # 2D structure with 7-fold rotational symmetry
    # basis = dg.utils.penrose_basis()          # Section of Penrose tiling.
    basis = dg.utils.icosahedral_basis()      # 3D quasicrystalline structure
    #basis = dg.utils.n_dimensional_cubic_basis(4) # 4D cubic structure
    
    print("OFFSETS:", basis.offsets)
    G = None
    
    # Set the k range, i.e the number of construction planes used in generating the vertices.
    # In 2D this corresponds to having line sets with lines of index -1, 0, 1 for a k range of 2 for example.
    # Higher k_range -> more vertices generated.
    # The results will later be filtered to remove outliers.
    k_range = 11
    
    # NOTE: It is advised to use a smaller k_range for 3D+ structures as
    # matplotlib starts to struggle with large numbers of shapes. I have
    # done an if statement here to change it for 3D+.
    if basis.dimensions > 2:
        k_range = 2
    
    # Run the algorithm. k_ranges sets the number of construction planes used in the method.
    # The function outputs a list of Cell objects.
    cells = dg.dualgrid_method(basis, k_range)
    print("Cells found.\nFiltering...")
    # Filter the output cells by some function. Pre-defined ones are: is_point_within_cube, is_point_within_radius, elements_are_below, contains_value. Each one can be toggled
    # to use the real space positions of vertices, or their indices in grid space.
    
    
    # To filter by highest index allowed (good for 2D, odd N-fold tilings):
    # cells = dg.utils.filter_cells(cells, filter=dg.utils.elements_are_below, filter_args=[max(k_range-1, 0)], filter_indices=True, invert_filter=False)
    
    # To filter out a radius of R:
    R = 11
    if basis.dimensions != 2:
        R = 2 # Reduce for 3D+ to reduce lag
    
    cells = dg.utils.filter_cells(cells, filter=dg.utils.is_point_within_radius, filter_args=[R])
    
    print("Cells filtered.")
    
    dg.utils.export_cells_to_json(cells, "cells_out.json")
    print("DONE :)")


# NOTE: This is needed in your program if using multithreading on Windows.
if __name__ == "__main__":
    main()
