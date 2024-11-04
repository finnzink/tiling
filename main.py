import dualgrid as dg

def main():
    # Make a Basis object. There are some presets available in the `utils`.
    basis = dg.utils.icosahedral_basis(random_offsets=False)
    print("OFFSETS:", basis.offsets)
    
    k_range = 0
    
    # Add center_point parameter to dualgrid_method call
    center_point = [10.0, -10.0, 10.0]
    cells = dg.dualgrid_method(basis, k_range, center_point=center_point, single_threaded=True)
    print("number of cells:", len(cells))
    
    print("Cells found.\nFiltering...")
    # Filter the output cells by some function. Pre-defined ones are: is_point_within_cube, is_point_within_radius, elements_are_below, contains_value. Each one can be toggled
    # to use the real space positions of vertices, or their indices in grid space.
    
    R = 2 # Reduce for 3D+ to reduce lag
    cells = dg.utils.filter_cells(cells, filter=dg.utils.is_point_within_cube, filter_args=[R*2], filter_centre=center_point)
    
    # print("Number of cells filtered:", len(cells))
    
    dg.utils.export_cells_to_json(cells, "../cells_out_py.json", center_point=center_point)
    print("DONE :)")


# NOTE: This is needed in your program if using multithreading on Windows.
if __name__ == "__main__":
    main()
