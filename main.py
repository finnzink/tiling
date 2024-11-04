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
    
    print("Cells found.")
    
    dg.utils.export_cells_to_json(cells, "../cells_out_py.json", center_point=center_point)
    print("DONE :)")


# NOTE: This is needed in your program if using multithreading on Windows.
if __name__ == "__main__":
    main()
