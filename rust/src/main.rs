use dualgrid::{core, utils};
use std::collections::HashMap;
use nalgebra as na;

fn main() {
    // Create an icosahedral basis
    let basis = utils::icosahedral_basis(false, HashMap::new());
    println!("Basis offsets: {:?}", basis.offsets());
    
    let k_range = 2;
    
    // Convert center_point to Vector3
    let center_point = Some(na::Vector3::new(10.0, -10.0, 10.0));
    
    // Get cells
    let cells: Vec<utils::Cell> = core::dualgrid_method(
        &basis,
        k_range,
        center_point.as_ref().map(|v| vec![v[0], v[1], v[2]]),
        4,
        true
    )
    .into_iter()
    .map(|c| utils::Cell {
        verts: c.verts().iter()
            .map(|v| na::Vector3::new(v[0], v[1], v[2]))
            .collect(),
        indices: c.indices().iter()
            .flatten()
            .cloned()
            .collect(),
        intersection: na::Vector3::new(
            c.intersection()[0],
            c.intersection()[1],
            c.intersection()[2]
        ),
        filled: true,
    })
    .collect();

    println!("number of cells: {}", cells.len());
    println!("Cells found.");
    
    // Filter cells to keep only those within a cube of size 8 (radius 4)
    let filtered_cells = utils::filter_cells(
        cells,
        utils::is_point_within_cube,
        &[4.0],  // size parameter (2 * radius)
        center_point,
        false,   // fast_filter
        false,   // filter_indices
        false,   // invert_filter
    );
    
    println!("number of cells after filtering: {}", filtered_cells.len());
    
    // Export filtered cells to JSON
    utils::export_cells_to_json(
        &filtered_cells,
        "../cells_out_rs.json",
        center_point
    ).expect("Failed to export cells to JSON");
    
    println!("DONE :)");
}
