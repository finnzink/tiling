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
        false
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
    println!("Cells found.\nFiltering...");
    
    // Filter cells within a cube
    let r = 2.0;
    let filtered_cells = utils::filter_cells(
        cells,
        utils::is_point_within_cube,
        &[r * 2.0],
        center_point,
        false,
        false,
        false
    );
    
    // Export cells to JSON
    utils::export_cells_to_json(
        &filtered_cells,
        "../cells_out.json",
        center_point
    ).expect("Failed to export cells to JSON");
    
    println!("DONE :)");
}
