use nalgebra as na;
use itertools::Itertools;
use std::collections::HashMap;
use rayon::prelude::*;

// Helper function equivalent to _get_k_combos
fn get_k_combos(k_range: i32, dimensions: usize) -> Vec<Vec<i32>> {
    let range: Vec<i32> = (-k_range..=k_range).collect();
    (0..dimensions)
        .map(|_| range.clone())
        .multi_cartesian_product()
        .collect()
}

// ConstructionSet equivalent
#[derive(Clone)]
pub struct ConstructionSet {
    normal: Vec<f64>,
    offset: f64,
}

impl ConstructionSet {
    pub fn new(normal: Vec<f64>, offset: f64) -> Self {
        Self { normal, offset }
    }

    pub fn get_intersections_with(
        &self,
        k_range: i32,
        others: &[ConstructionSet],
        basis: &Basis,
        js: &[usize],
        center_point: Option<&Vec<f64>>
    ) -> (Vec<Vec<f64>>, Vec<Vec<i32>>) {
        println!("\n[RS] get_intersections_with details:");
        println!("[RS] k_range: {}", k_range);
        println!("[RS] self.normal: {:?}", self.normal);
        println!("[RS] self.offset: {:?}", self.offset);
        println!("[RS] others normals: {:?}", others.iter().map(|o| &o.normal).collect::<Vec<_>>());
        println!("[RS] others offsets: {:?}", others.iter().map(|o| o.offset).collect::<Vec<_>>());
        
        let dimensions = self.normal.len();
        
        // Create coefficient matrix as a 2D array first
        let mut coef_matrix_data: Vec<f64> = Vec::with_capacity(dimensions * dimensions);
        coef_matrix_data.extend(&self.normal);
        for other in others {
            coef_matrix_data.extend(&other.normal);
        }
        
        let coef_matrix: na::DMatrix<f64> = na::DMatrix::from_row_slice(
            dimensions,
            dimensions,
            &coef_matrix_data
        );

        // Check determinant
        if coef_matrix.determinant() == 0.0 {
            println!("WARNING: Unit vectors form singular matrices.");
            return (vec![], vec![]);
        }

        let coef_inv: na::DMatrix<f64> = coef_matrix.try_inverse()
            .expect("Failed to invert coefficient matrix");

        // Get k combinations
        let k_combos = get_k_combos(k_range, dimensions);

        // Get the base offsets
        let mut base_offsets = vec![self.offset];
        base_offsets.extend(others.iter().map(|o| o.offset));

        let mut intersections = Vec::new();
        let mut final_k_combos = Vec::new();

        if let Some(cp) = center_point {
            // Calculate the plane indices for the center point
            let center_planes: Vec<f64> = std::iter::once(&self.normal)
                .chain(others.iter().map(|o| &o.normal))
                .map(|normal| {
                    normal.iter().zip(cp.iter()).map(|(n, p)| n * p).sum()
                })
                .collect();

            // Calculate center indices
            let center_indices: Vec<f64> = center_planes.iter()
                .zip(base_offsets.iter())
                .map(|(&plane, &offset)| (plane - offset).floor())
                .collect();

            // Process each k_combo with center point adjustment
            for k_combo in k_combos {
                let ds: Vec<f64> = k_combo.iter()
                    .zip(center_indices.iter())
                    .zip(base_offsets.iter())
                    .map(|((&k, &center_idx), &offset)| {
                        offset + (k as f64 + center_idx)
                    })
                    .collect();

                let adjusted_k_combo: Vec<i32> = k_combo.iter()
                    .zip(center_indices.iter())
                    .map(|(&k, &center_idx)| k + center_idx as i32)
                    .collect();

                let ds_matrix: na::DMatrix<f64> = na::DMatrix::from_column_slice(dimensions, 1, &ds);
                let result: na::DMatrix<f64> = &coef_inv * &ds_matrix;
                
                let intersection: Vec<f64> = (0..dimensions)
                    .map(|i| result[(i, 0)])
                    .collect();

                intersections.push(intersection);
                final_k_combos.push(adjusted_k_combo);
            }
        } else {
            // Process each k_combo without center point adjustment
            for k_combo in k_combos {
                let ds: Vec<f64> = k_combo.iter()
                    .zip(base_offsets.iter())
                    .map(|(&k, &offset)| offset + k as f64)
                    .collect();

                let ds_matrix: na::DMatrix<f64> = na::DMatrix::from_column_slice(dimensions, 1, &ds);
                let result: na::DMatrix<f64> = &coef_inv * &ds_matrix;
                
                let intersection: Vec<f64> = (0..dimensions)
                    .map(|i| result[(i, 0)])
                    .collect();

                intersections.push(intersection);
                final_k_combos.push(k_combo.to_vec());
            }
        }

        (intersections, final_k_combos)
    }
}

// Basis equivalent
pub struct Basis {
    vecs: Vec<Vec<f64>>,
    dimensions: usize,
    offsets: Vec<f64>,
}

impl Basis {
    pub fn new(vecs: Vec<Vec<f64>>, offsets: Vec<f64>) -> Self {
        let dimensions = vecs[0].len();
        Self { vecs, dimensions, offsets }
    }

    pub fn realspace(&self, indices: &[i32]) -> Vec<f64> {
        let mut out = vec![0.0; self.dimensions];
        for (j, e) in self.vecs.iter().enumerate() {
            for (i, val) in e.iter().enumerate() {
                out[i] += val * indices[j] as f64;
            }
        }
        out
    }

    pub fn gridspace(&self, r: &[f64]) -> Vec<i32> {
        let mut out = vec![0; self.vecs.len()];
        for j in 0..self.vecs.len() {
            let e = &self.vecs[j];
            let dot = dot_product(r, e);
            let ceil_value = (dot - self.offsets[j]).ceil() as i32;
            out[j] = ceil_value;
        }
        out
    }

    pub fn get_possible_cells(&self, decimals: i32) -> HashMap<String, Vec<Vec<usize>>> {
        let mut shapes: HashMap<String, Vec<Vec<usize>>> = HashMap::new();

        for inds in (0..self.vecs.len()).combinations(self.dimensions) {
            // Create matrix from selected basis vectors
            let matrix_data: Vec<f64> = inds.iter()
                .flat_map(|&j| self.vecs[j].clone())
                .collect();
            
            let matrix = na::DMatrix::from_vec(
                self.dimensions,
                self.dimensions,
                matrix_data
            );

            let vol = matrix.determinant().abs();
            
            if vol != 0.0 {
                // Round to specified decimal places and convert to string
                let rounded_vol = format!("{:.1$}", vol, decimals as usize);
                shapes.entry(rounded_vol)
                    .or_insert_with(Vec::new)
                    .push(inds);
            }
        }

        shapes
    }

    pub fn offsets(&self) -> &Vec<f64> {
        &self.offsets
    }
}

// Helper function for dot product
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

// Cell equivalent
#[derive(Clone)]
pub struct Cell {
    verts: Vec<Vec<f64>>,
    indices: Vec<Vec<i32>>,
    intersection: Vec<f64>,
    filled: bool,
}

impl Cell {
    pub fn new(vertices: Vec<Vec<f64>>, indices: Vec<Vec<i32>>, intersection: Vec<f64>, is_first: bool) -> Self {
        Self {
            verts: vertices,
            indices,
            intersection,
            filled: is_first,
        }
    }

    pub fn verts(&self) -> &Vec<Vec<f64>> {
        &self.verts
    }

    pub fn indices(&self) -> &Vec<Vec<i32>> {
        &self.indices
    }

    pub fn intersection(&self) -> &Vec<f64> {
        &self.intersection
    }
}

fn get_neighbours(
    intersection: &[f64],
    js: &[usize],
    ks: &[i32],
    basis: &Basis,
    cell_index: usize
) -> Vec<Vec<i32>> {
    if cell_index == 2 {
        println!("\n[RS] Cell 2 neighbour calculation:");
        println!("[RS] Input intersection: {:?}", intersection);
        println!("[RS] Input js: {:?}", js);
        println!("[RS] Input ks: {:?}", ks);
        
        let mut indices = basis.gridspace(intersection);
        println!("[RS] Initial gridspace indices: {:?}", indices);
        
        for (index, &j) in js.iter().enumerate() {
            indices[j] = ks[index];
        }
        println!("[RS] After setting js indices: {:?}", indices);
        
        let directions: Vec<Vec<i32>> = (0..basis.dimensions)
            .map(|_| vec![0, 1])
            .multi_cartesian_product()
            .collect();
        println!("[RS] Directions: {:?}", directions);
        
        let deltas: Vec<Vec<i32>> = (0..basis.dimensions)
            .map(|i| {
                (0..basis.vecs.len())
                    .map(|j| if js.contains(&j) && js.iter().position(|&x| x == j).unwrap() == i { 1 } else { 0 })
                    .collect()
            })
            .collect();
        println!("[RS] Deltas: {:?}", deltas);
        
        let neighbours = directions.iter()
            .map(|direction| {
                let mut neighbour = indices.clone();
                for (i, &e) in direction.iter().enumerate() {
                    for (j, delta) in deltas[i].iter().enumerate() {
                        neighbour[j] += e * delta;
                    }
                }
                neighbour
            })
            .collect();
        println!("[RS] Final neighbours: {:?}", neighbours);
        
        neighbours
    } else {
        let should_log = cell_index == 1 || cell_index == 2;
        let mut indices = basis.gridspace(intersection);
        
        if should_log {
            println!("\n[RS] Cell {}: Initial gridspace indices: {:?}", cell_index, indices);
        }
        
        let original_indices = indices.clone();
        
        for (index, &j) in js.iter().enumerate() {
            indices[j] = ks[index];
        }
        if should_log {
            println!("[RS] Cell {}: After setting js indices: {:?}", cell_index, indices);
        }
        
        for i in 0..indices.len() {
            if !js.contains(&i) {
                indices[i] = original_indices[i];
            }
        }
        if should_log {
            println!("[RS] Cell {}: After restoring non-js indices: {:?}", cell_index, indices);
        }

        let directions: Vec<Vec<i32>> = (0..basis.dimensions)
            .map(|_| vec![0, 1])
            .multi_cartesian_product()
            .collect();
        if should_log {
            println!("[RS] Cell {}: Directions: {:?}", cell_index, directions);
        }

        let deltas: Vec<Vec<i32>> = (0..basis.dimensions)
            .map(|i| {
                (0..basis.vecs.len())
                    .map(|j| if js.contains(&j) && js.iter().position(|&x| x == j).unwrap() == i { 1 } else { 0 })
                    .collect()
            })
            .collect();
        if should_log {
            println!("[RS] Cell {}: Deltas: {:?}", cell_index, deltas);
        }

        let neighbours = directions.iter()
            .map(|direction| {
                let mut neighbour = indices.clone();
                for (i, &e) in direction.iter().enumerate() {
                    for (j, delta) in deltas[i].iter().enumerate() {
                        neighbour[j] += e * delta;
                    }
                }
                neighbour
            })
            .collect();
        if should_log {
            println!("[RS] Cell {}: Final neighbours: {:?}", cell_index, neighbours);
        }
        
        neighbours
    }
}

fn get_cells_from_construction_sets(
    construction_sets: &[ConstructionSet],
    k_range: i32,
    basis: &Basis,
    shape_accuracy: i32,
    js: &[usize],
    center_point: Option<&Vec<f64>>
) -> Vec<Cell> {
    let others: Vec<ConstructionSet> = js[1..].iter()
        .map(|&j| construction_sets[j].clone())
        .collect();

    // Add debug info before intersection calculation
    println!("\n[RS] Calculating intersections for js: {:?}", js);
    println!("[RS] First normal: {:?}", construction_sets[js[0]].normal);
    println!("[RS] Other normals: {:?}", others.iter().map(|cs| &cs.normal).collect::<Vec<_>>());

    let (intersections, k_combos) = construction_sets[js[0]]
        .get_intersections_with(k_range, &others, basis, js, center_point);
    
    // Add debug info after intersection calculation
    println!("[RS] First two intersections: {:?}", &intersections[..2.min(intersections.len())]);
    println!("[RS] First two k_combos: {:?}", &k_combos[..2.min(k_combos.len())]);

    let mut cells = Vec::new();
    
    for (i, (intersection, k_combo)) in intersections.iter().zip(k_combos.iter()).enumerate() {
        let indices_set = get_neighbours(intersection, js, k_combo, basis, i);
        let vertices_set: Vec<Vec<f64>> = indices_set.iter()
            .map(|indices| basis.realspace(indices))
            .collect();

        let is_first = cells.is_empty(); // Only true for the first cell
        let cell = Cell::new(vertices_set.clone(), indices_set.clone(), intersection.to_vec(), is_first);
        cells.push(cell);

        // Only log for cells 1 and 2
        if cells.len() == 2 || cells.len() == 3 {
            println!("\n[RS] Processing cell {}", cells.len() - 1);
            println!("[RS] js: {:?}", js);
            println!("[RS] Intersection: {:?}", intersection);
            println!("[RS] k_combo: {:?}", k_combo);
            println!("[RS] Indices set: {:?}", indices_set);
            println!("[RS] Vertices set: {:?}", vertices_set);
        }
    }

    cells
}

pub fn construction_sets_from_basis(basis: &Basis) -> Vec<ConstructionSet> {
    basis.vecs.iter().enumerate()
        .map(|(i, e)| ConstructionSet::new(e.clone(), basis.offsets[i]))
        .collect()
}

pub fn dualgrid_method(
    basis: &Basis,
    k_range: i32,
    center_point: Option<Vec<f64>>,
    shape_accuracy: i32,
    single_threaded: bool
) -> Vec<Cell> {
    let center_point = center_point.map(|p| {
        p.iter().map(|x| x / 2.0).collect::<Vec<f64>>()
    });

    let construction_sets = construction_sets_from_basis(basis);
    let j_combos: Vec<Vec<usize>> = (0..construction_sets.len())
        .combinations(basis.dimensions)
        .collect();

    if single_threaded {
        j_combos.iter()
            .flat_map(|js| {
                get_cells_from_construction_sets(
                    &construction_sets,
                    k_range,
                    basis,
                    shape_accuracy,
                    js,
                    center_point.as_ref()
                )
            })
            .collect()
    } else {
        j_combos.par_iter()
            .flat_map(|js| {
                get_cells_from_construction_sets(
                    &construction_sets,
                    k_range,
                    basis,
                    shape_accuracy,
                    js,
                    center_point.as_ref()
                )
            })
            .collect()
    }
}
