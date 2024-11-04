use nalgebra as na;
use itertools::Itertools;
use std::collections::HashMap;
use rayon::prelude::*;

// Helper function equivalent to _get_k_combos
fn get_k_combos(k_range: i32, dimensions: usize) -> Vec<Vec<i32>> {
    let range: Vec<i32> = (-k_range..=k_range).collect();
    println!("[RS] k_combo range: {:?}", range);
    let combos = (0..dimensions)
        .map(|_| range.clone())
        .multi_cartesian_product()
        .collect();
    combos
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
        println!("[RS] Computing intersections with k_range: {}", k_range);
        
        let dimensions = self.normal.len();
        
        // Build coefficient matrix
        let mut coef_matrix = vec![self.normal.clone()];
        for other in others {
            coef_matrix.push(other.normal.clone());
        }

        let matrix_size = coef_matrix.len();
        
        // Create coefficient matrix with explicit column-major layout
        let coef_matrix = na::DMatrix::from_vec(
            matrix_size,
            matrix_size,
            coef_matrix.into_iter().flatten().collect()
        );
        println!("[RS] Coefficient matrix (before inversion):\n{:?}", coef_matrix);
        
        let coef_inv = coef_matrix.try_inverse()
            .expect("Failed to invert coefficient matrix");
        println!("[RS] Inverse matrix (after inversion):\n{:?}", coef_inv);

        // Get k combinations
        let k_combos = get_k_combos(k_range, dimensions);
        println!("[RS] k_combo range: {:?}", (-k_range..=k_range).collect::<Vec<i32>>());
        println!("[RS] Generated {} k_combos", k_combos.len());

        let base_offsets = std::iter::once(self.offset)
            .chain(others.iter().map(|o| o.offset))
            .collect::<Vec<_>>();

        let mut ds = Vec::new();
        let mut final_k_combos = Vec::new();
        for k_combo in &k_combos {
            let mut current_ds = Vec::new();
            let mut final_k_combo = Vec::new();
            
            if let Some(cp) = center_point {
                // Calculate center planes
                let center_planes = std::iter::once(dot_product(&self.normal, cp))
                    .chain(others.iter().map(|o| dot_product(&o.normal, cp)))
                    .collect::<Vec<_>>();
                
                // Calculate center indices
                let center_indices: Vec<f64> = center_planes.iter()
                    .zip(base_offsets.iter())
                    .map(|(&plane, &offset)| (plane - offset).floor())
                    .collect();

                // Apply offsets with adjusted k values
                for ((&k, &center_idx), &offset) in k_combo.iter()
                    .zip(center_indices.iter())
                    .zip(base_offsets.iter()) 
                {
                    let adjusted_k = k as f64 + center_idx;
                    current_ds.push(offset + adjusted_k);
                    final_k_combo.push((offset + adjusted_k).ceil() as i32);
                }
            } else {
                // Original logic for when no center point is provided
                for (k, &offset) in k_combo.iter().zip(base_offsets.iter()) {
                    let value = offset + *k as f64;
                    current_ds.push(value);
                    final_k_combo.push(value.ceil() as i32);
                }
            }
            
            ds.extend(current_ds);
            final_k_combos.push(final_k_combo);
        }

        println!("[RS] ds values: {:?}", ds);
        println!("[RS] First few ds values: {:?}", &ds[..std::cmp::min(ds.len(), 3)]);
        println!("[RS] ds_matrix shape: {}x{}", ds.len() / matrix_size, matrix_size);

        // Create ds as a column vector with explicit layout
        let ds_matrix = na::DMatrix::from_vec(matrix_size, 1, ds);
        
        println!("[RS] ds_matrix:\n{:?}", ds_matrix);
        println!("[RS] coef_inv:\n{:?}", coef_inv);
        
        // Try transposing coef_inv before multiplication
        let result = &coef_inv.transpose() * &ds_matrix;
        println!("[RS] Result before transpose:\n{:?}", result);
        
        let intersections: Vec<Vec<f64>> = (0..result.ncols())
            .map(|col| {
                (0..result.nrows())
                    .map(|row| result[(row, col)])
                    .collect()
            })
            .collect();

        println!("[RS] First intersection: {:?}", intersections.first().unwrap());
        
        // For each intersection, calculate its gridspace indices
        let final_k_combos: Vec<Vec<i32>> = intersections.iter()
            .map(|intersection| {
                // Use gridspace to get the indices
                let indices = basis.gridspace(intersection);
                // Only take the indices corresponding to js
                js.iter().map(|&j| indices[j]).collect()
            })
            .collect();

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
            println!("[RS] Vec {}: {:?}", j, e);
            println!("[RS] Dot product {}: {}", j, dot);
            println!("[RS] Offset {}: {}", j, self.offsets[j]);
            
            let ceil_value = (dot - self.offsets[j]).ceil() as i32;
            println!("[RS] Ceil value {}: {}", j, ceil_value);
            
            out[j] = ceil_value;  // Store the value directly
        }
        println!("[RS] Final gridspace indices: {:?}", out);
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
}

impl Cell {
    pub fn new(vertices: Vec<Vec<f64>>, indices: Vec<Vec<i32>>, intersection: Vec<f64>) -> Self {
        Self {
            verts: vertices,
            indices,
            intersection,
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
    basis: &Basis
) -> Vec<Vec<i32>> {
    println!("[RS] get_neighbours input:");
    println!("[RS]   js: {:?}", js);
    println!("[RS]   ks: {:?}", ks);
    
    // Get initial indices from gridspace
    let mut indices = basis.gridspace(intersection);
    println!("[RS] Gridspace returned indices: {:?}", indices);

    // Create a copy of the original indices
    let original_indices = indices.clone();

    // Load known indices - but only for the specific js indices
    for (index, &j) in js.iter().enumerate() {
        indices[j] = ks[index];
    }
    
    // For non-js indices, restore the original values
    for i in 0..indices.len() {
        if !js.contains(&i) {
            indices[i] = original_indices[i];
        }
    }

    println!("[RS] After loading known indices: {:?}", indices);
    println!("[RS] Initial indices: {:?}", indices);

    // Generate directions - use js.len() instead of basis.dimensions
    let directions: Vec<Vec<i32>> = (0..js.len())
        .map(|_| vec![0, 1])
        .multi_cartesian_product()
        .collect();

    // Generate deltas
    let deltas: Vec<Vec<i32>> = (0..js.len())
        .map(|i| {
            (0..basis.vecs.len())
                .map(|j| if js[i] == j { 1 } else { 0 })
                .collect()
        })
        .collect();

    println!("[RS] Deltas: {:?}", deltas);
    println!("[RS] First neighbour: {:?}", directions[0]);

    // Generate neighbours
    directions.iter()
        .map(|direction| {
            let mut neighbour = indices.clone();
            for (i, &e) in direction.iter().enumerate() {
                for (j, delta) in deltas[i].iter().enumerate() {
                    neighbour[j] += e * delta;
                }
            }
            println!("[RS] Generated neighbour: {:?}", neighbour);
            neighbour
        })
        .collect()
}

fn get_cells_from_construction_sets(
    construction_sets: &[ConstructionSet],
    k_range: i32,
    basis: &Basis,
    shape_accuracy: i32,
    js: &[usize],
    center_point: Option<&Vec<f64>>
) -> Vec<Cell> {
    println!("[RS] Getting cells for js: {:?}", js);
    
    let others: Vec<ConstructionSet> = js[1..].iter()
        .map(|&j| construction_sets[j].clone())
        .collect();

    let (intersections, k_combos) = construction_sets[js[0]]
        .get_intersections_with(k_range, &others, basis, js, center_point);
    
    println!("[RS] Found {} intersections for js {:?}", intersections.len(), js);

    intersections.iter().zip(k_combos.iter())
        .map(|(intersection, k_combo)| {
            let indices_set = get_neighbours(intersection, js, k_combo, basis);
            let vertices_set: Vec<Vec<f64>> = indices_set.iter()
                .map(|indices| basis.realspace(indices))
                .collect();

            // Add print statements for vertices
            println!("[RS] Cell vertices:");
            for (i, vertex) in vertices_set.iter().enumerate() {
                println!("[RS]   Vertex {}: {:?}", i, vertex);
            }

            Cell::new(vertices_set, indices_set, intersection.to_vec())
        })
        .collect()
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
    println!("[RS] Starting dualgrid_method");
    
    let center_point = center_point.map(|p| {
        let scaled = p.iter().map(|x| x / 2.0).collect::<Vec<f64>>();
        println!("[RS] Adjusted center_point: {:?}", scaled);
        scaled
    });

    let construction_sets = construction_sets_from_basis(basis);
    println!("[RS] Created {} construction sets", construction_sets.len());
    
    let j_combos: Vec<Vec<usize>> = (0..construction_sets.len())
        .combinations(basis.dimensions)
        .collect();
    println!("[RS] Generated {} j_combos", j_combos.len());

    if single_threaded {
        println!("[RS] Running single-threaded");
        j_combos.iter()
            .flat_map(|js| {
                let cells = get_cells_from_construction_sets(
                    &construction_sets,
                    k_range,
                    basis,
                    shape_accuracy,
                    js,
                    center_point.as_ref()
                );
                println!("[RS] Found {} cells for combo {:?}", cells.len(), js);
                cells
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
