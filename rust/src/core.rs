use nalgebra as na;
use itertools::Itertools;
use std::collections::HashMap;
use rayon::prelude::*;

// Helper function equivalent to _get_k_combos
fn get_k_combos(k_range: i32, dimensions: usize, center_point: Option<&Vec<f64>>) -> Vec<Vec<i32>> {
    let combos: Vec<Vec<i32>> = match center_point {
        Some(_) => {
            // Centered around provided point
            let range: Vec<i32> = (-k_range..=k_range).collect();
            (0..dimensions)
                .map(|_| range.clone())
                .multi_cartesian_product()
                .collect()
        }
        None => {
            // Original behavior centered around origin
            (0..dimensions)
                .map(|_| ((1-k_range)..=k_range).collect::<Vec<i32>>())
                .multi_cartesian_product()
                .collect()
        }
    };
    println!("Rust k_combos length: {}", combos.len());  // Debug print
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
        center_point: Option<&Vec<f64>>
    ) -> (Vec<Vec<f64>>, Vec<Vec<i32>>) {
        let dimensions = self.normal.len();
        
        // Build coefficient matrix
        let mut coef_matrix = vec![self.normal.clone()];
        for other in others {
            coef_matrix.push(other.normal.clone());
        }

        let matrix_size = coef_matrix.len();
        let coef_matrix = na::DMatrix::from_vec(
            matrix_size,
            matrix_size,
            coef_matrix.into_iter().flatten().collect()
        );

        // Clone coef_matrix before inverting
        let coef_inv = coef_matrix.clone().try_inverse().unwrap();

        // Now you can still use coef_matrix
        // println!("Coefficient Matrix:\n{}", coef_matrix);
        // println!("Inverse Coefficient Matrix:\n{}", coef_inv);

        let mut k_combos = get_k_combos(k_range, dimensions, center_point);

        // Get base offsets
        let mut base_offsets = vec![self.offset];
        base_offsets.extend(others.iter().map(|o| o.offset));

        if let Some(center_point) = center_point {
            // Calculate plane indices for center point
            let mut center_planes = vec![dot_product(center_point, &self.normal)];
            center_planes.extend(others.iter().map(|o| dot_product(center_point, &o.normal)));

            // Round to get nearest plane indices
            let center_indices: Vec<f64> = center_planes.iter()
                .zip(base_offsets.iter())
                .map(|(plane, offset)| (plane - offset).floor())
                .collect();

            // Shift k_combos to be centered around these indices
            for combo in k_combos.iter_mut() {
                for (i, idx) in center_indices.iter().enumerate() {
                    combo[i] += *idx as i32;
                }
            }
        }

        // Calculate intersections
        let ds: Vec<f64> = k_combos.iter()
            .flat_map(|combo| {
                combo.iter()
                    .zip(base_offsets.iter())
                    .map(|(k, offset)| *k as f64 + offset)
                    .collect::<Vec<f64>>()
            })
            .collect();

        let ds_matrix = na::DMatrix::from_vec(ds.len() / matrix_size, matrix_size, ds);
        let intersections_matrix = &coef_inv * ds_matrix.transpose();
        
        let intersections: Vec<Vec<f64>> = (0..intersections_matrix.ncols())
            .map(|col| {
                (0..intersections_matrix.nrows())
                    .map(|row| intersections_matrix[(row, col)])
                    .collect()
            })
            .collect();

        // println!("k_combos after adjustment:\n{:?}", k_combos);
        // println!("Intersections:\n{:?}", intersections);

        (intersections, k_combos)
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
        for (j, e) in self.vecs.iter().enumerate() {
            out[j] = (dot_product(r, e) - self.offsets[j]).ceil() as i32;
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
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
    // Generate all possible combinations of 0s and 1s for the given dimensions
    let directions: Vec<Vec<i32>> = (0..basis.dimensions)
        .map(|_| vec![0, 1])
        .multi_cartesian_product()
        .collect();

    let mut indices = basis.gridspace(intersection);

    // Load known indices
    for (index, &j) in js.iter().enumerate() {
        indices[j] = ks[index];
    }

    // Generate deltas (Kronecker delta function implementation)
    let deltas: Vec<Vec<i32>> = (0..basis.dimensions)
        .map(|i| {
            (0..basis.vecs.len())
                .map(|j| if js[i] == j { 1 } else { 0 })
                .collect()
        })
        .collect();

    // Generate neighbours using equation 4.5 from de Bruijn's paper
    directions.iter()
        .map(|direction| {
            let mut neighbour = indices.clone();
            for (i, &e) in direction.iter().enumerate() {
                for (j, delta) in deltas[i].iter().enumerate() {
                    neighbour[j] += e * delta;
                }
            }
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
    let others: Vec<ConstructionSet> = js[1..].iter()
        .map(|&j| construction_sets[j].clone())
        .collect();

    let (intersections, k_combos) = construction_sets[js[0]]
        .get_intersections_with(k_range, &others, center_point);

    intersections.iter().zip(k_combos.iter())
        .map(|(intersection, k_combo)| {
            let indices_set = get_neighbours(intersection, js, k_combo, basis);
            let vertices_set: Vec<Vec<f64>> = indices_set.iter()
                .map(|indices| basis.realspace(indices))
                .collect();

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
    // Scale center point once at the beginning
    let center_point = center_point.map(|p| {
        let scaled = p.iter().map(|x| x / 2.0).collect::<Vec<f64>>();
        println!("Rust adjusted center_point: {:?}", scaled);
        scaled
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
