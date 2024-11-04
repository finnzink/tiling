use nalgebra as na;
use rand::prelude::*;
use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::f64::consts::PI;
use serde_json::{json, Value};
use std::collections::HashSet;
use crate::core::Basis;

// Type aliases to make the code more readable
type Vector3 = na::Vector3<f64>;
type Point3 = na::Point3<f64>;

/// Gives offsets that give a tiling around centre of rotation.
pub fn offsets_fixed_around_centre(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

/// Generates offsets for use in the dualgrid method.
pub fn generate_offsets(
    num: usize,
    random: bool,
    below_one: bool,
    sum_zero: bool,
    centred: bool,
) -> Vec<f64> {
    if centred {
        return offsets_fixed_around_centre(num);
    }

    let mut offsets = if random {
        let mut rng = StdRng::from_entropy();
        (0..num).map(|_| rng.gen::<f64>()).collect()
    } else {
        // Fixed offsets for deterministic results
        vec![0.27, 0.37, 0.47, 0.57, 0.67, 0.77][..num].to_vec()
    };

    if below_one {
        offsets.iter_mut().for_each(|x| *x /= num as f64);
    }

    if sum_zero {
        let sum: f64 = offsets[..offsets.len()-1].iter().sum();
        *offsets.last_mut().unwrap() = -sum;
    }

    offsets
}

/// Creates an icosahedral basis
pub fn icosahedral_basis(random_offsets: bool, kwargs: HashMap<String, bool>) -> Basis {
    let offsets = generate_offsets(
        6,
        random_offsets,
        *kwargs.get("below_one").unwrap_or(&false),
        *kwargs.get("sum_zero").unwrap_or(&false),
        *kwargs.get("centred").unwrap_or(&false),
    );

    let sqrt5 = 5.0_f64.sqrt();
    let mut icos: Vec<Vec<f64>> = (0..5)
        .map(|n| {
            let angle = 2.0 * PI * n as f64 / 5.0;
            vec![
                (2.0 / sqrt5) * angle.cos(),
                (2.0 / sqrt5) * angle.sin(),
                1.0 / sqrt5
            ]
        })
        .collect();
    
    icos.push(vec![0.0, 0.0, 1.0]);
    
    // Debug print
    println!("Rust icosahedral basis:");
    for (i, vec) in icos.iter().enumerate() {
        println!("v{}: [{:.6}, {:.6}, {:.6}]", i, vec[0], vec[1], vec[2]);
    }
    println!("Offsets: {:?}", offsets);
    
    Basis::new(icos, offsets)
}

/// Filter function to check if a point is within a radius
pub fn is_point_within_radius(r: &Vector3, filter_centre: &Vector3, radius: f64) -> bool {
    (r - filter_centre).norm() < radius
}

/// Filter function to check if a point is within a cube
pub fn is_point_within_cube(r: &Vector3, filter_centre: &Vector3, size: f64) -> bool {
    let diff = r - filter_centre;
    let size_div2 = size / 2.0;
    
    !diff.iter().any(|&d| d.abs() > size_div2)
}

/// Checks for value within indices
pub fn contains_value(r: &Vector3, _filter_centre: &Vector3, value: f64) -> bool {
    r.iter().any(|&x| (x - value).abs() < f64::EPSILON)
}

/// Checks if all elements are below the given value
pub fn elements_are_below(r: &Vector3, _filter_centre: &Vector3, value: f64) -> bool {
    r.iter().all(|&x| x < value)
}

#[derive(Debug, Clone)]
pub struct Cell {
    pub verts: Vec<Vector3>,
    pub indices: Vec<i32>,
    pub intersection: Vector3,
    pub filled: bool,
}

// Add a serializable version of Cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCell {
    pub verts: Vec<Vec<f64>>,
    pub indices: Vec<Vec<i32>>,
    pub intersection: Vec<f64>,
    pub filled: bool,
}

impl Cell {
    pub fn is_in_filter<F>(&self, filter: F, filter_centre: &Vector3, filter_args: &[f64], fast: bool, filter_indices: bool, invert_filter: bool) -> bool 
    where
        F: Fn(&Vector3, &Vector3, f64) -> bool
    {
        let sum = if filter_indices {
            self.intersection.clone()
        } else if fast {
            self.verts[0].clone()
        } else {
            self.verts.iter().sum::<Vector3>() / self.verts.len() as f64
        };

        let result = filter_args.iter().any(|&arg| filter(&sum, filter_centre, arg));
        if invert_filter { !result } else { result }
    }

    // Add a method to convert to serializable form
    pub fn to_serializable(&self) -> SerializableCell {
        // Group indices into chunks of 6 (since each vertex has 6 coordinates)
        let indices: Vec<Vec<i32>> = self.indices
            .chunks(6)
            .map(|chunk| chunk.to_vec())
            .collect();

        SerializableCell {
            verts: self.verts.iter()
                .map(|v| vec![v[0], v[1], v[2]])
                .collect(),
            indices,
            intersection: vec![
                self.intersection[0],
                self.intersection[1],
                self.intersection[2]
            ],
            filled: self.filled,
        }
    }
}

/// Get the center of interest from a collection of cells
pub fn get_centre_of_interest(cells: &[Cell]) -> Vector3 {
    let mut all_verts = Vec::new();
    for cell in cells {
        all_verts.extend(cell.verts.clone());
    }
    
    all_verts.iter().sum::<Vector3>() / all_verts.len() as f64
}

/// Filter cells based on given criteria
pub fn filter_cells<F>(
    cells: Vec<Cell>,
    filter: F,
    filter_args: &[f64],
    filter_centre: Option<Vector3>,
    fast_filter: bool,
    filter_indices: bool,
    invert_filter: bool,
) -> Vec<Cell>
where
    F: Fn(&Vector3, &Vector3, f64) -> bool
{
    let centre = filter_centre.unwrap_or_else(|| {
        println!("FINDING COI");
        let coi = get_centre_of_interest(&cells);
        println!("COI: {:?}", coi);
        coi
    });

    cells
        .into_iter()
        .filter(|c| c.is_in_filter(&filter, &centre, filter_args, fast_filter, filter_indices, invert_filter))
        .collect()
}

const FACE_INDICES: [[usize; 4]; 6] = [
    [0, 2, 3, 1], // front
    [0, 1, 5, 4], // right
    [5, 7, 6, 4], // back
    [2, 6, 7, 3], // left
    [0, 4, 6, 2], // top
    [3, 7, 5, 1], // bottom
];

fn calculate_face_normal(vertices: &[Vector3]) -> Vector3 {
    let v0 = &vertices[0];
    let v1 = &vertices[1];
    let v2 = &vertices[2];
    
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let normal = edge1.cross(&edge2);
    normal.normalize()
}

fn face_needs_flip(vertices: &[Vector3], cell_center: &Vector3) -> bool {
    let normal = calculate_face_normal(vertices);
    let face_center = vertices.iter().sum::<Vector3>() / vertices.len() as f64;
    let direction_from_center = &face_center - cell_center;
    normal.dot(&direction_from_center) < 0.0
}

fn analyze_and_adjust_triangulation(
    vertices: &[Vector3],
    face_indices: &[usize],
    cell_center: &Vector3
) -> Vec<usize> {
    let v0 = &vertices[0];
    let v1 = &vertices[1];
    let v2 = &vertices[2];
    let v3 = &vertices[3];
    
    let diag1 = (v2 - v0).norm();
    let diag2 = (v3 - v1).norm();
    
    let mut adjusted_indices = if diag1 > diag2 {
        vec![
            face_indices[1],
            face_indices[2],
            face_indices[3],
            face_indices[0],
        ]
    } else {
        face_indices.to_vec()
    };
    
    if face_needs_flip(&vertices, cell_center) {
        adjusted_indices = vec![
            adjusted_indices[0],
            adjusted_indices[3],
            adjusted_indices[2],
            adjusted_indices[1],
        ];
    }
    
    adjusted_indices
}

#[derive(Serialize, Deserialize)]
struct TriangleData {
    center: Vec<f64>,
    cells: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct CellData {
    vertices: Vec<Vec<f64>>,
    indices: Vec<Vec<i32>>,
    intersection: Vec<f64>,
    filled: bool,
    face_indices: Vec<Vec<usize>>,
}

pub fn cells_to_dict(cells: &[Cell], center_point: Option<Vector3>) -> Value {
    // Create a mutable map for cells and triangles
    let mut cells_map = serde_json::Map::new();
    let mut triangles_map = serde_json::Map::new();

    // Process cells first
    for cell in cells {
        let cell_uuid = Uuid::new_v4().to_string();
        let cell_center = cell.verts.iter().sum::<Vector3>() / cell.verts.len() as f64;
        
        let serializable_cell = cell.to_serializable();
        let mut adjusted_face_indices = Vec::new();
        
        for face_indices in FACE_INDICES.iter() {
            let face_vertices: Vec<Vector3> = face_indices
                .iter()
                .map(|&idx| cell.verts[idx].clone())
                .collect();
                
            let adjusted_indices = analyze_and_adjust_triangulation(
                &face_vertices,
                face_indices,
                &cell_center
            );
            
            adjusted_face_indices.push(adjusted_indices);
        }

        // Convert cell data to serializable format
        let cell_data = CellData {
            vertices: serializable_cell.verts.clone(),
            indices: serializable_cell.indices.clone(),
            intersection: serializable_cell.intersection.clone(),
            filled: serializable_cell.filled,
            face_indices: adjusted_face_indices.clone(),
        };

        cells_map.insert(cell_uuid.clone(), json!(cell_data));

        // Process triangles
        for face_indices in adjusted_face_indices {
            let triangles = [
                [face_indices[0], face_indices[1], face_indices[2]],
                [face_indices[0], face_indices[2], face_indices[3]]
            ];
            
            for triangle in triangles.iter() {
                let tri_verts: Vec<Vector3> = triangle
                    .iter()
                    .map(|&idx| cell.verts[idx].clone())
                    .collect();
                
                let tri_center = tri_verts.iter().sum::<Vector3>() / 3.0;
                let tri_center_key = format!(
                    "{:.2},{:.2},{:.2}",
                    tri_center[0], tri_center[1], tri_center[2]
                );

                if let Some(existing_triangle) = triangles_map.get_mut(&tri_center_key) {
                    let cells = existing_triangle.as_object_mut().unwrap()
                        .get_mut("cells").unwrap()
                        .as_array_mut().unwrap();
                    if !cells.contains(&json!(cell_uuid)) {
                        cells.push(json!(cell_uuid.clone()));
                    }
                } else {
                    triangles_map.insert(
                        tri_center_key,
                        json!({
                            "center": vec![tri_center[0], tri_center[1], tri_center[2]],
                            "cells": vec![cell_uuid.clone()],
                        }),
                    );
                }
            }
        }
    }

    // Create the final result
    json!({
        "cells": cells_map,
        "triangles": triangles_map,
        "center_of_interest": center_point.map(|v| vec![v[0], v[1], v[2]]),
    })
}

pub fn export_cells_to_json(cells: &[Cell], filepath: &str, center_point: Option<Vector3>) -> std::io::Result<()> {
    let data = cells_to_dict(cells, center_point);
    let file = std::fs::File::create(filepath)?;
    serde_json::to_writer_pretty(file, &data)?;
    Ok(())
}
