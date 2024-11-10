use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use dualgrid::{core, utils};
use std::collections::HashMap;
use nalgebra as na;

#[derive(Deserialize)]
struct Request {
    center_point: Option<Vec<f64>>,
    k_range: Option<i32>,
    cube_size: Option<f64>,
}

#[derive(Serialize)]
struct Response {
    statusCode: i32,
    headers: HashMap<String, String>,
    body: String,
    isBase64Encoded: bool,
    cookies: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}

async fn handler(event: LambdaEvent<Request>) -> Result<Value, Error> {
    let payload = event.payload;
    
    // Create an icosahedral basis
    let basis = utils::icosahedral_basis(false, HashMap::new());
    
    let k_range = payload.k_range.unwrap_or(2);
    
    // Convert center_point to Vector3
    let center_point = payload.center_point.map(|v| na::Vector3::new(v[0], v[1], v[2]));
    
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
    
    // Filter cells to keep only those within a cube
    let cube_size = payload.cube_size.unwrap_or(8.0);
    let filtered_cells = utils::filter_cells(
        cells,
        utils::is_point_within_cube,
        &[cube_size/2.0],  // size parameter (2 * radius)
        center_point,
        false,
        false,
        false,
    );
    
    // Convert cells to JSON response
    let body = utils::cells_to_dict(&filtered_cells, center_point);
    
    // Format response for API Gateway
    let response = Response {
        statusCode: 200,
        headers: HashMap::from([
            ("Content-Type".to_string(), "application/json".to_string()),
            ("Access-Control-Allow-Origin".to_string(), "*".to_string()),
        ]),
        body: serde_json::to_string(&body)?,
        isBase64Encoded: false,
        cookies: Vec::new(),
    };
    
    Ok(json!(response))
}
