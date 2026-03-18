//! Generate OpenAPI JSON spec to stdout.
//!
//! Usage: `cargo run -p cvx-api --bin gen_openapi > docs/public/openapi.json`

use cvx_api::openapi::ApiDoc;
use utoipa::OpenApi;

fn main() {
    println!("{}", ApiDoc::openapi().to_pretty_json().unwrap());
}
