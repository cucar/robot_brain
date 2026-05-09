/// N-API bindings for brain-core.
///
/// This crate will expose the Brain struct as a JavaScript class
/// via napi-rs. Placeholder until Phase 1.9.
use napi_derive::napi;

#[napi]
pub fn version() -> String {
    "brain-core 0.1.0".to_string()
}
