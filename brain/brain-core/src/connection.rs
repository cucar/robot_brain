//! A connection (D25): exposures and earnings.

#[derive(Clone, Copy, Debug, Default)]
pub struct Conn {
    pub strength: f64,
    pub earnings: f64,
}
impl Conn {
    pub fn estimate(&self) -> f64 { if self.strength > 0.0 { self.earnings / self.strength } else { 0.0 } }
}

