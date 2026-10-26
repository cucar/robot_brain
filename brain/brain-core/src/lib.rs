//! UCAR: the machine of docs/algorithm.md. A neuron records and returns (`neuron`, `history`,
//! `connection`); the machine holds the dictionary (`function`), recognizes (`neighborhood`, `fit`,
//! `recognize`, `election`, `death`), and runs the future (`connect`, `expansion`, `selection`), in
//! `machine` and `activation`. The dispatch is `thalamus`, `region` and `column`: the thalamus owns the
//! regions, a region its columns, a column its neurons, and each neuron's activate call of a level fans
//! out over the columns in parallel.

pub mod activation;
pub mod column;
pub mod connect;
pub mod connection;
pub mod death;
pub mod election;
pub mod expansion;
pub mod fit;
pub mod function;
pub mod history;
pub mod machine;
pub mod neighborhood;
pub mod neuron;
pub mod offsets;
pub mod recognize;
pub mod region;
pub mod selection;
pub mod thalamus;
pub mod types;
