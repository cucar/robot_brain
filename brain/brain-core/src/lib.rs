//! UCAR: the machine of docs/algorithm.md. One file per type: the neuron side (Parts II and IV) is
//! `neuron`, `pattern`, `history`, `fit`, `collapse`, `margin`, `pick`, `merge`, `bid` and
//! `connection`; the machine side (Parts III and V) is `machine`, `activation`, `neighborhood`,
//! `election`, `index`, `death`, `connect`, `expansion` and `selection`; the dispatch is `thalamus`, `region` and
//! `column`: the thalamus owns the regions, a region its columns, a column its neurons, and the per-neuron work
//! of a level fans out over the columns in parallel.

pub mod activation;
pub mod bid;
pub mod collapse;
pub mod column;
pub mod connect;
pub mod connection;
pub mod fit;
pub mod death;
pub mod election;
pub mod expansion;
pub mod history;
pub mod index;
pub mod machine;
pub mod margin;
pub mod merge;
pub mod neighborhood;
pub mod neuron;
pub mod offsets;
pub mod pattern;
pub mod pick;
pub mod region;
pub mod selection;
pub mod thalamus;
pub mod types;
