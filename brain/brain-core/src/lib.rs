//! UCAR: the machine of docs/algorithm.md. One file per type: the neuron side (Parts II and IV) is
//! `neuron`, `pattern`, `history`, `cover`, `collapse`, `margin`, `pick`, `merge`, `bid` and
//! `connection`; the machine side (Parts III and V) is `machine`, `activation`, `neighborhood`,
//! `election`, `index`, `death`, `connect`, `expansion` and `selection`.

pub mod activation;
pub mod bid;
pub mod collapse;
pub mod connect;
pub mod connection;
pub mod cover;
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
pub mod selection;
pub mod types;
