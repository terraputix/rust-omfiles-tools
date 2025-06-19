# rust-omfiles-tools

A collection of command-line tools for working with omfiles written in Rust.

## CLI Tools

- **omdump**: Inspect OM file structure, metadata, and optionally print variable values for specified ranges.
- **omview**: Interactive GUI to visualize OM file data as heatmaps, with support for temporal and spatial chunking.
- **om_temporal_to_spatial**: Convert OM files from temporal chunking ([time, lat, lon]) to spatial chunking ([lat, lon, time]) layout.
