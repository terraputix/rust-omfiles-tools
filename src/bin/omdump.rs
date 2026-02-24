use omfiles::MmapFile;
use omfiles::OmFilesError;
use omfiles::reader::OmFileReader;
use omfiles::traits::OmArrayVariable;
use omfiles::traits::OmFileReadable;
use omfiles::traits::OmFileVariable;
use omfiles_tools::navigate;
use std::env;
use std::ops::Range;

/// Display information about a variable and its children recursively
fn print_variable_info(reader: &OmFileReader<MmapFile>, indent: usize, path: &str) {
    let indent_str = " ".repeat(indent);

    let variable_name = reader.name();
    let variable_data_type = reader.data_type();

    println!("{}Variable: {}", indent_str, path);
    println!("{}  Name: {:?}", indent_str, variable_name);
    println!("{}  Type: {:?}", indent_str, variable_data_type);

    if let Ok(array) = reader.expect_array() {
        let variable_compression = array.compression();

        let variable_dimensions = array.get_dimensions();
        let dims_str = variable_dimensions
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" × ");

        let chunks = array.get_chunk_dimensions();
        let chunks_str = if !chunks.is_empty() {
            chunks
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" × ")
        } else {
            "none".to_string()
        };

        let dim_ranges: Vec<Range<u64>> = variable_dimensions.iter().map(|&d| 0..d).collect();
        let byte_range = array.get_byte_range::<f32>(&dim_ranges).unwrap();

        // Helper to format byte counts as bytes, KB, or MB
        let format_bytes = |bytes: u64| -> String {
            const KB: f64 = 1024.0;
            const MB: f64 = KB * 1024.0;
            let b = bytes as f64;
            if b >= MB {
                format!("{:.2} MB", b / MB)
            } else if b >= KB {
                format!("{:.2} KB", b / KB)
            } else {
                format!("{} bytes", bytes)
            }
        };

        let byte_count = byte_range.end - byte_range.start;
        let byte_range_str = format!(
            "{}..{} ({})",
            byte_range.start,
            byte_range.end,
            format_bytes(byte_count)
        );

        // Compute compression factor: uncompressed_bytes / stored_bytes
        let num_elements = variable_dimensions
            .iter()
            .fold(1u128, |acc, &d| acc.saturating_mul(d as u128));
        let element_size = std::mem::size_of::<f32>() as u128;
        let uncompressed_bytes = num_elements.saturating_mul(element_size);
        let stored_bytes = (byte_range.end - byte_range.start) as u128;

        let compression_factor_str = if stored_bytes > 0 {
            let factor = (uncompressed_bytes as f64) / (stored_bytes as f64);
            if factor.is_finite() {
                format!("{:.2}×", factor)
            } else {
                "unknown".to_string()
            }
        } else {
            "unknown".to_string()
        };

        println!("{}  Dimensions: [{}]", indent_str, dims_str);
        println!("{}  Chunks: [{}]", indent_str, chunks_str);
        println!("{}  Byte Range: [{}]", indent_str, byte_range_str);
        println!(
            "{}  Compression: {:?} (factor: {})",
            indent_str, variable_compression, compression_factor_str
        );
    }

    let num_children = reader.number_of_children();
    for i in 0..num_children {
        if let Some(child) = reader.get_child_by_index(i) {
            let child_name = child.name();
            let child_path = if path.is_empty() {
                child_name.to_string()
            } else {
                format!("{}/{}", path, child_name)
            };
            print_variable_info(&child, indent + 2, &child_path);
        }
    }
}

fn parse_range(range_str: &str) -> Option<Range<u64>> {
    let parts: Vec<&str> = range_str.split("..").collect();
    if parts.len() != 2 {
        return None;
    }
    let start = parts[0].parse::<u64>().ok()?;
    let end = parts[1].parse::<u64>().ok()?;
    Some(start..end)
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage:
  {0} <om-file>
      # Info dump (recursive)
  {0} <om-file> <var-path> <dim0_range> [<dim1_range> ...]
      # Read values from a variable (by path) and ranges

  <var-path> can be:
    - the variable name (e.g. 'data')
    - a child index (e.g. 'child_0')
    - 'unnamed' for the first unnamed variable at each level
    - 'root' or '.' to refer to the root variable

  Example: {0} chunk.om data 0..1 0..100 0..50
  Example: {0} chunk.om root 0..1 0..100 0..50
  Example: {0} chunk.om . 0..1 0..100 0..50",
        program
    );
}

fn main() -> Result<(), OmFilesError> {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 {
        // Info dump mode
        let filename = &args[1];
        let reader = OmFileReader::from_file(filename)?;
        println!("OM File: {}", filename);
        println!("=========================================");
        print_variable_info(&reader, 0, "");
        return Ok(());
    } else if args.len() >= 4 {
        // Value read mode
        let filename = &args[1];
        let var_path = &args[2];
        let ranges: Vec<Option<Range<u64>>> = args[3..].iter().map(|s| parse_range(s)).collect();

        let reader = OmFileReader::from_file(filename)?;

        let variable = match navigate::resolve_variable(reader, var_path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error: {}", e);
                print_usage(&args[0]);
                return Ok(());
            }
        };

        // Only attempt to read data if the target is an array
        if let Ok(array) = variable.expect_array() {
            let dims = array.get_dimensions();
            println!("dimensions: {:?}", dims);
            println!("chunk_dimensions: {:?}", array.get_chunk_dimensions());

            if ranges.len() != dims.len() || ranges.iter().any(|r| r.is_none()) {
                eprintln!(
                    "Number of valid ranges ({}) doesn't match number of dimensions ({}), or invalid range format.",
                    ranges.iter().filter(|r| r.is_some()).count(),
                    dims.len()
                );
                print_usage(&args[0]);
                return Ok(());
            }

            let ranges: Vec<Range<u64>> = ranges.into_iter().map(|r| r.unwrap()).collect();

            // Only f32 is supported here, but could be extended with a match on data_type()
            let data = array.read::<f32>(&ranges).expect("Failed to read data");
            println!("{:?}", data);
        } else {
            eprintln!("Error: The variable at '{}' is not an array.", var_path);
        }

        return Ok(());
    } else {
        print_usage(&args[0]);
        return Ok(());
    }
}
