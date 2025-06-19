use omfiles_rs::backend::mmapfile::MmapFile;
use omfiles_rs::errors::OmFilesRsError;
use omfiles_rs::io::reader::OmFileReader;
use std::env;
use std::ops::Range;

/// Display information about a variable and its children recursively
fn print_variable_info(reader: &OmFileReader<MmapFile>, indent: usize, path: &str) {
    let indent_str = " ".repeat(indent);

    let variable_name = reader.get_name().unwrap_or_else(|| "unnamed".to_string());
    let variable_data_type = reader.data_type();
    let variable_compression = reader.compression();

    // Get dimensions
    let variable_dimensions = reader.get_dimensions();
    let dims_str = variable_dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" × ");

    // Get chunks
    let chunks = reader.get_chunk_dimensions();
    let chunks_str = if !chunks.is_empty() {
        chunks
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" × ")
    } else {
        "none".to_string()
    };

    // Print information
    println!("{}Variable: {}", indent_str, path);
    println!("{}  Name: {}", indent_str, variable_name);
    println!("{}  Type: {:?}", indent_str, variable_data_type);
    println!("{}  Compression: {:?}", indent_str, variable_compression);
    println!("{}  Dimensions: [{}]", indent_str, dims_str);
    println!("{}  Chunks: [{}]", indent_str, chunks_str);

    // Process children recursively
    let num_children = reader.number_of_children();
    for i in 0..num_children {
        if let Some(child) = reader.get_child(i) {
            let child_name = child.get_name().unwrap_or_else(|| format!("child_{}", i));
            let child_path = if path.is_empty() {
                child_name.clone()
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

fn print_variable_data(
    variable: &OmFileReader<MmapFile>,
    ranges: &Vec<Range<u64>>,
) -> Result<(), OmFilesRsError> {
    // Only f32 is supported here, but we could extend this with a match on variable.data_type()
    let data = variable
        .read::<f32>(&ranges, None, None)
        .expect("Failed to read data");

    println!("{:?}", data);
    Ok(())
}

fn main() -> Result<(), OmFilesRsError> {
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
        let mut variable = reader;

        let mut path_parts = if var_path.is_empty() || var_path == "root" || var_path == "." {
            vec![]
        } else {
            var_path.split('/').collect::<Vec<_>>()
        };

        if !path_parts.is_empty() {
            // Check if the first path part refers to the root variable itself
            let root_name = variable.get_name();
            let first = path_parts[0];

            let root_matches = match &root_name {
                Some(name) if name == first => true,
                None if first == "unnamed" || first == "child_0" => true,
                _ => false,
            };

            if root_matches {
                // The root is the target or the starting point for further traversal
                path_parts.remove(0);
            }

            // Traverse remaining path parts as children
            for part in path_parts {
                let mut found = false;
                for i in 0..variable.number_of_children() {
                    if let Some(child) = variable.get_child(i) {
                        let name = child.get_name();
                        if part.starts_with("child_") {
                            if let Ok(idx) = part["child_".len()..].parse::<u32>() {
                                if idx == i {
                                    variable = child;
                                    found = true;
                                    break;
                                }
                            }
                        } else if part == "unnamed" && name.is_none() {
                            variable = child;
                            found = true;
                            break;
                        } else if name.as_deref() == Some(part) {
                            variable = child;
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    eprintln!("Variable path '{}' not found.", var_path);
                    print_usage(&args[0]);
                    return Ok(());
                }
            }
        }

        let dims = variable.get_dimensions();
        println!("dimensions: {:?}", dims);
        println!("chunk_dimensions: {:?}", variable.get_chunk_dimensions());

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

        return print_variable_data(&variable, &ranges);
    } else {
        print_usage(&args[0]);
        return Ok(());
    }
}
