use omfiles::MmapFile;
use omfiles::OmFilesError;
use omfiles::reader::OmFileArray;
use omfiles::reader::OmFileReader;
use omfiles::traits::OmArrayVariable;
use omfiles::traits::OmFileReadable;
use omfiles::traits::OmFileVariable;
use std::env;
use std::ops::Range;

/// Display information about a variable and its children recursively
fn print_variable_info(reader: &OmFileReader<MmapFile>, indent: usize, path: &str) {
    let indent_str = " ".repeat(indent);

    let variable_name = reader.name();
    let variable_data_type = reader.data_type();

    // Print common information
    println!("{}Variable: {}", indent_str, path);
    println!("{}  Name: {:?}", indent_str, variable_name);
    println!("{}  Type: {:?}", indent_str, variable_data_type);

    // Only print array-specific information if it can be cast as an array
    if let Ok(array) = reader.expect_array() {
        let variable_compression = array.compression();

        // Get dimensions
        let variable_dimensions = array.get_dimensions();
        let dims_str = variable_dimensions
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" × ");

        // Get chunks
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

        println!("{}  Compression: {:?}", indent_str, variable_compression);
        println!("{}  Dimensions: [{}]", indent_str, dims_str);
        println!("{}  Chunks: [{}]", indent_str, chunks_str);
    }

    // Process children recursively
    let num_children = reader.number_of_children();
    for i in 0..num_children {
        if let Some(child) = reader.get_child_by_index(i) {
            let child_name = child.name();
            let child_path = if path.is_empty() {
                child_name
            } else {
                &format!("{}/{}", path, child_name)
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
    variable: &OmFileArray<MmapFile>,
    ranges: &Vec<Range<u64>>,
) -> Result<(), OmFilesError> {
    // Only f32 is supported here, but we could extend this with a match on variable.data_type()
    let data = variable.read::<f32>(&ranges).expect("Failed to read data");

    println!("{:?}", data);
    Ok(())
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
        let mut variable = reader;

        let mut path_parts = if var_path.is_empty() || var_path == "root" || var_path == "." {
            vec![]
        } else {
            var_path.split('/').collect::<Vec<_>>()
        };

        if !path_parts.is_empty() {
            // Check if the first path part refers to the root variable itself
            let root_name = variable.name();
            let first = path_parts[0];

            let root_matches = root_name == first || first == "unnamed" || first == "child_0";

            if root_matches {
                path_parts.remove(0);
            }

            for part in path_parts {
                let mut found = false;
                for i in 0..variable.number_of_children() {
                    if let Some(child) = variable.get_child_by_index(i) {
                        let name = child.name();
                        if part.starts_with("child_") {
                            if let Ok(idx) = part["child_".len()..].parse::<u32>() {
                                if idx == i {
                                    variable = child;
                                    found = true;
                                    break;
                                }
                            }
                        } else if part == "unnamed" && name.is_empty() {
                            variable = child;
                            found = true;
                            break;
                        } else if name == part {
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
            return print_variable_data(&array, &ranges);
        } else {
            eprintln!("Error: The variable at '{}' is not an array.", var_path);
            return Ok(());
        }
    } else {
        print_usage(&args[0]);
        return Ok(());
    }
}
