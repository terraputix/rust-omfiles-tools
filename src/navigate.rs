use omfiles::MmapFile;
use omfiles::reader::OmFileReader;
use omfiles::traits::OmArrayVariable;
use omfiles::traits::OmFileReadable;
use omfiles::traits::OmFileVariable;

/// Navigate from a root reader to a child variable by slash-separated path.
///
/// Supports:
/// - Name-based lookup via `get_child_by_name` (e.g. `"temperature"`, `"group/temperature"`)
/// - Index-based lookup via `"child_N"` syntax (e.g. `"child_0"`, `"group/child_2"`)
/// - `"unnamed"` to match the first child with an empty name
/// - `""`, `"root"`, or `"."` to return the reader unchanged
///
/// Consumes the input reader since `OmFileReader` is not `Clone`.
pub fn resolve_variable(
    reader: OmFileReader<MmapFile>,
    path: &str,
) -> Result<OmFileReader<MmapFile>, String> {
    if path.is_empty() || path == "root" || path == "." {
        return Ok(reader);
    }

    let parts: Vec<&str> = path.split('/').collect();
    let mut current = reader;

    // If the first path segment matches the root variable name, skip it
    let root_name = current.name();
    let start_idx = if !parts.is_empty() && !root_name.is_empty() && root_name == parts[0] {
        1
    } else {
        0
    };

    for part in &parts[start_idx..] {
        let found = if part.starts_with("child_") {
            // Index-based access: child_0, child_1, etc.
            if let Ok(idx) = part["child_".len()..].parse::<u32>() {
                current.get_child_by_index(idx)
            } else {
                None
            }
        } else if *part == "unnamed" {
            // Match first child with empty name
            let mut result = None;
            for i in 0..current.number_of_children() {
                if let Some(child) = current.get_child_by_index(i) {
                    if child.name().is_empty() {
                        result = Some(child);
                        break;
                    }
                }
            }
            result
        } else {
            // Name-based lookup
            current.get_child_by_name(part)
        };

        current = found.ok_or_else(|| {
            let children = list_children(&current);
            format!(
                "Variable path segment '{}' not found. Available children: {}",
                part, children
            )
        })?;
    }

    Ok(current)
}

/// List available children names for error/diagnostic messages.
pub fn list_children(reader: &OmFileReader<MmapFile>) -> String {
    let n = reader.number_of_children();
    if n == 0 {
        return "(none)".to_string();
    }
    let mut names = Vec::new();
    for i in 0..n {
        if let Some(child) = reader.get_child_by_index(i) {
            let name = child.name();
            if name.is_empty() {
                names.push(format!("child_{} (unnamed)", i));
            } else {
                names.push(name.to_string());
            }
        }
    }
    names.join(", ")
}

/// Print the variable tree recursively to stderr, with dimension info for arrays.
pub fn print_children_recursive(reader: &OmFileReader<MmapFile>, indent: usize) {
    let indent_str = " ".repeat(indent);
    let n = reader.number_of_children();

    for i in 0..n {
        if let Some(child) = reader.get_child_by_index(i) {
            let name = child.name();
            let display_name = if name.is_empty() {
                format!("(unnamed, index {})", i)
            } else {
                name.to_string()
            };

            let type_info = if let Ok(array) = child.expect_array() {
                let dims = array.get_dimensions();
                let dims_str = dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("×");
                format!(" [{}]", dims_str)
            } else {
                " (group)".to_string()
            };

            eprintln!("{}  - {}{}", indent_str, display_name, type_info);
            print_children_recursive(&child, indent + 2);
        }
    }
}
