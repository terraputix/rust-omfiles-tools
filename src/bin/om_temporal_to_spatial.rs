use omfiles_rs::io::reader::OmFileReader;
use omfiles_rs::io::writer::OmFileWriter;
use std::env;
use std::fs::File;
use std::io;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input_om_file> <output_om_file>", args[0]);
        std::process::exit(1);
    }

    let input_file_path = &args[1];
    let output_file_path = &args[2];

    // Read data from the input OM file
    let reader = OmFileReader::from_file(input_file_path)
        .expect(&format!("Failed to open file: {}", input_file_path));

    let dimensions = reader.get_dimensions();
    let chunks = reader.get_chunk_dimensions();

    println!("Input file info:");
    println!("  compression: {:?}", reader.compression());
    println!("  dimensions: {:?}", dimensions);
    println!("  chunks: {:?}", chunks);
    println!("  scale_factor: {}", reader.scale_factor());

    // Original dimensions are [lat, lon, time]
    let lat_dim = dimensions[0];
    let lon_dim = dimensions[1];
    let time_dim = dimensions[2];

    let file_handle = File::create(output_file_path).expect("Failed to create output file");

    // Write the compressed data to the output OM file
    let mut file_writer = OmFileWriter::new(
        &file_handle,
        1024 * 1024 * 1024, // Initial capacity of 1GB
    );
    println!("Created writer");

    let reformatted_dimensions = vec![time_dim, lat_dim, lon_dim];
    // Use single time slices as chunks
    let rechunked_dimensions = vec![1, lat_dim, lon_dim];

    let mut writer = file_writer
        .prepare_array::<f32>(
            reformatted_dimensions.clone(),
            rechunked_dimensions.clone(),
            reader.compression(),
            reader.scale_factor(),
            reader.add_offset(),
        )
        .expect("Failed to prepare array");

    println!("Prepared output array");
    println!("Reformatting data from [lat, lon, time] to [time, lat, lon]...");

    // Process one time slice at a time
    for t in 0..time_dim {
        // Read a time slice from the input file and convert dynamic array to 3D array
        let time_slice_data = reader
            .read(&[0u64..lat_dim, 0u64..lon_dim, t..t + 1u64], None, None)
            .expect("Failed to read data")
            .into_shape_clone(ndarray::Ix3(lat_dim as usize, lon_dim as usize, 1))
            .expect("Failed to convert to 3D array");

        // Transpose from [lat, lon, 1] to [1, lat, lon]
        let spatial_data = time_slice_data.permuted_axes([2, 0, 1]);

        // Write this time slice to the new file
        writer
            .write_data(spatial_data.into_dyn().view(), None, None)
            .expect(&format!("Failed to write data for time {}", t));

        if t % 10 == 0 || t == time_dim - 1 {
            println!("Processed time slice {}/{}", t + 1, time_dim);
        }
    }

    let variable_meta = writer.finalize();
    println!("Finalized array");

    let variable = file_writer
        .write_array(variable_meta, "data", &[])
        .expect("Failed to write array metadata");
    file_writer
        .write_trailer(variable)
        .expect("Failed to write trailer");

    println!("Finished writing");

    Ok(())
}
