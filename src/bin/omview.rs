use eframe::egui::{self, CentralPanel, TopBottomPanel};
use omfiles_rs::io::reader::OmFileReader;
use std::env;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum ChunkingMode {
    Spatial,
    Temporal,
}

impl ChunkingMode {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "spatial" => Some(ChunkingMode::Spatial),
            "temporal" => Some(ChunkingMode::Temporal),
            _ => None,
        }
    }
}

struct DataLoader {
    reader: OmFileReader<omfiles_rs::backend::mmapfile::MmapFile>,
    n_timestamps: u64,
    chunking: ChunkingMode,
}

impl DataLoader {
    fn new(file_path: &str, chunking: ChunkingMode) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = OmFileReader::from_file(file_path)?;
        let dims = reader.get_dimensions();
        let n_timestamps = *dims.last().unwrap();

        Ok(Self {
            reader,
            n_timestamps,
            chunking,
        })
    }

    fn get_timestamp_data(
        &self,
        timestamp: u64,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Ix2>, Box<dyn std::error::Error>>
    {
        let dims = self.reader.get_dimensions();
        let (rows, cols, ranges) = match self.chunking {
            ChunkingMode::Temporal => {
                // [lat, lon, time]
                let mut ranges = vec![];
                for (i, &dim) in dims.iter().enumerate() {
                    if i == dims.len() - 1 {
                        ranges.push(timestamp..timestamp + 1);
                    } else {
                        ranges.push(0..dim);
                    }
                }
                (dims[0] as usize, dims[1] as usize, ranges)
            }
            ChunkingMode::Spatial => {
                // [time, lat, lon]
                let mut ranges = vec![];
                for (i, &dim) in dims.iter().enumerate() {
                    if i == 0 {
                        ranges.push(timestamp..timestamp + 1);
                    } else {
                        ranges.push(0..dim);
                    }
                }
                (dims[1] as usize, dims[2] as usize, ranges)
            }
        };

        let nd_data = self.reader.read::<f32>(&ranges, None, None)?;
        let result = nd_data
            .squeeze()
            .into_shape_clone(ndarray::Ix2(rows, cols))?;
        Ok(result)
    }
}

struct App {
    data_loader: Arc<DataLoader>,
    current_timestamp: u64,
    plot_data: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Ix2>,
}

impl App {
    fn new(data_loader: Arc<DataLoader>) -> Result<Self, Box<dyn std::error::Error>> {
        let dims = data_loader.reader.get_dimensions().to_vec();
        println!("dimensions {:?}", dims);
        let initial_data = data_loader.get_timestamp_data(0)?;

        Ok(Self {
            data_loader,
            current_timestamp: 0,
            plot_data: initial_data,
        })
    }

    fn update_plot_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.plot_data = self
            .data_loader
            .get_timestamp_data(self.current_timestamp)?;
        Ok(())
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::bottom("playmenu").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("<-").clicked() && self.current_timestamp > 0 {
                    self.current_timestamp -= 1;
                    if let Err(e) = self.update_plot_data() {
                        eprintln!("Error updating plot data: {}", e);
                    }
                }

                ui.label(format!("Timestamp: {}", self.current_timestamp));

                if ui.button("->").clicked()
                    && self.current_timestamp < self.data_loader.n_timestamps - 1
                {
                    self.current_timestamp += 1;
                    if let Err(e) = self.update_plot_data() {
                        eprintln!("Error updating plot data: {}", e);
                    }
                }
            });
        });
        CentralPanel::default().show(ctx, |ui| {
            let all_nan = self.plot_data.iter().all(|val| val.is_nan());
            if all_nan {
                println!("All values are nan");
                return;
            }

            let min_value: f32 = *self
                .plot_data
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let max_value: f32 = *self
                .plot_data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            let (rows, cols) = self.plot_data.dim();
            // Prepare RGBA buffer
            let mut rgba_data = Vec::with_capacity(rows * cols * 4);
            for y in (0..rows).rev() {
                for x in 0..cols {
                    let value = self.plot_data[[y, x]];
                    let normalized = (value - min_value) / (max_value - min_value);
                    let color = viridis_color(normalized);
                    rgba_data.push(color.0); // R
                    rgba_data.push(color.1); // G
                    rgba_data.push(color.2); // B
                    rgba_data.push(255); // A
                }
            }

            // Create egui image and texture
            let image = egui::ColorImage::from_rgba_unmultiplied([cols, rows], &rgba_data);
            let texture = ui
                .ctx()
                .load_texture("heatmap", image, egui::TextureOptions::NEAREST);

            // Show the image, scaling to fit the available space
            let image_response = ui.image(&texture, ui.available_size());

            // Only proceed if hovered
            if image_response.hovered() {
                if let Some(pointer_pos) = ui.ctx().pointer_hover_pos() {
                    let rect = image_response.rect;
                    // Convert pointer position to image coordinates
                    let x = ((pointer_pos.x - rect.left()) / rect.width() * cols as f32).floor()
                        as usize;
                    let y = ((pointer_pos.y - rect.top()) / rect.height() * rows as f32).floor()
                        as usize;

                    if x < cols && y < rows {
                        let value = self.plot_data[[y, x]];
                        // Show tooltip at mouse
                        ui.ctx().output_mut(|o| {
                            o.cursor_icon = egui::CursorIcon::PointingHand;
                        });
                        image_response.on_hover_ui(|ui| {
                            ui.label(format!("({}, {}): {:.4}", x, y, value));
                        });
                    }
                }
            }
        });
    }
}

struct RGBColor(pub u8, pub u8, pub u8);

fn viridis_color(v: f32) -> RGBColor {
    // Ensure v is in [0, 1]
    let v = v.clamp(0.0, 1.0);

    // Red component
    let r = if v < 0.5 {
        0.0
    } else {
        ((v - 0.5) * 2.0).powf(1.5) * 255.0
    };

    // Green component
    let g = if v < 0.4 {
        v * 3.0 * 255.0
    } else {
        (1.0 - (v - 0.4) / 0.6) * 255.0
    };

    // Blue component
    let b = if v < 0.7 {
        255.0 * (1.0 - v.powf(0.5))
    } else {
        0.0
    };

    RGBColor(r as u8, g as u8, b as u8)
}

fn print_usage_and_exit(program: &str) -> ! {
    eprintln!(
        "Usage: {} <omfile> [--chunking spatial|temporal]\n\
         Default is temporal chunking (last (and fast) dimension is time).",
        program
    );
    std::process::exit(1);
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage_and_exit(&args[0]);
    }

    let mut chunking = ChunkingMode::Temporal;
    let mut omfile = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--chunking" => {
                i += 1;
                if i >= args.len() {
                    print_usage_and_exit(&args[0]);
                }
                chunking = ChunkingMode::from_str(&args[i]).unwrap_or_else(|| {
                    eprintln!("Invalid chunking mode: {}", args[i]);
                    print_usage_and_exit(&args[0]);
                });
            }
            s if omfile.is_none() => {
                omfile = Some(s.to_string());
            }
            _ => {
                print_usage_and_exit(&args[0]);
            }
        }
        i += 1;
    }

    let omfile = omfile.unwrap_or_else(|| {
        print_usage_and_exit(&args[0]);
    });

    let data_loader =
        Arc::new(DataLoader::new(&omfile, chunking).expect("Could not init DataLoader"));

    let native_options = eframe::NativeOptions {
        ..Default::default()
    };

    eframe::run_native(
        "Heatmap Viewer",
        native_options,
        Box::new(move |_cc| {
            let app = App::new(data_loader.clone()).unwrap();
            Box::new(app) as Box<dyn eframe::App>
        }),
    )
    .unwrap();

    Ok(())
}
