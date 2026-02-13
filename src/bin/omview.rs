use eframe::egui::load::SizedTexture;
use eframe::egui::{self, CentralPanel, TopBottomPanel};
use omfiles::MmapFile;
use omfiles::reader::OmFileReader;
use omfiles::traits::OmArrayVariable;
use omfiles::traits::OmFileVariable;
use omfiles_tools::navigate;
use std::env;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Debug)]
enum DataShape {
    TwoDim {
        rows: u64,
        cols: u64,
    },
    ThreeDim {
        rows: u64,
        cols: u64,
        n_timestamps: u64,
    },
}

struct DataLoader {
    reader: OmFileReader<MmapFile>,
    variable_path: Option<String>,
    shape: DataShape,
    chunking: ChunkingMode,
}

impl DataLoader {
    fn new(
        file_path: &str,
        variable_path: Option<&str>,
        chunking: ChunkingMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let root_reader = OmFileReader::from_file(file_path)?;

        let target = if let Some(path) = variable_path {
            navigate::resolve_variable(root_reader, path)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
        } else {
            root_reader
        };

        let array = target
            .expect_array()
            .map_err(|e| format!("Target variable is not an array: {}", e))?;
        let dims = array.get_dimensions();

        let shape = match dims.len() {
            2 => DataShape::TwoDim {
                rows: dims[0],
                cols: dims[1],
            },
            3 => match chunking {
                ChunkingMode::Temporal => DataShape::ThreeDim {
                    rows: dims[0],
                    cols: dims[1],
                    n_timestamps: dims[2],
                },
                ChunkingMode::Spatial => DataShape::ThreeDim {
                    rows: dims[1],
                    cols: dims[2],
                    n_timestamps: dims[0],
                },
            },
            n => {
                return Err(
                    format!("Unsupported number of dimensions: {} (expected 2 or 3)", n).into(),
                );
            }
        };

        drop(array);

        Ok(Self {
            reader: target,
            variable_path: variable_path.map(|s| s.to_string()),
            shape,
            chunking,
        })
    }

    fn n_timestamps(&self) -> u64 {
        match &self.shape {
            DataShape::TwoDim { .. } => 1,
            DataShape::ThreeDim { n_timestamps, .. } => *n_timestamps,
        }
    }

    fn rows(&self) -> u64 {
        match &self.shape {
            DataShape::TwoDim { rows, .. } | DataShape::ThreeDim { rows, .. } => *rows,
        }
    }

    fn cols(&self) -> u64 {
        match &self.shape {
            DataShape::TwoDim { cols, .. } | DataShape::ThreeDim { cols, .. } => *cols,
        }
    }

    fn is_temporal(&self) -> bool {
        matches!(&self.shape, DataShape::ThreeDim { .. })
    }

    fn get_timestamp_data(
        &self,
        timestamp: u64,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Ix2>, Box<dyn std::error::Error>>
    {
        let array = self
            .reader
            .expect_array()
            .map_err(|e| format!("Failed to access array data: {}", e))?;

        let rows = self.rows() as usize;
        let cols = self.cols() as usize;

        let ranges = match &self.shape {
            DataShape::TwoDim {
                rows: r, cols: c, ..
            } => {
                vec![0..*r, 0..*c]
            }
            DataShape::ThreeDim {
                rows: r, cols: c, ..
            } => match self.chunking {
                ChunkingMode::Temporal => {
                    vec![0..*r, 0..*c, timestamp..timestamp + 1]
                }
                ChunkingMode::Spatial => {
                    vec![timestamp..timestamp + 1, 0..*r, 0..*c]
                }
            },
        };

        let nd_data = array
            .read::<f32>(&ranges)
            .map_err(|e| format!("Failed to read data at timestamp {}: {}", timestamp, e))?;

        let result = nd_data
            .into_shape_clone(ndarray::Ix2(rows, cols))
            .map_err(|e| format!("Failed to reshape data to {}x{}: {}", rows, cols, e))?;

        Ok(result)
    }
}

struct App {
    data_loader: Arc<DataLoader>,
    current_timestamp: u64,
    plot_data: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Ix2>,
    error_message: Option<String>,
}

impl App {
    fn new(data_loader: Arc<DataLoader>) -> Result<Self, Box<dyn std::error::Error>> {
        let initial_data = data_loader.get_timestamp_data(0)?;

        Ok(Self {
            data_loader,
            current_timestamp: 0,
            plot_data: initial_data,
            error_message: None,
        })
    }

    fn update_plot_data(&mut self) {
        match self.data_loader.get_timestamp_data(self.current_timestamp) {
            Ok(data) => {
                self.plot_data = data;
                self.error_message = None;
            }
            Err(e) => {
                self.error_message = Some(format!(
                    "Error loading timestamp {}: {}",
                    self.current_timestamp, e
                ));
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let var_label = if let Some(ref path) = self.data_loader.variable_path {
                    format!("Variable: {} | ", path)
                } else {
                    String::new()
                };
                ui.label(format!(
                    "{}Shape: {}×{} | {:?}",
                    var_label,
                    self.data_loader.rows(),
                    self.data_loader.cols(),
                    self.data_loader.chunking,
                ));
            });
            if let Some(ref err) = self.error_message {
                ui.colored_label(egui::Color32::RED, err);
            }
        });

        if self.data_loader.is_temporal() {
            TopBottomPanel::bottom("playmenu").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let max_ts = self.data_loader.n_timestamps().saturating_sub(1);

                    if ui.button("⏮").clicked() && self.current_timestamp != 0 {
                        self.current_timestamp = 0;
                        self.update_plot_data();
                    }

                    if ui.button("◀").clicked() && self.current_timestamp > 0 {
                        self.current_timestamp -= 1;
                        self.update_plot_data();
                    }

                    let mut ts = self.current_timestamp as f64;
                    let slider = egui::Slider::new(&mut ts, 0.0..=max_ts as f64)
                        .step_by(1.0)
                        .text("Timestamp");
                    if ui.add(slider).changed() {
                        self.current_timestamp = ts as u64;
                        self.update_plot_data();
                    }

                    if ui.button("▶").clicked() && self.current_timestamp < max_ts {
                        self.current_timestamp += 1;
                        self.update_plot_data();
                    }

                    if ui.button("⏭").clicked() && self.current_timestamp != max_ts {
                        self.current_timestamp = max_ts;
                        self.update_plot_data();
                    }

                    ui.label(format!("{} / {}", self.current_timestamp, max_ts));
                });
            });
        }

        CentralPanel::default().show(ctx, |ui| {
            if self.error_message.is_some() {
                return;
            }

            let valid_values: Vec<f32> = self
                .plot_data
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .collect();

            if valid_values.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("All values are NaN for this timestamp.");
                });
                return;
            }

            let min_value = valid_values.iter().copied().fold(f32::INFINITY, f32::min);
            let max_value = valid_values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            let range = if (max_value - min_value).abs() < f32::EPSILON {
                1.0
            } else {
                max_value - min_value
            };

            let (rows, cols) = self.plot_data.dim();
            let mut rgba_data = Vec::with_capacity(rows * cols * 4);
            for y in (0..rows).rev() {
                for x in 0..cols {
                    let value = self.plot_data[[y, x]];
                    if value.is_nan() {
                        rgba_data.push(30);
                        rgba_data.push(30);
                        rgba_data.push(30);
                        rgba_data.push(255);
                    } else {
                        let normalized = (value - min_value) / range;
                        let color = viridis_color(normalized);
                        rgba_data.push(color.0);
                        rgba_data.push(color.1);
                        rgba_data.push(color.2);
                        rgba_data.push(255);
                    }
                }
            }

            let image = egui::ColorImage::from_rgba_unmultiplied([cols, rows], &rgba_data);
            let texture = ui
                .ctx()
                .load_texture("heatmap", image, egui::TextureOptions::NEAREST);

            let image_response = ui.image(SizedTexture::new(&texture, ui.available_size()));

            if image_response.hovered() {
                if let Some(pointer_pos) = ui.ctx().pointer_hover_pos() {
                    let rect = image_response.rect;
                    let x = ((pointer_pos.x - rect.left()) / rect.width() * cols as f32).floor()
                        as usize;
                    let y = ((pointer_pos.y - rect.top()) / rect.height() * rows as f32).floor()
                        as usize;

                    if x < cols && y < rows {
                        let value = self.plot_data[[y, x]];
                        ui.ctx().output_mut(|o| {
                            o.cursor_icon = egui::CursorIcon::PointingHand;
                        });
                        image_response.on_hover_ui(|ui| {
                            ui.label(format!("({}, {}): {:.4}", x, y, value));
                        });
                    }
                }
            }

            ui.with_layout(egui::Layout::bottom_up(egui::Align::RIGHT), |ui| {
                ui.label(format!("Range: [{:.4}, {:.4}]", min_value, max_value));
            });
        });
    }
}

struct RGBColor(pub u8, pub u8, pub u8);

fn viridis_color(v: f32) -> RGBColor {
    let v = v.clamp(0.0, 1.0);

    let r = if v < 0.5 {
        0.0
    } else {
        ((v - 0.5) * 2.0).powf(1.5) * 255.0
    };

    let g = if v < 0.4 {
        v * 3.0 * 255.0
    } else {
        (1.0 - (v - 0.4) / 0.6) * 255.0
    };

    let b = if v < 0.7 {
        255.0 * (1.0 - v.powf(0.5))
    } else {
        0.0
    };

    RGBColor(r as u8, g as u8, b as u8)
}

fn print_usage_and_exit(program: &str) -> ! {
    eprintln!(
        "Usage: {} <omfile> [--variable <path>] [--chunking spatial|temporal]\n\
         \n\
         Options:\n\
         \x20 --variable <path>    Select a variable by path (e.g. 'data', 'group/temperature')\n\
         \x20                      Navigates using get_child_by_name.\n\
         \x20                      Also supports 'child_N', 'unnamed', 'root', '.'.\n\
         \x20 --chunking <mode>    Set chunking mode: 'spatial' or 'temporal' (default: temporal)\n\
         \x20                      - temporal: last dimension is time [lat, lon, time]\n\
         \x20                      - spatial: first dimension is time [time, lat, lon]\n\
         \n\
         Supports both 2D data (static image) and 3D data (with time slider).\n\
         For 2D data, the chunking mode is ignored.\n\
         \n\
         Examples:\n\
         \x20 {0} weather.om\n\
         \x20 {0} weather.om --variable temperature\n\
         \x20 {0} weather.om --variable group/temperature --chunking spatial",
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
    let mut omfile: Option<String> = None;
    let mut variable_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--chunking" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --chunking requires a value (spatial or temporal)");
                    print_usage_and_exit(&args[0]);
                }
                chunking = ChunkingMode::from_str(&args[i]).unwrap_or_else(|| {
                    eprintln!(
                        "Error: Invalid chunking mode '{}'. Use 'spatial' or 'temporal'.",
                        args[i]
                    );
                    print_usage_and_exit(&args[0]);
                });
            }
            "--variable" | "--var" | "-v" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --variable requires a variable path");
                    print_usage_and_exit(&args[0]);
                }
                variable_path = Some(args[i].clone());
            }
            "--help" | "-h" => {
                print_usage_and_exit(&args[0]);
            }
            s if s.starts_with('-') => {
                eprintln!("Error: Unknown option '{}'", s);
                print_usage_and_exit(&args[0]);
            }
            s if omfile.is_none() => {
                omfile = Some(s.to_string());
            }
            _ => {
                eprintln!("Error: Unexpected argument '{}'", args[i]);
                print_usage_and_exit(&args[0]);
            }
        }
        i += 1;
    }

    let omfile = omfile.unwrap_or_else(|| {
        eprintln!("Error: No input file specified");
        print_usage_and_exit(&args[0]);
    });

    let data_loader = match DataLoader::new(&omfile, variable_path.as_deref(), chunking) {
        Ok(loader) => Arc::new(loader),
        Err(e) => {
            eprintln!("Error: Failed to initialize data loader: {}", e);
            if let Ok(reader) = OmFileReader::from_file(&omfile) {
                let n = reader.number_of_children();
                if n > 0 {
                    eprintln!("\nAvailable variables in '{}':", omfile);
                    navigate::print_children_recursive(&reader, 0);
                } else {
                    let root_name = reader.name();
                    if root_name.is_empty() {
                        eprintln!("  (root is an unnamed array variable, try without --variable)");
                    } else {
                        eprintln!("  Root variable: {}", root_name);
                    }
                }
            }
            std::process::exit(1);
        }
    };

    let title = if let Some(ref var) = variable_path {
        format!("OM Viewer - {} [{}]", omfile, var)
    } else {
        format!("OM Viewer - {}", omfile)
    };

    let native_options = eframe::NativeOptions {
        ..Default::default()
    };

    eframe::run_native(
        &title,
        native_options,
        Box::new(move |_cc| match App::new(data_loader.clone()) {
            Ok(app) => Ok(Box::new(app) as Box<dyn eframe::App>),
            Err(e) => {
                eprintln!("Error: Failed to create application: {}", e);
                std::process::exit(1);
            }
        }),
    )
    .map_err(|e| format!("Failed to run application: {}", e))?;

    Ok(())
}
