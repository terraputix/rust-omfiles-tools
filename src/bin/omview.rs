use eframe::egui::{self, CentralPanel, TopBottomPanel};
use omfiles::MmapFile;
use omfiles::reader::OmFileReader;
use omfiles::traits::OmArrayVariable;
use omfiles::traits::OmFileVariable;
use omfiles_tools::colorscales::{ColorMap, ColorMapper, ScalingMode};
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

/// Represents the visible viewport in data coordinates.
///
/// Data coordinate system:
/// - x: 0 = left column, data_cols = right edge
/// - y: 0 = bottom row (data row 0), data_rows = top edge (data row N-1)
///
/// The image is rendered with data row 0 at the bottom (y-flipped), so:
/// - Screen top corresponds to y_max (high data row index)
/// - Screen bottom corresponds to y_min (low data row index)
#[derive(Clone, Debug)]
struct Viewport {
    x_min: f64,
    x_max: f64,
    /// Bottom edge in data y coords (low row index)
    y_min: f64,
    /// Top edge in data y coords (high row index)
    y_max: f64,
    data_cols: f64,
    data_rows: f64,
}

impl Viewport {
    fn new(rows: u64, cols: u64) -> Self {
        Self {
            x_min: 0.0,
            x_max: cols as f64,
            y_min: 0.0,
            y_max: rows as f64,
            data_cols: cols as f64,
            data_rows: rows as f64,
        }
    }

    fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    fn zoom_level(&self) -> f64 {
        self.data_cols / self.width()
    }

    /// Zoom by a factor centered on a point in data coordinates.
    fn zoom(&mut self, factor: f64, center_x: f64, center_y: f64) {
        let new_width = (self.width() / factor).max(1.0).min(self.data_cols);
        let new_height = (self.height() / factor).max(1.0).min(self.data_rows);

        let fx = (center_x - self.x_min) / self.width();
        let fy = (center_y - self.y_min) / self.height();

        self.x_min = center_x - fx * new_width;
        self.x_max = center_x + (1.0 - fx) * new_width;
        self.y_min = center_y - fy * new_height;
        self.y_max = center_y + (1.0 - fy) * new_height;

        self.clamp();
    }

    /// Pan by delta in data coordinates.
    fn pan(&mut self, dx: f64, dy: f64) {
        self.x_min += dx;
        self.x_max += dx;
        self.y_min += dy;
        self.y_max += dy;
        self.clamp();
    }

    fn reset(&mut self) {
        self.x_min = 0.0;
        self.x_max = self.data_cols;
        self.y_min = 0.0;
        self.y_max = self.data_rows;
    }

    fn clamp(&mut self) {
        let w = self.width();
        let h = self.height();

        if self.x_min < 0.0 {
            self.x_min = 0.0;
            self.x_max = w.min(self.data_cols);
        }
        if self.x_max > self.data_cols {
            self.x_max = self.data_cols;
            self.x_min = (self.data_cols - w).max(0.0);
        }
        if self.y_min < 0.0 {
            self.y_min = 0.0;
            self.y_max = h.min(self.data_rows);
        }
        if self.y_max > self.data_rows {
            self.y_max = self.data_rows;
            self.y_min = (self.data_rows - h).max(0.0);
        }
    }

    /// Convert a screen position within the image rect to data coordinates.
    ///
    /// Screen x maps linearly to data x: left → x_min, right → x_max.
    /// Screen y is INVERTED: top → y_max (high row), bottom → y_min (low row).
    /// This matches the image rendering where row 0 is at the bottom.
    fn screen_to_data(&self, pos: egui::Pos2, rect: egui::Rect) -> (f64, f64) {
        let fx = ((pos.x - rect.left()) / rect.width()) as f64;
        let fy = ((pos.y - rect.top()) / rect.height()) as f64;
        let data_x = self.x_min + fx * self.width();
        // Invert y: screen top (fy=0) → y_max, screen bottom (fy=1) → y_min
        let data_y = self.y_max - fy * self.height();
        (data_x, data_y)
    }

    /// Convert a screen drag delta to data coordinate deltas.
    ///
    /// Positive screen dx (drag right) should move viewport right → positive data dx.
    /// Positive screen dy (drag down) should move viewport down → negative data dy (lower row indices).
    fn screen_delta_to_data(&self, delta: egui::Vec2, rect: egui::Rect) -> (f64, f64) {
        let dx = (delta.x as f64) * self.width() / rect.width() as f64;
        // Invert y for the same reason as screen_to_data
        let dy = -(delta.y as f64) * self.height() / rect.height() as f64;
        (dx, dy)
    }
}

struct App {
    data_loader: Arc<DataLoader>,
    current_timestamp: u64,
    plot_data: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Ix2>,
    error_message: Option<String>,
    viewport: Viewport,

    // Color Scale Settings
    color_map: ColorMap,
    scaling_mode: ScalingMode,
    invert_color_scale: bool,
}

impl App {
    fn new(data_loader: Arc<DataLoader>) -> Result<Self, Box<dyn std::error::Error>> {
        let initial_data = data_loader.get_timestamp_data(0)?;
        let viewport = Viewport::new(data_loader.rows(), data_loader.cols());

        Ok(Self {
            data_loader,
            current_timestamp: 0,
            plot_data: initial_data,
            error_message: None,
            viewport,
            color_map: ColorMap::Viridis,
            scaling_mode: ScalingMode::Linear,
            invert_color_scale: false,
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
                    "{}Shape: {}×{} | {:?} | Zoom: {:.1}x",
                    var_label,
                    self.data_loader.rows(),
                    self.data_loader.cols(),
                    self.data_loader.chunking,
                    self.viewport.zoom_level(),
                ));
                if ui.button("Reset Zoom").clicked() {
                    self.viewport.reset();
                }
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

        TopBottomPanel::top("settings").show(ctx, |ui| {
            ui.horizontal(|ui| {
                egui::ComboBox::from_label("Colormap")
                    .selected_text(format!("{:?}", self.color_map))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.color_map, ColorMap::Viridis, "Viridis");
                        ui.selectable_value(&mut self.color_map, ColorMap::Grayscale, "Grayscale");
                    });

                egui::ComboBox::from_label("Scaling")
                    .selected_text(format!("{:?}", self.scaling_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.scaling_mode, ScalingMode::Linear, "Linear");
                        ui.selectable_value(
                            &mut self.scaling_mode,
                            ScalingMode::Logarithmic,
                            "Logarithmic",
                        );
                        ui.selectable_value(&mut self.scaling_mode, ScalingMode::SymLog, "Sym-Log");
                    });

                ui.checkbox(&mut self.invert_color_scale, "Invert");
            })
        });

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

            let mapper = ColorMapper::new(
                self.color_map,
                self.scaling_mode,
                self.invert_color_scale,
                min_value,
                max_value,
            );

            let (full_rows, full_cols) = self.plot_data.dim();

            let vp = &self.viewport;
            let x_start = (vp.x_min.floor() as usize).min(full_cols.saturating_sub(1));
            let x_end = (vp.x_max.ceil() as usize).min(full_cols);
            let y_start = (vp.y_min.floor() as usize).min(full_rows.saturating_sub(1));
            let y_end = (vp.y_max.ceil() as usize).min(full_rows);

            let view_cols = x_end - x_start;
            let view_rows = y_end - y_start;

            if view_cols == 0 || view_rows == 0 {
                return;
            }

            let mut rgba_data = Vec::with_capacity(view_rows * view_cols * 4);
            for y in (y_start..y_end).rev() {
                for x in x_start..x_end {
                    let value = self.plot_data[[y, x]];
                    let color = mapper.map_value(value);
                    rgba_data.push(color.0);
                    rgba_data.push(color.1);
                    rgba_data.push(color.2);
                    rgba_data.push(255);
                }
            }

            let image =
                egui::ColorImage::from_rgba_unmultiplied([view_cols, view_rows], &rgba_data);
            let texture = ui
                .ctx()
                .load_texture("heatmap", image, egui::TextureOptions::NEAREST);

            let available = ui.available_size();
            let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

            ui.painter().image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            if response.hovered() {
                let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = if scroll_delta > 0.0 { 1.15 } else { 1.0 / 1.15 };
                    if let Some(pointer_pos) = ui.ctx().pointer_hover_pos() {
                        let (data_x, data_y) = self.viewport.screen_to_data(pointer_pos, rect);
                        self.viewport.zoom(zoom_factor, data_x, data_y);
                        ctx.request_repaint();
                    }
                }
            }

            if response.dragged() {
                let delta = response.drag_delta();
                let (dx, dy) = self.viewport.screen_delta_to_data(delta, rect);
                self.viewport.pan(-dx, -dy);
                ctx.request_repaint();
            }

            if response.double_clicked() {
                self.viewport.reset();
            }

            if response.hovered() {
                if let Some(pointer_pos) = ui.ctx().pointer_hover_pos() {
                    let (data_x, data_y) = self.viewport.screen_to_data(pointer_pos, rect);
                    let x = data_x.floor() as isize;
                    let y = data_y.floor() as isize;

                    if x >= 0 && y >= 0 && (x as usize) < full_cols && (y as usize) < full_rows {
                        let xu = x as usize;
                        let yu = y as usize;
                        let value = self.plot_data[[yu, xu]];
                        ui.ctx().output_mut(|o| {
                            o.cursor_icon = egui::CursorIcon::Crosshair;
                        });
                        response.on_hover_ui(|ui| {
                            ui.label(format!("({}, {}): {:.4}", xu, yu, value));
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
         Controls:\n\
         \x20 Scroll wheel         Zoom in/out (centered on cursor)\n\
         \x20 Click + Drag         Pan the view\n\
         \x20 Double-click         Reset zoom to fit all data\n\
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
