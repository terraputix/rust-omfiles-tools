// --- Color Scale Interface ---

pub struct RGBColor(pub u8, pub u8, pub u8);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorMap {
    Viridis,
    Grayscale,
    // Add more color scales here in the future
}

impl ColorMap {
    pub fn evaluate(&self, t: f32) -> RGBColor {
        match self {
            ColorMap::Viridis => viridis_color(t),
            ColorMap::Grayscale => {
                let v = (t.clamp(0.0, 1.0) * 255.0) as u8;
                RGBColor(v, v, v)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalingMode {
    Linear,
    Logarithmic,
    SymLog,
}

pub struct ColorMapper {
    map: ColorMap,
    mode: ScalingMode,
    invert: bool,
    scale_min: f32,
    scale_range: f32,
    is_constant: bool,
}

impl ColorMapper {
    pub fn new(map: ColorMap, mode: ScalingMode, invert: bool, min_val: f32, max_val: f32) -> Self {
        let (scale_min, scale_max) = match mode {
            ScalingMode::Linear => (min_val, max_val),
            ScalingMode::Logarithmic => (min_val.max(1e-5).ln(), max_val.max(1e-5).ln()),
            ScalingMode::SymLog => {
                let symlog = |x: f32| x.signum() * (x.abs() + 1.0).ln();
                (symlog(min_val), symlog(max_val))
            }
        };

        let scale_range = scale_max - scale_min;
        let is_constant = scale_range.abs() < f32::EPSILON;

        Self {
            map,
            mode,
            invert,
            scale_min,
            scale_range: if is_constant { 1.0 } else { scale_range },
            is_constant,
        }
    }

    pub fn map_value(&self, value: f32) -> RGBColor {
        if value.is_nan() {
            return RGBColor(30, 30, 30); // NaN color
        }

        if self.is_constant {
            return self.map.evaluate(0.5);
        }

        let scaled_val = match self.mode {
            ScalingMode::Linear => value,
            ScalingMode::Logarithmic => value.max(1e-5).ln(),
            ScalingMode::SymLog => value.signum() * (value.abs() + 1.0).ln(),
        };

        let mut t = (scaled_val - self.scale_min) / self.scale_range;
        t = t.clamp(0.0, 1.0);

        if self.invert {
            t = 1.0 - t;
        }

        self.map.evaluate(t)
    }
}

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
