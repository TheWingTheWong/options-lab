//! Plain-ASCII smile chart for terminals, logs and READMEs.
//!
//! Strike runs along the x axis, implied vol (in %) up the y axis. Axis ticks
//! are snapped to round values so the gridlines land on exact rows and
//! columns. Market points are drawn over the fitted curve:
//!
//! * `o` put implied vol, `x` call implied vol, `*` both in the same cell
//! * `.` fitted quadratic smile
//! * `:` the forward

use super::Smile;
use crate::types::OptionType;

/// Chart dimensions, in characters, of the plotting area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChartOptions {
    /// Plot width (columns), excluding the y-axis labels.
    pub width: usize,
    /// Target plot height (rows). The actual height is adjusted so that y
    /// ticks fall on whole rows.
    pub height: usize,
}

impl Default for ChartOptions {
    fn default() -> Self {
        ChartOptions {
            width: 64,
            height: 16,
        }
    }
}

/// Render the smile as a multi-line string. Returns an explanatory line
/// instead of a chart when there is nothing to plot.
pub fn render_chart(smile: &Smile, opts: &ChartOptions) -> String {
    let width = opts.width.max(20);
    if smile.points.len() < 2 {
        return "  (need at least two implied vols to draw a smile)\n".into();
    }

    let x_min = smile.points.first().map(|p| p.quote.strike).unwrap_or(0.0);
    let x_max = smile.points.last().map(|p| p.quote.strike).unwrap_or(1.0);
    if x_max <= x_min {
        return "  (all quotes share one strike; nothing to draw)\n".into();
    }
    let col_of = |x: f64| ((x - x_min) / (x_max - x_min) * (width - 1) as f64).round() as usize;
    let strike_at = |col: usize| x_min + (x_max - x_min) * col as f64 / (width - 1) as f64;

    // Fitted curve sampled once per column.
    let curve: Option<Vec<f64>> = smile.fit.map(|fit| {
        (0..width)
            .map(|c| 100.0 * fit.vol_at((strike_at(c) / smile.forward).ln()))
            .collect()
    });

    // Y range covers the market points and the fitted curve, snapped to ticks.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in smile
        .points
        .iter()
        .map(|p| 100.0 * p.iv)
        .chain(curve.iter().flatten().copied())
    {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let (y_step, y_lo, y_hi) = nice_range(lo, hi, 6, VOL_STEPS);
    let intervals = ((y_hi - y_lo) / y_step).round().max(1.0) as usize;
    let rows_per = (opts.height.saturating_sub(1) as f64 / intervals as f64)
        .round()
        .max(1.0) as usize;
    let height = intervals * rows_per + 1;
    let row_of = |v: f64| {
        let r = ((y_hi - v) / (y_hi - y_lo) * (height - 1) as f64).round();
        r.clamp(0.0, (height - 1) as f64) as usize
    };

    let mut grid = vec![vec![' '; width]; height];

    // Forward marker.
    let fwd_col = (smile.forward >= x_min && smile.forward <= x_max).then(|| col_of(smile.forward));
    if let Some(c) = fwd_col {
        for row in grid.iter_mut() {
            row[c] = ':';
        }
    }

    // Fitted curve, with vertical fill on steep stretches so it reads as a line.
    if let Some(curve) = &curve {
        let mut prev: Option<usize> = None;
        for (c, &v) in curve.iter().enumerate() {
            let r = row_of(v);
            if let Some(p) = prev {
                let (a, b) = if p < r {
                    (p + 1, r)
                } else {
                    (r, p.saturating_sub(1))
                };
                for row in grid.iter_mut().take(b + 1).skip(a) {
                    row[c] = '.';
                }
            }
            grid[r][c] = '.';
            prev = Some(r);
        }
    }

    // Market points on top.
    for p in &smile.points {
        let (r, c) = (row_of(100.0 * p.iv), col_of(p.quote.strike));
        let mark = match p.quote.kind {
            OptionType::Put => 'o',
            OptionType::Call => 'x',
        };
        let cell = &mut grid[r][c];
        *cell = match *cell {
            'o' | 'x' | '*' if *cell != mark => '*',
            _ => mark,
        };
    }

    // Assemble with axes.
    let decimals = (0..3)
        .find(|&d| {
            let scaled = y_step * 10f64.powi(d);
            (scaled - scaled.round()).abs() < 1e-9
        })
        .unwrap_or(3) as usize;
    let y_label = |v: f64| format!("{v:.decimals$}");
    let label_width = y_label(y_hi).len().max(y_label(y_lo).len());
    let indent = " ".repeat(label_width + 3);

    let mut out = String::new();
    out.push_str(&format!(
        "{indent}implied vol %     o put   x call   . fit   : forward {:.2}\n",
        smile.forward
    ));
    for (r, row) in grid.iter().enumerate() {
        let line: String = row.iter().collect();
        if r % rows_per == 0 {
            let v = y_hi - (r / rows_per) as f64 * y_step;
            out.push_str(&format!(
                "  {:>label_width$} +{}\n",
                y_label(v),
                line.trim_end()
            ));
        } else {
            out.push_str(&format!("  {:>label_width$} |{}\n", "", line.trim_end()));
        }
    }

    // X axis with snapped ticks.
    let (x_step, _, _) = nice_range(x_min, x_max, (width / 10).max(2), STRIKE_STEPS);
    let first_tick = (x_min / x_step).ceil() * x_step;
    let mut axis = vec!['-'; width];
    let mut labels = vec![' '; width + 8];
    let mut next_free = 0usize;
    let mut t = first_tick;
    while t <= x_max + 1e-9 * x_step {
        let c = col_of(t);
        if c > 0 {
            // Column 0 sits right of the axis corner, which already marks it.
            axis[c] = '+';
        }
        let text = trim_number(t);
        let start = c.saturating_sub(text.len() / 2);
        if start >= next_free && start + text.len() <= labels.len() {
            for (i, ch) in text.chars().enumerate() {
                labels[start + i] = ch;
            }
            next_free = start + text.len() + 1;
        }
        t += x_step;
    }
    out.push_str(&format!(
        "  {:>label_width$} +{}\n",
        "",
        axis.iter().collect::<String>()
    ));
    out.push_str(&format!(
        "  {:>label_width$}  {}\n",
        "",
        labels.iter().collect::<String>().trim_end()
    ));
    out.push_str(format!("{indent}{:^width$}", "strike").trim_end());
    out.push('\n');
    out
}

/// Tick multipliers for the vol axis: whole vol points read best.
const VOL_STEPS: &[f64] = &[1.0, 2.0, 5.0, 10.0, 20.0];
/// Tick multipliers for strikes: listed strikes are often 2.5 apart.
const STRIKE_STEPS: &[f64] = &[1.0, 2.0, 2.5, 5.0, 10.0, 20.0];

/// Pick a "nice" step (one of `multipliers` times a power of ten) that splits
/// `[lo, hi]` into at most `max_intervals`, and snap the range outward to it.
fn nice_range(lo: f64, hi: f64, max_intervals: usize, multipliers: &[f64]) -> (f64, f64, f64) {
    let span = (hi - lo).max(1e-9 * hi.abs().max(1.0));
    let raw = span / max_intervals.max(1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let snap = |step: f64| ((lo / step).floor() * step, (hi / step).ceil() * step);
    let step = multipliers
        .iter()
        .map(|m| m * mag)
        .find(|&s| {
            let (a, b) = snap(s);
            (b - a) / s <= max_intervals as f64 + 1e-9
        })
        .unwrap_or(20.0 * mag);
    let (lo_snap, mut hi_snap) = snap(step);
    if hi_snap <= lo_snap {
        hi_snap = lo_snap + step;
    }
    (step, lo_snap, hi_snap)
}

fn trim_number(x: f64) -> String {
    let s = format!("{x:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smile::{Chain, Market, Quote, Selection, build_smile};
    use crate::{OptionParams, bsm};

    fn smile() -> Smile {
        let m = Market {
            spot: 100.0,
            rate: 0.02,
            dividend: 0.0,
            expiry: 0.25,
        };
        let quotes = (80..=120)
            .step_by(5)
            .map(|k| {
                let k = f64::from(k);
                let p = OptionParams::new(100.0, k, 0.25).with_rate(0.02);
                let lm = p.log_moneyness();
                let vol = 0.2 - 0.3 * lm + 1.0 * lm * lm;
                let kind = if k < p.forward() {
                    OptionType::Put
                } else {
                    OptionType::Call
                };
                Quote {
                    strike: k,
                    kind,
                    bid: None,
                    ask: None,
                    mid: bsm::price(kind, &p.with_vol(vol)).unwrap(),
                    line: 0,
                }
            })
            .collect();
        build_smile(
            &Chain {
                quotes,
                ..Chain::default()
            },
            m,
            Selection::OutOfTheMoney,
        )
        .unwrap()
    }

    #[test]
    fn nice_ranges_are_round() {
        assert_eq!(nice_range(17.3, 31.9, 6, VOL_STEPS), (5.0, 15.0, 35.0));
        assert_eq!(nice_range(90.0, 105.0, 6, STRIKE_STEPS), (2.5, 90.0, 105.0));
        assert_eq!(
            nice_range(80.0, 120.0, 6, STRIKE_STEPS),
            (10.0, 80.0, 120.0)
        );
        let (step, lo, hi) = nice_range(19.6, 20.4, 6, VOL_STEPS);
        assert_eq!(step, 0.2);
        assert!((lo - 19.6).abs() < 1e-9 && (hi - 20.4).abs() < 1e-9);
    }

    #[test]
    fn chart_contains_points_curve_and_axes() {
        let s = smile();
        let chart = render_chart(&s, &ChartOptions::default());
        assert!(chart.contains('o') && chart.contains('x'));
        assert!(chart.contains('.'));
        assert!(chart.contains(':'));
        assert!(chart.contains("strike"));
        for label in ["80", "100", "120"] {
            assert!(chart.contains(label), "missing x label {label}\n{chart}");
        }
        // Every plot row is the same structure: label column then axis.
        let widths: Vec<_> = chart
            .lines()
            .filter(|l| l.contains(" |") || l.contains(" +"))
            .map(|l| l.find(['|', '+']).unwrap())
            .collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{chart}");
    }

    #[test]
    fn puts_sit_left_of_calls() {
        let chart = render_chart(&smile(), &ChartOptions::default());
        let plot = || chart.lines().filter(|l| !l.contains("implied vol"));
        let last_o = plot().filter_map(|l| l.rfind('o')).max().unwrap();
        let first_x = plot().filter_map(|l| l.find('x')).min().unwrap();
        assert!(last_o < first_x, "{chart}");
    }
}
