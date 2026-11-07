//! Weighted least-squares quadratic fit of implied vol in log-moneyness.
//!
//! ```text
//! σ(k) = a + b·k + c·k²,   k = ln(K / F)
//! ```
//!
//! `a` is the at-the-forward vol, `b` the skew (negative for equity-style
//! put skew) and `c` the curvature of the smile. A quadratic is the simplest
//! shape that captures level, skew and smile; it is not arbitrage-free far
//! outside the quoted strikes, so treat extrapolation with care.

use crate::error::{Error, Result};

/// A fitted quadratic smile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticFit {
    /// At-the-forward volatility, `σ(0)`.
    pub atm_vol: f64,
    /// First derivative `dσ/dk` at the forward.
    pub skew: f64,
    /// Quadratic coefficient `c` (half of `d²σ/dk²`).
    pub curvature: f64,
    /// Weighted root-mean-square residual, in vol units.
    pub rmse: f64,
    /// Number of points used.
    pub points: usize,
}

impl QuadraticFit {
    /// Fitted volatility at log-moneyness `k`.
    pub fn vol_at(&self, k: f64) -> f64 {
        self.atm_vol + k * (self.skew + k * self.curvature)
    }
}

/// Fit `σ(k)` to `(k, σ, weight)` triples. Needs at least three distinct `k`.
///
/// ```
/// use options_lab::smile::fit_quadratic;
/// let pts: Vec<(f64, f64, f64)> = [-0.2, -0.1, 0.0, 0.1, 0.2]
///     .iter()
///     .map(|&k| (k, 0.2 - 0.3 * k + 0.5 * k * k, 1.0))
///     .collect();
/// let fit = fit_quadratic(&pts).unwrap();
/// assert!((fit.skew + 0.3).abs() < 1e-12);
/// ```
pub fn fit_quadratic(points: &[(f64, f64, f64)]) -> Result<QuadraticFit> {
    let used: Vec<_> = points
        .iter()
        .copied()
        .filter(|&(k, v, w)| k.is_finite() && v.is_finite() && w.is_finite() && w > 0.0)
        .collect();
    if used.len() < 3 {
        return Err(Error::parse(
            0,
            format!(
                "need at least 3 implied vols to fit a smile, have {}",
                used.len()
            ),
        ));
    }

    // Normal equations  (Xᵀ W X) β = Xᵀ W y  with X = [1, k, k²].
    let mut m = [[0.0_f64; 4]; 3];
    for &(k, v, w) in &used {
        let x = [1.0, k, k * k];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += w * x[i] * x[j];
            }
            m[i][3] += w * x[i] * v;
        }
    }
    let [a, b, c] = solve3(m).ok_or_else(|| {
        Error::parse(
            0,
            "smile fit is singular: need at least three distinct strikes",
        )
    })?;

    let (mut sse, mut wsum) = (0.0, 0.0);
    for &(k, v, w) in &used {
        let r = v - (a + b * k + c * k * k);
        sse += w * r * r;
        wsum += w;
    }
    Ok(QuadraticFit {
        atm_vol: a,
        skew: b,
        curvature: c,
        rmse: (sse / wsum).sqrt(),
        points: used.len(),
    })
}

/// Gaussian elimination with partial pivoting on a 3x4 augmented matrix.
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    let scale = m.iter().flatten().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[pivot][col].abs() <= 1e-14 * scale {
            return None;
        }
        m.swap(col, pivot);
        let pivot_row = m[col];
        for row in m.iter_mut().skip(col + 1) {
            let f = row[col] / pivot_row[col];
            for (dst, src) in row.iter_mut().zip(pivot_row).skip(col) {
                *dst -= f * src;
            }
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let tail: f64 = (row + 1..3).map(|k| m[row][k] * x[k]).sum();
        x[row] = (m[row][3] - tail) / m[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_exact_quadratic() {
        let pts: Vec<_> = (-10..=10)
            .map(|i| {
                let k = f64::from(i) * 0.03;
                (k, 0.25 - 0.4 * k + 1.2 * k * k, 1.0 + f64::from(i).abs())
            })
            .collect();
        let fit = fit_quadratic(&pts).unwrap();
        assert!((fit.atm_vol - 0.25).abs() < 1e-12);
        assert!((fit.skew + 0.4).abs() < 1e-11);
        assert!((fit.curvature - 1.2).abs() < 1e-10);
        assert!(fit.rmse < 1e-12);
        assert_eq!(fit.points, 21);
    }

    #[test]
    fn weights_pull_the_fit() {
        // Two clusters disagree on the level; the heavier one wins.
        let pts = [
            (-0.1, 0.20, 100.0),
            (0.0, 0.20, 100.0),
            (0.1, 0.20, 100.0),
            (-0.05, 0.30, 1.0),
            (0.05, 0.30, 1.0),
        ];
        let fit = fit_quadratic(&pts).unwrap();
        assert!((fit.atm_vol - 0.2).abs() < 0.01);
    }

    #[test]
    fn needs_three_distinct_points() {
        assert!(fit_quadratic(&[(0.0, 0.2, 1.0), (0.1, 0.2, 1.0)]).is_err());
        assert!(fit_quadratic(&[(0.1, 0.2, 1.0), (0.1, 0.21, 1.0), (0.1, 0.22, 1.0)]).is_err());
    }
}
