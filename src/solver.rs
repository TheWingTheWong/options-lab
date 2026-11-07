//! One-dimensional root finding.
//!
//! [`brent`] is a faithful implementation of Brent's (1973) method: inverse
//! quadratic interpolation and secant steps, guarded by bisection so the
//! bracket always shrinks. It is guaranteed to converge for any continuous
//! function whose endpoints have opposite signs, which is why the implied
//! volatility solver falls back to it when Newton misbehaves.

use crate::error::{Error, Result};

/// Result of a successful root search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Root {
    /// Abscissa of the root.
    pub x: f64,
    /// Function value at `x`.
    pub fx: f64,
    /// Number of function evaluations performed (excluding the two endpoints).
    pub iterations: usize,
}

/// Find a root of `f` in `[a, b]` with Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs (or one of them to be
/// zero). Terminates when the bracket is narrower than `2 * (4ε|x| + xtol / 2)`
/// or `f(x)` is exactly zero.
///
/// ```
/// use options_lab::solver::brent;
/// let r = brent(|x| x * x - 2.0, 0.0, 2.0, 1e-14, 100).unwrap();
/// assert!((r.x - 2f64.sqrt()).abs() < 1e-13);
/// ```
pub fn brent<F>(mut f: F, a: f64, b: f64, xtol: f64, max_iter: usize) -> Result<Root>
where
    F: FnMut(f64) -> f64,
{
    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    if fa == 0.0 {
        return Ok(Root {
            x: a,
            fx: fa,
            iterations: 0,
        });
    }
    if fb == 0.0 {
        return Ok(Root {
            x: b,
            fx: fb,
            iterations: 0,
        });
    }
    if fa.is_nan() || fb.is_nan() || fa.signum() == fb.signum() {
        return Err(Error::InvalidBracket { fa, fb });
    }

    // `b` is the best estimate, `a` the previous one, `c` the contrapoint
    // such that f(b) and f(c) bracket the root.
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;

    for iter in 1..=max_iter {
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * xtol;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return Ok(Root {
                x: b,
                fx: fb,
                iterations: iter - 1,
            });
        }

        if e.abs() >= tol && fa.abs() > fb.abs() {
            // Attempt interpolation.
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                // Secant.
                (2.0 * m * s, 1.0 - s)
            } else {
                // Inverse quadratic interpolation.
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            // Accept only if the step stays well inside the bracket and is
            // shrinking fast enough; otherwise bisect.
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }

        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b);
        if fb.is_nan() {
            return Err(Error::NoConvergence {
                method: "brent",
                iterations: iter,
            });
        }
    }

    Err(Error::NoConvergence {
        method: "brent",
        iterations: max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_polynomial_root() {
        let r = brent(|x| (x - 1.0) * (x + 2.0) * (x - 3.5), 2.0, 5.0, 1e-15, 100).unwrap();
        assert!((r.x - 3.5).abs() < 1e-13);
    }

    #[test]
    fn handles_flat_and_steep_functions() {
        // Very flat near the root: x^9.
        let r = brent(|x: f64| x.powi(9), -1.0, 3.0, 1e-12, 200).unwrap();
        assert!(r.x.abs() < 1e-2, "{}", r.x);
        // Discontinuous sign change behaves like bisection.
        let r = brent(|x| if x < 0.3 { -1.0 } else { 1.0 }, 0.0, 1.0, 1e-12, 200).unwrap();
        assert!((r.x - 0.3).abs() < 1e-11);
    }

    #[test]
    fn rejects_non_bracketing_interval() {
        let err = brent(|x| x * x + 1.0, -1.0, 1.0, 1e-12, 50).unwrap_err();
        assert!(matches!(err, Error::InvalidBracket { .. }));
    }

    #[test]
    fn converges_quickly_on_smooth_functions() {
        let r = brent(|x: f64| x.cos() - x, 0.0, 1.0, 1e-15, 100).unwrap();
        assert!((r.x - 0.739_085_133_215_160_6).abs() < 1e-14);
        assert!(r.iterations < 12, "took {} iterations", r.iterations);
    }
}
