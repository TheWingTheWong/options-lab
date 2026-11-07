//! Special functions used throughout the crate.
//!
//! The standard normal CDF is the workhorse of every closed-form formula here,
//! so it is implemented with a double-precision rational approximation
//! (Hart 1968, as popularised by Graeme West) rather than the common
//! Abramowitz-Stegun 7.1.26 formula, whose ~1e-7 absolute error is far too
//! coarse for implied-volatility work in the wings.

/// `1 / sqrt(2 * pi)`.
pub const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// `sqrt(2 * pi)`.
pub const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Standard normal probability density function.
///
/// ```
/// use options_lab::math::norm_pdf;
/// assert!((norm_pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-15);
/// ```
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal cumulative distribution function.
///
/// Absolute error is below `1e-14` across the real line, and the tails are
/// computed directly (not as `1 - x`), so `norm_cdf(-30.0)` is a meaningful
/// tiny number rather than zero from cancellation.
///
/// ```
/// use options_lab::math::norm_cdf;
/// assert!((norm_cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-14);
/// assert!((norm_cdf(0.0) - 0.5).abs() < 1e-15);
/// ```
pub fn norm_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let z = x.abs();
    // Upper-tail probability Q(z) = 1 - N(z) for z >= 0.
    let tail = if z > 37.0 {
        0.0
    } else {
        let e = (-0.5 * z * z).exp();
        if z < 7.071_067_811_865_47 {
            let num = ((((((3.526_249_659_989_11e-2 * z + 0.700_383_064_443_688) * z
                + 6.373_962_203_531_65)
                * z
                + 33.912_866_078_383)
                * z
                + 112.079_291_497_871)
                * z
                + 221.213_596_169_931)
                * z
                + 220.206_867_912_376)
                * e;
            let den = ((((((8.838_834_764_831_84e-2 * z + 1.755_667_163_182_64) * z
                + 16.064_177_579_207)
                * z
                + 86.780_732_202_946_1)
                * z
                + 296.564_248_779_674)
                * z
                + 637.333_633_378_831)
                * z
                + 793.826_512_519_948)
                * z
                + 440.413_735_824_752;
            num / den
        } else {
            // Continued-fraction expansion of the Mills ratio.
            let cf = z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 0.65))));
            e / cf / SQRT_2PI
        }
    };
    if x > 0.0 { 1.0 - tail } else { tail }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from scipy.stats.norm.cdf (double precision).
    const CDF_REFERENCE: &[(f64, f64)] = &[
        (-38.0, 0.0),
        (-10.0, 7.619_853_024_160_474e-24),
        (-8.0, 6.220_960_574_271_74e-16),
        (-5.0, 2.866_515_718_791_934_4e-7),
        (-3.0, 1.349_898_031_630_093e-3),
        (-1.96, 2.499_789_514_822_043_5e-2),
        (-1.0, 0.158_655_253_931_457_07),
        (-0.5, 0.308_537_538_725_986_9),
        (0.0, 0.5),
        (0.25, 0.598_706_325_682_923_7),
        (1.0, 0.841_344_746_068_542_9),
        (2.326_347_874_040_840_8, 0.99),
        (4.0, 0.999_968_328_758_166_9),
    ];

    #[test]
    fn cdf_matches_reference_values() {
        for &(x, expected) in CDF_REFERENCE {
            let got = norm_cdf(x);
            let tol = 1e-14_f64.max(expected * 1e-10);
            assert!(
                (got - expected).abs() <= tol,
                "N({x}) = {got:e}, expected {expected:e}"
            );
        }
    }

    #[test]
    fn cdf_is_symmetric() {
        for i in -400..=400 {
            let x = f64::from(i) * 0.02;
            let s = norm_cdf(x) + norm_cdf(-x);
            assert!((s - 1.0).abs() < 1e-15, "symmetry broken at {x}: {s}");
        }
    }

    #[test]
    fn cdf_is_monotone() {
        let mut prev = 0.0;
        for i in -1000..=1000 {
            let v = norm_cdf(f64::from(i) * 0.01);
            assert!(v >= prev);
            prev = v;
        }
    }

    #[test]
    fn cdf_derivative_is_pdf() {
        let h = 1e-5;
        for i in -60..=60 {
            let x = f64::from(i) * 0.1;
            let fd = (norm_cdf(x + h) - norm_cdf(x - h)) / (2.0 * h);
            assert!((fd - norm_pdf(x)).abs() < 1e-9, "x = {x}");
        }
    }

    #[test]
    fn nan_propagates() {
        assert!(norm_cdf(f64::NAN).is_nan());
    }
}
