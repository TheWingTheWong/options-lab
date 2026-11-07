//! Monte Carlo pricing of European options under geometric Brownian motion.
//!
//! The terminal price is sampled exactly,
//! `S_T = S exp((r - q - σ²/2)T + σ√T Z)`, so there is no time-stepping bias;
//! the only error is statistical, and it is reported as a standard error.
//!
//! Two variance-reduction techniques can be combined:
//!
//! * **Antithetic variates.** Each normal draw `Z` is paired with `-Z`; the
//!   pair average is one sample. The payoff is monotone in `Z`, so the two
//!   halves are negatively correlated and the variance drops.
//! * **Control variate.** The discounted terminal price has a known mean,
//!   `E[S_T] = S e^{(r-q)T}`. With `Y` the payoff and `X = S_T` the
//!   estimator `Ȳ - β(X̄ - E[X])`, `β = Cov(Y, X) / Var(X)`, removes the part of
//!   the noise that is explained by where the underlying ended up. For
//!   in-the-money calls the correlation is close to 1.
//!
//! Random numbers come from ChaCha8 seeded with a `u64`, so results are
//! bit-for-bit reproducible across platforms.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Error, Result};
use crate::types::{OptionParams, OptionType};

/// Monte Carlo pricer configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonteCarlo {
    /// Number of simulated terminal prices (an antithetic pair counts as two).
    pub paths: usize,
    /// RNG seed.
    pub seed: u64,
    /// Pair every draw `Z` with `-Z`.
    pub antithetic: bool,
    /// Use the terminal spot as a control variate.
    pub control_variate: bool,
}

impl Default for MonteCarlo {
    fn default() -> Self {
        MonteCarlo {
            paths: 200_000,
            seed: 42,
            antithetic: true,
            control_variate: true,
        }
    }
}

/// A Monte Carlo price with its statistical error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McEstimate {
    /// Discounted price estimate.
    pub price: f64,
    /// Standard error of `price`.
    pub std_error: f64,
    /// Independent samples behind the estimate (pairs, if antithetic).
    pub samples: usize,
    /// Control-variate coefficient that was applied, if any.
    pub beta: Option<f64>,
}

impl McEstimate {
    /// Normal-approximation 95% confidence interval.
    pub fn confidence_95(&self) -> (f64, f64) {
        let half = 1.959_963_984_540_054 * self.std_error;
        (self.price - half, self.price + half)
    }
}

impl MonteCarlo {
    /// Plain Monte Carlo: no variance reduction.
    pub fn plain(paths: usize, seed: u64) -> Self {
        MonteCarlo {
            paths,
            seed,
            antithetic: false,
            control_variate: false,
        }
    }

    /// Price a European option.
    ///
    /// ```
    /// use options_lab::{bsm, monte_carlo::MonteCarlo, OptionParams, OptionType};
    /// let p = OptionParams::new(100.0, 100.0, 1.0).with_rate(0.05).with_vol(0.2);
    /// let mc = MonteCarlo::default().price(OptionType::Call, &p).unwrap();
    /// let exact = bsm::price(OptionType::Call, &p).unwrap();
    /// assert!((mc.price - exact).abs() < 4.0 * mc.std_error);
    /// ```
    pub fn price(&self, kind: OptionType, p: &OptionParams) -> Result<McEstimate> {
        p.validate()?;
        let min_paths = if self.antithetic { 4 } else { 2 };
        if self.paths < min_paths {
            return Err(Error::invalid(
                "paths",
                self.paths as f64,
                "need at least two independent samples",
            ));
        }

        let df = p.discount();
        let forward = p.forward();
        let drift = (p.rate - p.dividend - 0.5 * p.vol * p.vol) * p.expiry;
        let diffusion = p.vol * p.expiry.sqrt();
        let terminal = |z: f64| p.spot * (drift + diffusion * z).exp();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let n = if self.antithetic {
            self.paths / 2
        } else {
            self.paths
        };
        // Accumulate shifted moments: X is centred on its known mean and Y on
        // the first sample, which keeps the one-pass variance numerically sound.
        let mut acc = Moments::default();
        let mut y_shift = None;
        for _ in 0..n {
            let z: f64 = StandardNormal.sample(&mut rng);
            let (y, x) = if self.antithetic {
                let (s_up, s_dn) = (terminal(z), terminal(-z));
                (
                    0.5 * (kind.payoff(s_up, p.strike) + kind.payoff(s_dn, p.strike)),
                    0.5 * (s_up + s_dn),
                )
            } else {
                let s = terminal(z);
                (kind.payoff(s, p.strike), s)
            };
            let shift = *y_shift.get_or_insert(y);
            acc.push(y - shift, x - forward);
        }
        let shift = y_shift.unwrap_or(0.0);

        let nf = n as f64;
        let mean_y = acc.sy / nf;
        let mean_x = acc.sx / nf;
        let var_y = (acc.syy - nf * mean_y * mean_y) / (nf - 1.0);
        let var_x = (acc.sxx - nf * mean_x * mean_x) / (nf - 1.0);
        let cov = (acc.sxy - nf * mean_x * mean_y) / (nf - 1.0);

        let (estimate, variance, beta) = if self.control_variate && var_x > 0.0 {
            let beta = cov / var_x;
            // mean_x is already X̄ - E[X] because X was centred on the forward.
            let est = mean_y + shift - beta * mean_x;
            let var = var_y - 2.0 * beta * cov + beta * beta * var_x;
            (est, var, Some(beta))
        } else {
            (mean_y + shift, var_y, None)
        };

        Ok(McEstimate {
            price: df * estimate,
            std_error: df * (variance.max(0.0) / nf).sqrt(),
            samples: n,
            beta,
        })
    }
}

#[derive(Default)]
struct Moments {
    sy: f64,
    sx: f64,
    syy: f64,
    sxx: f64,
    sxy: f64,
}

impl Moments {
    #[inline]
    fn push(&mut self, y: f64, x: f64) {
        self.sy += y;
        self.sx += x;
        self.syy += y * y;
        self.sxx += x * x;
        self.sxy += x * y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsm;

    fn cases() -> Vec<OptionParams> {
        vec![
            OptionParams::new(100.0, 100.0, 1.0)
                .with_rate(0.05)
                .with_vol(0.2),
            OptionParams::new(100.0, 120.0, 0.5)
                .with_rate(0.03)
                .with_dividend(0.02)
                .with_vol(0.35),
            OptionParams::new(50.0, 40.0, 2.0)
                .with_rate(0.01)
                .with_dividend(0.03)
                .with_vol(0.5),
        ]
    }

    #[test]
    fn agrees_with_bsm_within_four_standard_errors() {
        for p in cases() {
            for kind in [OptionType::Call, OptionType::Put] {
                for mc in [MonteCarlo::default(), MonteCarlo::plain(200_000, 7)] {
                    let est = mc.price(kind, &p).unwrap();
                    let exact = bsm::price(kind, &p).unwrap();
                    let z = (est.price - exact) / est.std_error;
                    assert!(z.abs() < 4.0, "{kind} {p:?} {mc:?}: z = {z:.2}");
                }
            }
        }
    }

    #[test]
    fn variance_reduction_shrinks_standard_error() {
        let p = cases()[0];
        let plain = MonteCarlo::plain(100_000, 1)
            .price(OptionType::Call, &p)
            .unwrap();
        let reduced = MonteCarlo {
            paths: 100_000,
            seed: 1,
            ..MonteCarlo::default()
        }
        .price(OptionType::Call, &p)
        .unwrap();
        assert!(
            reduced.std_error * 3.0 < plain.std_error,
            "plain {:.5} vs reduced {:.5}",
            plain.std_error,
            reduced.std_error
        );
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let p = cases()[1];
        let a = MonteCarlo::default().price(OptionType::Put, &p).unwrap();
        let b = MonteCarlo::default().price(OptionType::Put, &p).unwrap();
        assert_eq!(a, b);
        let c = MonteCarlo {
            seed: 43,
            ..MonteCarlo::default()
        }
        .price(OptionType::Put, &p)
        .unwrap();
        assert_ne!(a.price, c.price);
    }

    #[test]
    fn degenerate_inputs_are_exact() {
        let p = OptionParams::new(110.0, 100.0, 0.0);
        let est = MonteCarlo::default().price(OptionType::Call, &p).unwrap();
        assert!((est.price - 10.0).abs() < 1e-12);
        assert_eq!(est.std_error, 0.0);
        assert!(MonteCarlo::plain(1, 0).price(OptionType::Call, &p).is_err());
    }
}
