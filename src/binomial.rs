//! Cox-Ross-Rubinstein binomial tree, European or American exercise.
//!
//! Over a step `Δt = T/N` the underlying moves up by `u = e^{σ√Δt}` or down by
//! `d = 1/u`, with risk-neutral up-probability
//!
//! ```text
//! p = (e^{(r-q)Δt} - d) / (u - d)
//! ```
//!
//! Option values are rolled back from expiry; for American exercise each
//! node takes `max(continuation, exercise)`. The tree is recombining, so a
//! single `Vec` of `N + 1` values is reused in place: `O(N²)` time, `O(N)` memory.
//!
//! Delta, gamma and theta are read off the first two time steps of the same
//! tree at no extra cost, which is how practitioners get Greeks for American
//! options where no closed form exists.

use crate::error::{Error, Result};
use crate::types::{ExerciseStyle, OptionParams, OptionType};

/// A CRR tree configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinomialTree {
    /// Number of time steps `N`.
    pub steps: usize,
    /// European or American exercise.
    pub style: ExerciseStyle,
}

/// Price and tree Greeks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeValuation {
    /// Option value at the root.
    pub price: f64,
    /// `∂V/∂S` from the two nodes at step 1.
    pub delta: f64,
    /// `∂²V/∂S²` from the three nodes at step 2.
    pub gamma: f64,
    /// `∂V/∂t` per year, from the root and the middle node at step 2.
    pub theta: f64,
    /// Value of immediate exercise minus continuation at the root; positive
    /// means an American holder should exercise now.
    pub early_exercise_premium: f64,
}

impl BinomialTree {
    /// A tree with `steps` time steps.
    pub fn new(steps: usize, style: ExerciseStyle) -> Self {
        BinomialTree { steps, style }
    }

    /// American-exercise tree.
    pub fn american(steps: usize) -> Self {
        Self::new(steps, ExerciseStyle::American)
    }

    /// European-exercise tree.
    pub fn european(steps: usize) -> Self {
        Self::new(steps, ExerciseStyle::European)
    }

    /// Option value at the root of the tree.
    ///
    /// ```
    /// use options_lab::{binomial::BinomialTree, OptionParams, OptionType};
    /// // Hull, "Options, Futures and Other Derivatives": 5-step American put.
    /// let p = OptionParams::new(50.0, 50.0, 5.0 / 12.0).with_rate(0.10).with_vol(0.40);
    /// let v = BinomialTree::american(5).price(OptionType::Put, &p).unwrap();
    /// assert!((v - 4.49).abs() < 0.005);
    /// ```
    pub fn price(&self, kind: OptionType, p: &OptionParams) -> Result<f64> {
        self.valuation(kind, p).map(|v| v.price)
    }

    /// Price plus delta, gamma and theta from the tree.
    pub fn valuation(&self, kind: OptionType, p: &OptionParams) -> Result<TreeValuation> {
        p.validate()?;
        if self.steps < 2 {
            return Err(Error::invalid(
                "steps",
                self.steps as f64,
                "need at least 2 steps",
            ));
        }
        if p.expiry == 0.0 {
            let payoff = kind.payoff(p.spot, p.strike);
            let itm = payoff > 0.0;
            return Ok(TreeValuation {
                price: payoff,
                delta: if itm { kind.sign() } else { 0.0 },
                gamma: 0.0,
                theta: 0.0,
                early_exercise_premium: 0.0,
            });
        }
        if p.vol <= 0.0 {
            return Err(Error::invalid(
                "vol",
                p.vol,
                "binomial tree needs positive volatility",
            ));
        }

        let n = self.steps;
        let dt = p.expiry / n as f64;
        let u = (p.vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let growth = ((p.rate - p.dividend) * dt).exp();
        let pu = (growth - d) / (u - d);
        if !(pu > 0.0 && pu < 1.0) {
            return Err(Error::invalid(
                "steps",
                n as f64,
                "too few steps: risk-neutral probability falls outside (0, 1)",
            ));
        }
        let disc = (-p.rate * dt).exp();
        let (w_up, w_down) = (disc * pu, disc * (1.0 - pu));
        let u2 = u * u;
        let american = self.style == ExerciseStyle::American;

        // Terminal layer: node j has spot S·u^(2j - N).
        let mut values: Vec<f64> = Vec::with_capacity(n + 1);
        let mut s = p.spot * d.powi(n as i32);
        for _ in 0..=n {
            values.push(kind.payoff(s, p.strike));
            s *= u2;
        }

        let mut step2 = [0.0; 3];
        let mut step1 = [0.0; 2];
        for i in (0..n).rev() {
            let mut s = p.spot * d.powi(i as i32);
            for j in 0..=i {
                let cont = w_up * values[j + 1] + w_down * values[j];
                values[j] = if american {
                    cont.max(kind.payoff(s, p.strike))
                } else {
                    cont
                };
                s *= u2;
            }
            match i {
                2 => step2.copy_from_slice(&values[..3]),
                1 => step1.copy_from_slice(&values[..2]),
                _ => {}
            }
        }

        let price = values[0];
        let (s_u, s_d) = (p.spot * u, p.spot * d);
        let (s_uu, s_dd) = (p.spot * u2, p.spot * d * d);
        let delta = (step1[1] - step1[0]) / (s_u - s_d);
        let delta_up = (step2[2] - step2[1]) / (s_uu - p.spot);
        let delta_down = (step2[1] - step2[0]) / (p.spot - s_dd);
        let gamma = (delta_up - delta_down) / (0.5 * (s_uu - s_dd));
        let theta = (step2[1] - price) / (2.0 * dt);

        let continuation = w_up * step1[1] + w_down * step1[0];
        Ok(TreeValuation {
            price,
            delta,
            gamma,
            theta,
            early_exercise_premium: kind.payoff(p.spot, p.strike) - continuation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsm;

    fn hull_put() -> OptionParams {
        OptionParams::new(50.0, 50.0, 5.0 / 12.0)
            .with_rate(0.10)
            .with_vol(0.40)
    }

    #[test]
    fn matches_hull_american_put() {
        // Reference values from an independent Python implementation.
        let p = hull_put();
        let five = BinomialTree::american(5)
            .price(OptionType::Put, &p)
            .unwrap();
        assert!((five - 4.488_458_534_725_914).abs() < 1e-12, "{five}");
        let fine = BinomialTree::american(500)
            .price(OptionType::Put, &p)
            .unwrap();
        assert!((fine - 4.283_021_276_450_367).abs() < 1e-10, "{fine}");
    }

    #[test]
    fn european_tree_converges_to_bsm() {
        let p = OptionParams::new(100.0, 95.0, 0.75)
            .with_rate(0.04)
            .with_dividend(0.02)
            .with_vol(0.3);
        for kind in [OptionType::Call, OptionType::Put] {
            let exact = bsm::price(kind, &p).unwrap();
            // CRR oscillates between odd and even N; averaging cancels most of it.
            let a = BinomialTree::european(2000).price(kind, &p).unwrap();
            let b = BinomialTree::european(2001).price(kind, &p).unwrap();
            assert!(
                (0.5 * (a + b) - exact).abs() < 5e-4,
                "{kind}: {a} {b} vs {exact}"
            );
        }
    }

    #[test]
    fn american_call_without_dividends_is_european() {
        let p = OptionParams::new(100.0, 100.0, 1.0)
            .with_rate(0.05)
            .with_vol(0.25);
        let am = BinomialTree::american(400)
            .price(OptionType::Call, &p)
            .unwrap();
        let eu = BinomialTree::european(400)
            .price(OptionType::Call, &p)
            .unwrap();
        assert!((am - eu).abs() < 1e-12);
    }

    #[test]
    fn early_exercise_premium_is_non_negative() {
        let p = OptionParams::new(100.0, 110.0, 1.0)
            .with_rate(0.06)
            .with_dividend(0.04)
            .with_vol(0.2);
        for kind in [OptionType::Call, OptionType::Put] {
            let am = BinomialTree::american(300).price(kind, &p).unwrap();
            let eu = BinomialTree::european(300).price(kind, &p).unwrap();
            assert!(am >= eu - 1e-12, "{kind}");
        }
        let deep = p.with_dividend(0.0).with_vol(0.05);
        let am = BinomialTree::american(300)
            .price(OptionType::Put, &deep)
            .unwrap();
        let eu = BinomialTree::european(300)
            .price(OptionType::Put, &deep)
            .unwrap();
        assert!(am > eu + 0.1, "deep ITM put should carry exercise value");
    }

    #[test]
    fn tree_greeks_track_bsm() {
        let p = OptionParams::new(100.0, 100.0, 0.5)
            .with_rate(0.03)
            .with_vol(0.2);
        let tree = BinomialTree::european(2000)
            .valuation(OptionType::Call, &p)
            .unwrap();
        let g = bsm::greeks(OptionType::Call, &p).unwrap();
        assert!((tree.delta - g.delta).abs() < 1e-3);
        assert!((tree.gamma - g.gamma).abs() < 1e-3);
        assert!((tree.theta - g.theta).abs() < 2e-2);
    }

    #[test]
    fn deep_itm_american_put_signals_exercise() {
        let p = OptionParams::new(40.0, 100.0, 1.0)
            .with_rate(0.05)
            .with_vol(0.2);
        let v = BinomialTree::american(200)
            .valuation(OptionType::Put, &p)
            .unwrap();
        assert!((v.price - 60.0).abs() < 1e-9);
        assert!(v.early_exercise_premium > 0.0);
    }

    #[test]
    fn rejects_degenerate_trees() {
        let p = OptionParams::new(100.0, 100.0, 1.0)
            .with_rate(2.0)
            .with_vol(0.01);
        assert!(
            BinomialTree::american(2)
                .price(OptionType::Put, &p)
                .is_err()
        );
        assert!(
            BinomialTree::american(1)
                .price(OptionType::Put, &hull_put())
                .is_err()
        );
        assert!(
            BinomialTree::american(10)
                .price(OptionType::Put, &hull_put().with_vol(0.0))
                .is_err()
        );
    }
}
