//! Core value types: option kind, exercise style and model inputs.

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

/// Calendar convention used to convert days to expiry into year fractions.
pub const DAYS_PER_YEAR: f64 = 365.0;

/// Call or put.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    /// Right to buy the underlying at the strike.
    Call,
    /// Right to sell the underlying at the strike.
    Put,
}

impl OptionType {
    /// `+1` for calls, `-1` for puts. Lets most formulas be written once.
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }

    /// Exercise value `max(φ(S - K), 0)` where `φ` is [`OptionType::sign`].
    ///
    /// ```
    /// use options_lab::OptionType;
    /// assert_eq!(OptionType::Call.payoff(110.0, 100.0), 10.0);
    /// assert_eq!(OptionType::Put.payoff(110.0, 100.0), 0.0);
    /// ```
    #[inline]
    pub fn payoff(self, spot: f64, strike: f64) -> f64 {
        (self.sign() * (spot - strike)).max(0.0)
    }

    /// The other side: calls become puts and vice versa.
    pub fn flip(self) -> Self {
        match self {
            OptionType::Call => OptionType::Put,
            OptionType::Put => OptionType::Call,
        }
    }
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OptionType::Call => "call",
            OptionType::Put => "put",
        })
    }
}

impl FromStr for OptionType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "c" | "call" | "calls" => Ok(OptionType::Call),
            "p" | "put" | "puts" => Ok(OptionType::Put),
            other => Err(Error::parse(
                0,
                format!("unknown option type '{other}' (expected call or put)"),
            )),
        }
    }
}

/// When the holder may exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExerciseStyle {
    /// Only at expiry.
    #[default]
    European,
    /// At any time up to and including expiry.
    American,
}

impl fmt::Display for ExerciseStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ExerciseStyle::European => "European",
            ExerciseStyle::American => "American",
        })
    }
}

/// Market and contract inputs shared by every pricing model.
///
/// All rates are continuously compounded and annualised; `expiry` is in years.
///
/// ```
/// use options_lab::OptionParams;
/// let p = OptionParams::new(100.0, 105.0, 30.0 / 365.0)
///     .with_vol(0.2)
///     .with_rate(0.03);
/// assert!(p.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionParams {
    /// Spot price of the underlying, `S`.
    pub spot: f64,
    /// Strike price, `K`.
    pub strike: f64,
    /// Risk-free rate, `r`.
    pub rate: f64,
    /// Continuous dividend (or carry / foreign rate) yield, `q`.
    pub dividend: f64,
    /// Volatility of log returns, `σ`.
    pub vol: f64,
    /// Time to expiry in years, `T`.
    pub expiry: f64,
}

impl OptionParams {
    /// New parameters with zero rates and a placeholder 20% volatility.
    pub fn new(spot: f64, strike: f64, expiry: f64) -> Self {
        OptionParams {
            spot,
            strike,
            rate: 0.0,
            dividend: 0.0,
            vol: 0.2,
            expiry,
        }
    }

    /// Set the risk-free rate.
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate;
        self
    }

    /// Set the continuous dividend yield.
    pub fn with_dividend(mut self, dividend: f64) -> Self {
        self.dividend = dividend;
        self
    }

    /// Set the volatility.
    pub fn with_vol(mut self, vol: f64) -> Self {
        self.vol = vol;
        self
    }

    /// Set the time to expiry in years.
    pub fn with_expiry(mut self, expiry: f64) -> Self {
        self.expiry = expiry;
        self
    }

    /// Check that every input is finite and inside its domain.
    pub fn validate(&self) -> Result<()> {
        check_positive("spot", self.spot)?;
        check_positive("strike", self.strike)?;
        check_finite("rate", self.rate)?;
        check_finite("dividend", self.dividend)?;
        check_finite("vol", self.vol)?;
        if self.vol < 0.0 {
            return Err(Error::invalid("vol", self.vol, "must be non-negative"));
        }
        check_finite("expiry", self.expiry)?;
        if self.expiry < 0.0 {
            return Err(Error::invalid(
                "expiry",
                self.expiry,
                "must be non-negative",
            ));
        }
        Ok(())
    }

    /// Discount factor `e^{-rT}`.
    #[inline]
    pub fn discount(&self) -> f64 {
        (-self.rate * self.expiry).exp()
    }

    /// Dividend discount factor `e^{-qT}`.
    #[inline]
    pub fn dividend_discount(&self) -> f64 {
        (-self.dividend * self.expiry).exp()
    }

    /// Forward price `F = S e^{(r - q)T}`.
    #[inline]
    pub fn forward(&self) -> f64 {
        self.spot * ((self.rate - self.dividend) * self.expiry).exp()
    }

    /// Log-moneyness relative to the forward, `ln(K / F)`.
    #[inline]
    pub fn log_moneyness(&self) -> f64 {
        (self.strike / self.forward()).ln()
    }
}

fn check_finite(field: &'static str, v: f64) -> Result<()> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid(field, v, "must be finite"))
    }
}

fn check_positive(field: &'static str, v: f64) -> Result<()> {
    check_finite(field, v)?;
    if v > 0.0 {
        Ok(())
    } else {
        Err(Error::invalid(field, v, "must be positive"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_option_types() {
        assert_eq!("Call".parse::<OptionType>().unwrap(), OptionType::Call);
        assert_eq!(" p ".parse::<OptionType>().unwrap(), OptionType::Put);
        assert!("straddle".parse::<OptionType>().is_err());
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let ok = OptionParams::new(100.0, 100.0, 1.0);
        assert!(ok.validate().is_ok());
        assert!(ok.with_vol(-0.1).validate().is_err());
        assert!(ok.with_expiry(-1.0).validate().is_err());
        assert!(OptionParams::new(0.0, 100.0, 1.0).validate().is_err());
        assert!(OptionParams::new(100.0, f64::NAN, 1.0).validate().is_err());
        assert!(ok.with_rate(f64::INFINITY).validate().is_err());
    }

    #[test]
    fn forward_and_moneyness() {
        let p = OptionParams::new(100.0, 100.0, 2.0)
            .with_rate(0.05)
            .with_dividend(0.02);
        assert!((p.forward() - 100.0 * (0.06_f64).exp()).abs() < 1e-12);
        assert!((p.log_moneyness() + 0.06).abs() < 1e-12);
    }
}
