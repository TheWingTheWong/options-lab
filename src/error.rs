//! Error type shared by every module in the crate.

use std::fmt;

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong while pricing, solving or parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A model input is outside its valid domain (e.g. negative spot).
    InvalidInput {
        /// Name of the offending parameter.
        field: &'static str,
        /// The value that was supplied.
        value: f64,
        /// Human-readable constraint that was violated.
        reason: &'static str,
    },
    /// A quoted option price violates static no-arbitrage bounds, so no
    /// volatility can reproduce it.
    ArbitrageBounds {
        /// The quoted price.
        price: f64,
        /// Lower no-arbitrage bound (discounted intrinsic value of the forward).
        lower: f64,
        /// Upper no-arbitrage bound.
        upper: f64,
    },
    /// An iterative method did not converge within its iteration budget.
    NoConvergence {
        /// Which algorithm gave up.
        method: &'static str,
        /// Iterations performed.
        iterations: usize,
    },
    /// A root finder was given an interval whose endpoints do not bracket a root.
    InvalidBracket {
        /// Function value at the left endpoint.
        fa: f64,
        /// Function value at the right endpoint.
        fb: f64,
    },
    /// Malformed input data (for example an option chain CSV).
    Parse {
        /// 1-based line number, or 0 when the error is not tied to a line.
        line: usize,
        /// What was wrong.
        message: String,
    },
}

impl Error {
    pub(crate) fn invalid(field: &'static str, value: f64, reason: &'static str) -> Self {
        Error::InvalidInput {
            field,
            value,
            reason,
        }
    }

    pub(crate) fn parse(line: usize, message: impl Into<String>) -> Self {
        Error::Parse {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput {
                field,
                value,
                reason,
            } => write!(f, "invalid {field} = {value}: {reason}"),
            Error::ArbitrageBounds {
                price,
                lower,
                upper,
            } => write!(
                f,
                "price {price:.6} is outside the no-arbitrage range [{lower:.6}, {upper:.6}]"
            ),
            Error::NoConvergence { method, iterations } => {
                write!(f, "{method} did not converge after {iterations} iterations")
            }
            Error::InvalidBracket { fa, fb } => write!(
                f,
                "interval does not bracket a root (f(a) = {fa:e}, f(b) = {fb:e})"
            ),
            Error::Parse { line: 0, message } => write!(f, "parse error: {message}"),
            Error::Parse { line, message } => write!(f, "parse error on line {line}: {message}"),
        }
    }
}

impl std::error::Error for Error {}
