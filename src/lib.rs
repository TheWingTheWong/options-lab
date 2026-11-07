//! # options-lab
//!
//! An options pricing toolkit: closed-form Black-Scholes-Merton pricing and
//! Greeks, implied volatility, Cox-Ross-Rubinstein trees for American
//! exercise, variance-reduced Monte Carlo, and volatility smile fitting.
#![warn(missing_docs)]

pub mod bsm;
pub mod error;
pub mod iv;
pub mod math;
pub mod parity;
pub mod solver;
pub mod types;

pub use error::{Error, Result};
pub use types::{DAYS_PER_YEAR, ExerciseStyle, OptionParams, OptionType};
