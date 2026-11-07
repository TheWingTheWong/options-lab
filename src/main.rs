//! `options-lab` command-line interface.

mod cli;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

/// Options pricing workbench: Black-Scholes-Merton, Greeks, implied vol,
/// binomial trees, Monte Carlo and volatility smiles.
#[derive(Parser, Debug)]
#[command(name = "options-lab", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Price an option: BSM Greeks plus tree and Monte Carlo cross-checks.
    Price(cli::price::PriceArgs),
    /// Implied volatility from a quoted price.
    Iv(cli::iv::IvArgs),
    /// Check a call/put pair against put-call parity.
    Parity(cli::parity::ParityArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Price(args) => cli::price::run(args),
        Command::Iv(args) => cli::iv::run(args),
        Command::Parity(args) => cli::parity::run(args),
    };
    match result {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
