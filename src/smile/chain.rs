//! Option chain CSV parsing.
//!
//! The format is deliberately plain so a broker export can be massaged into
//! it with a spreadsheet:
//!
//! ```text
//! # spot=100, rate=0.04, dividend=0.01, days=45
//! strike,type,bid,ask
//! 95,put,1.95,2.05
//! 105,call,1.40,1.48
//! ```
//!
//! * Header names are case-insensitive; unknown columns are ignored.
//! * `strike` and `type` are required, plus either `bid` and `ask` or a
//!   single `mid` (aliases `price`, `last`).
//! * `# key=value` comment lines supply market data (`spot`, `rate`,
//!   `dividend`/`div`/`q`, `days` or `years`). Any of them can be
//!   overridden on the command line.

use crate::error::{Error, Result};
use crate::types::{DAYS_PER_YEAR, OptionType};

/// One quoted option.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quote {
    /// Strike price.
    pub strike: f64,
    /// Call or put.
    pub kind: OptionType,
    /// Best bid, if quoted.
    pub bid: Option<f64>,
    /// Best ask, if quoted.
    pub ask: Option<f64>,
    /// Mid price used for implied volatility.
    pub mid: f64,
    /// 1-based line number in the source, for error messages.
    pub line: usize,
}

impl Quote {
    /// `ask - bid` when both sides are quoted.
    pub fn spread(&self) -> Option<f64> {
        Some(self.ask? - self.bid?)
    }
}

/// Market data found in `# key=value` lines. Every field is optional.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChainMeta {
    /// Spot price.
    pub spot: Option<f64>,
    /// Risk-free rate.
    pub rate: Option<f64>,
    /// Continuous dividend yield.
    pub dividend: Option<f64>,
    /// Time to expiry in years.
    pub expiry: Option<f64>,
}

/// A parsed chain for a single expiry.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Chain {
    /// Market data from comment directives.
    pub meta: ChainMeta,
    /// Quotes in file order.
    pub quotes: Vec<Quote>,
}

#[derive(Default)]
struct Columns {
    strike: Option<usize>,
    kind: Option<usize>,
    bid: Option<usize>,
    ask: Option<usize>,
    mid: Option<usize>,
}

/// Parse a chain from CSV text.
///
/// ```
/// use options_lab::smile::parse_chain;
/// let chain = parse_chain("# spot=100, days=30\nstrike,type,mid\n100,call,2.5\n").unwrap();
/// assert_eq!(chain.meta.spot, Some(100.0));
/// assert_eq!(chain.quotes[0].mid, 2.5);
/// ```
pub fn parse_chain(text: &str) -> Result<Chain> {
    let mut chain = Chain::default();
    let mut cols: Option<Columns> = None;

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            parse_directives(comment, line_no, &mut chain.meta)?;
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let Some(c) = &cols else {
            cols = Some(parse_header(&fields, line_no)?);
            continue;
        };

        let get = |i: Option<usize>| {
            i.and_then(|i| fields.get(i).copied())
                .filter(|s| !s.is_empty())
        };
        let number = |name: &str, i: Option<usize>| -> Result<Option<f64>> {
            get(i)
                .map(|s| {
                    s.parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| {
                            Error::parse(line_no, format!("{name} '{s}' is not a number"))
                        })
                })
                .transpose()
        };

        let strike =
            number("strike", c.strike)?.ok_or_else(|| Error::parse(line_no, "missing strike"))?;
        if strike <= 0.0 {
            return Err(Error::parse(line_no, "strike must be positive"));
        }
        let kind: OptionType = get(c.kind)
            .ok_or_else(|| Error::parse(line_no, "missing option type"))?
            .parse()
            .map_err(|e| match e {
                Error::Parse { message, .. } => Error::parse(line_no, message),
                other => other,
            })?;
        let bid = number("bid", c.bid)?;
        let ask = number("ask", c.ask)?;
        let mid = match (number("mid", c.mid)?, bid, ask) {
            (Some(m), _, _) => m,
            (None, Some(b), Some(a)) => {
                if b < 0.0 || a < b {
                    return Err(Error::parse(
                        line_no,
                        format!("crossed or negative market {b}/{a}"),
                    ));
                }
                0.5 * (b + a)
            }
            _ => return Err(Error::parse(line_no, "need either mid or both bid and ask")),
        };
        chain.quotes.push(Quote {
            strike,
            kind,
            bid,
            ask,
            mid,
            line: line_no,
        });
    }

    if cols.is_none() {
        return Err(Error::parse(0, "no header row found"));
    }
    Ok(chain)
}

fn parse_header(fields: &[&str], line_no: usize) -> Result<Columns> {
    let mut c = Columns::default();
    for (i, name) in fields.iter().enumerate() {
        match name.to_ascii_lowercase().as_str() {
            "strike" | "k" => c.strike = Some(i),
            "type" | "kind" | "cp" | "right" | "side" => c.kind = Some(i),
            "bid" => c.bid = Some(i),
            "ask" | "offer" => c.ask = Some(i),
            "mid" | "price" | "last" => c.mid = Some(i),
            _ => {}
        }
    }
    if c.strike.is_none() || c.kind.is_none() {
        return Err(Error::parse(
            line_no,
            "header must contain 'strike' and 'type' columns",
        ));
    }
    if c.mid.is_none() && (c.bid.is_none() || c.ask.is_none()) {
        return Err(Error::parse(
            line_no,
            "header needs 'bid' and 'ask' columns, or a 'mid' column",
        ));
    }
    Ok(c)
}

fn parse_directives(comment: &str, line_no: usize, meta: &mut ChainMeta) -> Result<()> {
    for part in comment.split([',', ';']) {
        let Some((key, value)) = part.split_once(['=', ':']) else {
            continue; // free-form comment text
        };
        let key = key.trim().to_ascii_lowercase();
        let parse = || {
            value.trim().parse::<f64>().map_err(|_| {
                Error::parse(line_no, format!("{key} '{}' is not a number", value.trim()))
            })
        };
        match key.as_str() {
            "spot" | "s" | "underlying" => meta.spot = Some(parse()?),
            "rate" | "r" => meta.rate = Some(parse()?),
            "dividend" | "div" | "q" => meta.dividend = Some(parse()?),
            "days" | "dte" => meta.expiry = Some(parse()? / DAYS_PER_YEAR),
            "years" | "t" => meta.expiry = Some(parse()?),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_directives_and_bid_ask() {
        let text = "\
# synthetic chain
# spot=100, rate=0.04; dividend=0.01, days=73
Strike, Type, Bid, Ask, Volume
95, P, 1.90, 2.10, 120
105, call, 1.40, 1.50,
";
        let chain = parse_chain(text).unwrap();
        assert_eq!(chain.meta.spot, Some(100.0));
        assert_eq!(chain.meta.rate, Some(0.04));
        assert_eq!(chain.meta.dividend, Some(0.01));
        assert!((chain.meta.expiry.unwrap() - 0.2).abs() < 1e-15);
        assert_eq!(chain.quotes.len(), 2);
        assert_eq!(chain.quotes[0].kind, OptionType::Put);
        assert!((chain.quotes[0].mid - 2.0).abs() < 1e-12);
        assert!((chain.quotes[1].spread().unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(chain.quotes[1].line, 5);
    }

    #[test]
    fn mid_column_takes_precedence() {
        let chain = parse_chain("strike,type,bid,ask,mid\n100,c,1,2,1.6\n").unwrap();
        assert_eq!(chain.quotes[0].mid, 1.6);
    }

    #[test]
    fn reports_line_numbers() {
        let err = parse_chain("strike,type,mid\n100,call,1.0\n105,straddle,2.0\n").unwrap_err();
        assert!(matches!(err, Error::Parse { line: 3, .. }), "{err}");
        let err = parse_chain("strike,type,bid,ask\n100,call,2.0,1.0\n").unwrap_err();
        assert!(err.to_string().contains("line 2"), "{err}");
    }

    #[test]
    fn rejects_bad_headers() {
        assert!(parse_chain("strike,bid,ask\n").is_err());
        assert!(parse_chain("strike,type,bid\n").is_err());
        assert!(parse_chain("# spot=100\n").is_err());
    }
}
