//! Minimal fixed-width table rendering for terminal reports.

/// Column alignment.
#[derive(Clone, Copy)]
pub enum Align {
    Left,
    Right,
}

/// Render rows under a header, padding each column to its widest cell.
/// Columns are separated by two spaces and the output is indented by two.
pub fn render(header: &[&str], align: &[Align], rows: &[Vec<String>]) -> String {
    let cols = header.len();
    let mut widths: Vec<usize> = header.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let line = |cells: &[String]| {
        let mut out = String::from("  ");
        for (i, cell) in cells.iter().enumerate().take(cols) {
            let pad = widths[i] - cell.chars().count();
            let last = i + 1 == cols;
            match align[i] {
                Align::Left if last => out.push_str(cell),
                Align::Left => {
                    out.push_str(cell);
                    out.push_str(&" ".repeat(pad));
                }
                Align::Right => {
                    out.push_str(&" ".repeat(pad));
                    out.push_str(cell);
                }
            }
            if !last {
                out.push_str("  ");
            }
        }
        out.trim_end().to_string()
    };
    let head: Vec<String> = header.iter().map(|h| h.to_string()).collect();
    let mut out = line(&head);
    out.push('\n');
    out.push_str("  ");
    out.push_str(&"-".repeat(widths.iter().sum::<usize>() + 2 * (cols - 1)));
    for row in rows {
        out.push('\n');
        out.push_str(&line(row));
    }
    out
}
