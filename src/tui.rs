//! Terminal renderer: the Doom frame as 24-bit half-block cells on the left, a
//! status panel on the right. Raw mode for single-key control.

use anyhow::Result;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{Stdout, Write, stdout};
use std::time::Duration;

pub struct Tui {
    out: Stdout,
    pub cols: usize,
}

impl Tui {
    pub fn new(cols: usize) -> Result<Self> {
        let mut out = stdout();
        terminal::enable_raw_mode()?;
        execute!(out, EnterAlternateScreen, cursor::Hide, terminal::Clear(terminal::ClearType::All))?;
        Ok(Self { out, cols })
    }

    /// Non-blocking: the key pressed since the last poll, if any.
    pub fn key(&mut self) -> Result<Option<KeyCode>> {
        if event::poll(Duration::ZERO)? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    return Ok(Some(k.code));
                }
            }
        }
        Ok(None)
    }

    /// `rgb` is w*h*3 bytes. Two pixel rows per text row, `self.cols` cells wide.
    pub fn draw(&mut self, rgb: &[u8], w: usize, h: usize, panel: &[String]) -> Result<()> {
        // Keep everything inside the terminal: the frame never takes more than the width minus a
        // minimal panel, and every panel line is wrapped to what is left (escape codes take no width).
        let term = term_cols();
        let cols = self.cols.min(w).min(term.saturating_sub(26).max(16));
        let width = term.saturating_sub(cols + 2).max(24);
        let panel: Vec<String> = panel.iter().flat_map(|l| wrap_ansi(l, width)).collect();
        let panel = &panel[..];
        let sx = w as f32 / cols as f32; // pixels per cell horizontally
        let rows = ((h as f32 / (2.0 * sx)) as usize).max(1); // keep aspect: a cell is ~2x taller than wide
        let sy = h as f32 / (rows as f32 * 2.0);
        let mut buf = String::with_capacity(rows * cols * 40);
        buf.push_str("\x1b[H");
        let px = |x: usize, y: usize| -> (u8, u8, u8) {
            let i = (y.min(h - 1) * w + x.min(w - 1)) * 3;
            (rgb[i], rgb[i + 1], rgb[i + 2])
        };
        for r in 0..rows {
            let y0 = (r as f32 * 2.0 * sy) as usize;
            let y1 = ((r as f32 * 2.0 + 1.0) * sy) as usize;
            for c in 0..cols {
                let x = (c as f32 * sx) as usize;
                let (tr, tg, tb) = px(x, y0);
                let (br, bg, bb) = px(x, y1);
                buf.push_str(&format!("\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m\u{2580}"));
            }
            buf.push_str("\x1b[0m  ");
            if let Some(line) = panel.get(r) {
                buf.push_str(line);
            }
            buf.push_str("\x1b[K\r\n");
        }
        for line in panel.iter().skip(rows) {
            buf.push_str(line);
            buf.push_str("\x1b[K\r\n");
        }
        buf.push_str("\x1b[J");
        self.out.write_all(buf.as_bytes())?;
        self.out.flush()?;
        Ok(())
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = execute!(self.out, cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

/// Terminal width in columns; 160 when unknown (no tty, or a zero-sized pty).
fn term_cols() -> usize {
    terminal::size().ok().map(|(c, _)| c as usize).filter(|&c| c > 0).unwrap_or(160)
}

/// Visible width of a string: SGR escape sequences (`ESC [ ... m`) count for nothing.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            if c == 'm' {
                in_esc = false;
            }
        } else if c == '\x1b' {
            in_esc = true;
        } else {
            n += 1;
        }
    }
    n
}

/// Wrap one panel line to `width` visible columns. Escape sequences pass through with zero width
/// (SGR state carries across the break, so a bold span stays bold). Breaks after the last comma
/// or space in range, else hard. Continuation lines are indented two spaces.
pub fn wrap_ansi(line: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut vis = 0usize;
    let mut soft: Option<usize> = None; // byte offset in `cur` just after the last ',' or ' '
    let mut it = line.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            cur.push(c);
            while let Some(&n) = it.peek() {
                cur.push(n);
                it.next();
                if n == 'm' {
                    break;
                }
            }
            continue;
        }
        let limit = if out.is_empty() { width } else { width - 2 };
        if vis >= limit {
            let (head, tail) = match soft {
                Some(b) if b > 0 && b < cur.len() => (cur[..b].to_string(), cur[b..].to_string()),
                _ => (std::mem::take(&mut cur), String::new()),
            };
            out.push(head.trim_end().to_string());
            let tail = tail.trim_start().to_string();
            vis = 2 + visible_len(&tail);
            cur = format!("  {tail}");
            soft = None;
        }
        cur.push(c);
        vis += 1;
        if c == ',' || c == ' ' {
            soft = Some(cur.len());
        }
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

pub fn bar(p: f32, width: usize) -> String {
    let n = ((p.clamp(0.0, 1.0)) * width as f32).round() as usize;
    format!("{}{}", "\u{2588}".repeat(n), "\u{2591}".repeat(width.saturating_sub(n)))
}
