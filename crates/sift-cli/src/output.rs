#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use std::io::BufRead;
use sift_core::{IndexStats, SearchResult};

use crate::OutputFormat;

// ---------------------------------------------------------------------------
// Search results
// ---------------------------------------------------------------------------

/// Format and print search results in the requested output format.
pub fn format_search_results(results: &[SearchResult], format: &OutputFormat, show_context: bool) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(results).unwrap_or_default()
            );
        }
        OutputFormat::Csv => {
            println!("uri,score,chunk_index,file_type,text");
            for r in results {
                // Escape text for CSV: wrap in quotes, double any internal quotes
                let escaped = r.text.replace('"', "\"\"").replace('\n', " ");
                println!(
                    "{},{:.4},{},{},\"{}\"",
                    r.uri, r.score, r.chunk_index, r.file_type, escaped
                );
            }
        }
        OutputFormat::Human => {
            print_search_results_human(results, show_context);
        }
    }
}

/// Legacy wrapper — delegates to `format_search_results`.
pub fn print_search_results(results: &[SearchResult], format: &OutputFormat, show_context: bool) {
    format_search_results(results, format, show_context);
}

fn print_search_results_human(results: &[SearchResult], show_context: bool) {
    if results.is_empty() {
        println!("{}", "No results found.".dimmed());
        return;
    }

    for (i, result) in results.iter().enumerate() {
        let uri_display = result.uri.strip_prefix("file://").unwrap_or(&result.uri);

        let score_color = if result.score > 0.8 {
            Color::Green
        } else if result.score > 0.5 {
            Color::Yellow
        } else {
            Color::Red
        };

        println!(
            "  {}. {}  {}",
            (i + 1).to_string().bold(),
            uri_display.cyan(),
            format!("{:.2}", result.score).color(score_color).bold(),
        );

        if show_context {
            if let Some(ctx) = read_context_lines(&result.uri, result.byte_range) {
                for line in &ctx {
                    println!("     {}", line);
                }
            } else {
                print_snippet(result);
            }
        } else {
            print_snippet(result);
        }

        if i < results.len() - 1 {
            println!();
        }
    }
}

fn print_snippet(result: &SearchResult) {
    let snippet: String = result
        .text
        .chars()
        .take(120)
        .collect::<String>()
        .replace('\n', " ");
    println!("     {}", snippet.dimmed());
}

/// Read +/-2 lines of context around a byte range from a file URI.
fn read_context_lines(uri: &str, byte_range: Option<(u64, u64)>) -> Option<Vec<String>> {
    let path = uri.strip_prefix("file://")?;
    let (start_byte, _end_byte) = byte_range?;

    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);

    let mut offset: u64 = 0;
    let mut lines: Vec<(usize, String)> = Vec::new();
    let mut target_line = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.ok()?;
        let line_end = offset + line.len() as u64 + 1; // +1 for newline
        if offset <= start_byte && start_byte < line_end && target_line == 0 {
            target_line = line_num;
        }
        lines.push((line_num, line));
        offset = line_end;
    }

    let context_radius = 2;
    let start = target_line.saturating_sub(context_radius);
    let end = (target_line + context_radius + 1).min(lines.len());

    let mut output = Vec::new();
    for &(line_num, ref line) in &lines[start..end] {
        let num = format!("{:>4}", line_num + 1);
        let marker = if line_num == target_line { ">" } else { " " };
        output.push(format!("{}{}{}", num.dimmed(), marker.green().bold(), line));
    }

    Some(output)
}

// ---------------------------------------------------------------------------
// Index stats
// ---------------------------------------------------------------------------

/// Format and print index stats in the requested output format.
pub fn format_stats(stats: &IndexStats, format: &OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(stats).unwrap_or_default()
            );
        }
        OutputFormat::Csv => {
            println!("total_sources,total_chunks,index_size_bytes");
            println!(
                "{},{},{}",
                stats.total_sources, stats.total_chunks, stats.index_size_bytes
            );
        }
        OutputFormat::Human => {
            print_index_stats_human(stats);
        }
    }
}

/// Legacy wrapper — delegates to `format_stats`.
pub fn print_index_stats(stats: &IndexStats, format: &OutputFormat) {
    format_stats(stats, format);
}

fn print_index_stats_human(stats: &IndexStats) {
    println!("{}", "Index Status".bold().underline());
    println!("  Sources:  {}", stats.total_sources.to_string().green());
    println!("  Chunks:   {}", stats.total_chunks.to_string().green());

    if stats.index_size_bytes > 0 {
        println!(
            "  Size:     {}",
            format_bytes(stats.index_size_bytes).green()
        );
    }

    if !stats.file_type_counts.is_empty() {
        println!();
        println!("  {}", "File Types:".bold());
        let mut types: Vec<_> = stats.file_type_counts.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1));
        for (ft, count) in types {
            println!("    {}: {}", ft.cyan(), count);
        }
    }
}

// ---------------------------------------------------------------------------
// Source list
// ---------------------------------------------------------------------------

/// Format and print a source list in the requested output format.
pub fn print_source_list(sources: &[(String, String, u32)], format: &OutputFormat) {
    match format {
        OutputFormat::Json => {
            let items: Vec<serde_json::Value> = sources
                .iter()
                .map(|(uri, ft, chunks)| {
                    serde_json::json!({
                        "uri": uri,
                        "file_type": ft,
                        "chunks": chunks,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&items).unwrap_or_default()
            );
        }
        OutputFormat::Csv => {
            println!("uri,file_type,chunks");
            for (uri, ft, chunks) in sources {
                println!("{},{},{}", uri, ft, chunks);
            }
        }
        OutputFormat::Human => {
            print_source_list_human(sources);
        }
    }
}

fn print_source_list_human(sources: &[(String, String, u32)]) {
    if sources.is_empty() {
        println!("{}", "No sources indexed.".dimmed());
        return;
    }

    println!(
        "{} {} indexed",
        sources.len().to_string().bold(),
        if sources.len() == 1 {
            "source"
        } else {
            "sources"
        }
    );
    println!();

    for (uri, ft, chunks) in sources {
        let display = uri.strip_prefix("file://").unwrap_or(uri);
        println!(
            "  {} {} {}",
            display.cyan(),
            format!("[{}]", ft).dimmed(),
            format!("({} chunks)", chunks).dimmed(),
        );
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_context_lines_returns_surrounding_lines() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        let content =
            "line one\nline two\nline three\nline four\nline five\nline six\nline seven\n";
        std::fs::write(&file_path, content).unwrap();

        let uri = format!("file://{}", file_path.display());
        // byte offset of "line three" = 9 (line one\n) + 9 (line two\n) = 18
        let result = read_context_lines(&uri, Some((18, 28)));
        assert!(result.is_some());
        let lines = result.unwrap();
        // Target is line 3 (0-indexed: 2), context +/-2 -> lines 1-5 (0-indexed: 0-4)
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn test_read_context_lines_at_file_start() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "first\nsecond\nthird\nfourth\n").unwrap();

        let uri = format!("file://{}", file_path.display());
        // byte offset 0 -> first line
        let result = read_context_lines(&uri, Some((0, 5)));
        assert!(result.is_some());
        let lines = result.unwrap();
        // Target is line 1 (0-indexed: 0), context +/-2, but start is clamped to 0
        // So lines 0..3 (0-indexed), i.e. 3 lines
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_read_context_lines_at_file_end() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        let content = "a\nb\nc\nd\ne\n";
        std::fs::write(&file_path, content).unwrap();

        let uri = format!("file://{}", file_path.display());
        // byte offset of "e" = 8
        let result = read_context_lines(&uri, Some((8, 9)));
        assert!(result.is_some());
        let lines = result.unwrap();
        // Target is line 5 (0-indexed: 4), context +/-2, clamped to end
        assert!(lines.len() >= 3); // at least lines c, d, e
    }

    #[test]
    fn test_read_context_lines_returns_none_for_non_file_uri() {
        let result = read_context_lines("http://example.com", Some((0, 10)));
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_lines_returns_none_for_no_byte_range() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "content").unwrap();

        let uri = format!("file://{}", file_path.display());
        let result = read_context_lines(&uri, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_lines_returns_none_for_missing_file() {
        let result = read_context_lines("file:///nonexistent/path.txt", Some((0, 5)));
        assert!(result.is_none());
    }

    #[test]
    fn test_read_context_lines_single_line_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("single.txt");
        std::fs::write(&file_path, "only one line").unwrap();

        let uri = format!("file://{}", file_path.display());
        let result = read_context_lines(&uri, Some((0, 5)));
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_format_search_results_json_mode() {
        let results = vec![SearchResult {
            uri: "file:///test.txt".to_string(),
            text: "hello world".to_string(),
            score: 0.9,
            chunk_index: 0,
            content_type: sift_core::ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            byte_range: None,
        }];
        // Should not panic in any mode
        format_search_results(&results, &OutputFormat::Json, false);
        format_search_results(&results, &OutputFormat::Json, true);
        format_search_results(&results, &OutputFormat::Csv, false);
    }

    #[test]
    fn test_format_search_results_empty() {
        format_search_results(&[], &OutputFormat::Human, false);
        format_search_results(&[], &OutputFormat::Human, true);
        format_search_results(&[], &OutputFormat::Json, false);
        format_search_results(&[], &OutputFormat::Csv, false);
    }

    #[test]
    fn test_print_snippet_truncates_long_text() {
        let result = SearchResult {
            uri: "file:///test.txt".to_string(),
            text: "a".repeat(200),
            score: 0.5,
            chunk_index: 0,
            content_type: sift_core::ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            byte_range: None,
        };
        // Should not panic
        print_snippet(&result);
    }

    #[test]
    fn test_format_stats_json() {
        let stats = IndexStats {
            total_sources: 5,
            total_chunks: 20,
            index_size_bytes: 1024,
            file_type_counts: Default::default(),
        };
        // Should not panic
        format_stats(&stats, &OutputFormat::Json);
    }

    #[test]
    fn test_format_stats_csv() {
        let stats = IndexStats {
            total_sources: 5,
            total_chunks: 20,
            index_size_bytes: 1024,
            file_type_counts: Default::default(),
        };
        // Should not panic
        format_stats(&stats, &OutputFormat::Csv);
    }

    #[test]
    fn test_print_source_list_csv() {
        let sources = vec![
            ("file:///a.rs".to_string(), "rs".to_string(), 3),
            ("file:///b.py".to_string(), "py".to_string(), 5),
        ];
        // Should not panic
        print_source_list(&sources, &OutputFormat::Csv);
    }
}
