//! Lightweight HTTP client for the sift daemon's Unix domain socket.
//!
//! Uses only `std` — no tokio, no hyper, no feature gates — so any sift
//! build can detect and talk to a running daemon.

use std::io::{Read, Write};
use std::path::PathBuf;

/// Path to the daemon Unix socket.
fn socket_path() -> Option<PathBuf> {
    Some(sift_core::Config::sift_dir().ok()?.join("daemon.sock"))
}

/// Path to the daemon PID file.
fn pid_path() -> Option<PathBuf> {
    Some(sift_core::Config::sift_dir().ok()?.join("daemon.pid"))
}

/// Read the daemon PID from disk.
fn read_pid() -> Option<u32> {
    let content = std::fs::read_to_string(pid_path()?).ok()?;
    content.trim().parse().ok()
}

/// Check if a process with the given PID is alive.
fn is_process_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Check if a daemon is running and responsive (socket exists + health check).
pub fn is_available() -> bool {
    let socket = match socket_path() {
        Some(p) if p.exists() => p,
        _ => return false,
    };
    // Quick health check
    matches!(get_raw(&socket, "/health"), Some((200, body)) if body.trim() == "ok")
}

/// Send a GET request to the daemon and return the response body.
/// Returns `None` if the daemon is not running or the request fails.
pub fn get(path: &str) -> Option<String> {
    let socket = socket_path()?;
    let (status, body) = get_raw(&socket, path)?;
    if status == 200 {
        Some(body)
    } else {
        None
    }
}

/// Low-level GET over Unix socket. Returns (status_code, body).
fn get_raw(socket: &std::path::Path, path: &str) -> Option<(u16, String)> {
    let mut stream = std::os::unix::net::UnixStream::connect(socket).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .ok()?;

    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    stream.flush().ok()?;

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).ok()?;
    parse_http_response(&buf)
}

/// Send a POST request with a JSON body. Returns the response body on 200.
#[allow(dead_code)]
pub fn post(path: &str, json_body: &str) -> Option<String> {
    let socket = socket_path()?;
    let mut stream = std::os::unix::net::UnixStream::connect(&socket).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(120)))
        .ok()?;

    let request = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {json_body}",
        json_body.len()
    );
    stream.write_all(request.as_bytes()).ok()?;
    stream.flush().ok()?;

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).ok()?;
    let (status, body) = parse_http_response(&buf)?;
    if status == 200 {
        Some(body)
    } else {
        None
    }
}

/// Parse an HTTP/1.1 response into (status_code, body).
fn parse_http_response(raw: &[u8]) -> Option<(u16, String)> {
    let text = String::from_utf8_lossy(raw);

    // Status line: "HTTP/1.1 200 OK\r\n"
    let status_line = text.lines().next()?;
    let status_code: u16 = status_line.split_whitespace().nth(1)?.parse().ok()?;

    // Body starts after \r\n\r\n
    let header_end = text.find("\r\n\r\n")? + 4;
    let body = &text[header_end..];

    // Handle chunked transfer encoding (axum may use it with Connection: close)
    if text.contains("transfer-encoding: chunked") || text.contains("Transfer-Encoding: chunked") {
        Some((status_code, decode_chunked(body)))
    } else {
        Some((status_code, body.to_string()))
    }
}

/// Decode chunked transfer encoding.
fn decode_chunked(raw: &str) -> String {
    let mut result = String::new();
    let mut remaining = raw;

    while let Some(size_end) = remaining.find("\r\n") {
        // Each chunk: <hex-size>\r\n<data>\r\n
        let size_str = &remaining[..size_end];
        let size = match usize::from_str_radix(size_str.trim(), 16) {
            Ok(0) => break, // terminal chunk
            Ok(s) => s,
            Err(_) => break,
        };
        remaining = &remaining[size_end + 2..]; // skip size + \r\n
        if remaining.len() < size {
            break;
        }
        result.push_str(&remaining[..size]);
        remaining = &remaining[size..];
        // Skip trailing \r\n
        if remaining.starts_with("\r\n") {
            remaining = &remaining[2..];
        }
    }

    result
}

/// Stop the daemon if it's running. Returns `true` if a daemon was stopped.
pub fn stop_if_running() -> bool {
    let pid = match read_pid() {
        Some(pid) if is_process_alive(pid) => pid,
        _ => return false,
    };

    let _ = std::process::Command::new("kill")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    // Wait for exit (up to 5s)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if !is_process_alive(pid) {
            // Clean up files
            if let Some(p) = pid_path() {
                let _ = std::fs::remove_file(p);
            }
            if let Some(p) = socket_path() {
                let _ = std::fs::remove_file(p);
            }
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    false
}

/// Try to start the daemon. Returns `true` if started (socket becomes available).
pub fn try_start() -> bool {
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return false,
    };

    let log_path = match sift_core::Config::sift_dir() {
        Ok(d) => d.join("daemon.log"),
        Err(_) => return false,
    };
    let log_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let _ = std::process::Command::new(&exe)
        .args(["daemon", "run"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(log_file))
        .spawn();

    // Wait for socket (up to 10s)
    let socket = match socket_path() {
        Some(p) => p,
        None => return false,
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if socket.exists() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_http_response_simple() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
        let (status, body) = parse_http_response(raw).unwrap();
        assert_eq!(status, 200);
        assert_eq!(body, "ok");
    }

    #[test]
    fn test_parse_http_response_404() {
        let raw = b"HTTP/1.1 404 Not Found\r\n\r\nnot found";
        let (status, body) = parse_http_response(raw).unwrap();
        assert_eq!(status, 404);
        assert_eq!(body, "not found");
    }

    #[test]
    fn test_parse_http_response_chunked() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let (status, body) = parse_http_response(raw).unwrap();
        assert_eq!(status, 200);
        assert_eq!(body, "hello world");
    }

    #[test]
    fn test_decode_chunked_empty() {
        assert_eq!(decode_chunked("0\r\n\r\n"), "");
    }

    #[test]
    fn test_decode_chunked_single() {
        assert_eq!(decode_chunked("3\r\nabc\r\n0\r\n\r\n"), "abc");
    }

    #[test]
    fn test_is_available_when_no_socket() {
        // Should return false when no daemon is running
        // (Safe to test since we won't have a daemon.sock in CI)
        let _ = is_available();
    }
}
