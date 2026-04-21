#!/bin/sh
# sift installer — fetches the latest release binary and puts it on your PATH.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/raymondj99/sift/main/install.sh | sh
#
# Environment overrides (all optional):
#   SIFT_VERSION=v0.1.6      # install a specific version instead of latest
#   SIFT_INSTALL_DIR=...     # install location (default: /usr/local/bin if
#                              writable, else ~/.local/bin)
#   SIFT_NO_COMPLETIONS=1    # skip shell completion install
#   SIFT_NO_MODIFY_PATH=1    # silence the PATH-guidance note when installing
#                              to ~/.local/bin
#
# The script verifies SHA-256 checksums before installing and prints the
# download URL before fetching. Reads no files outside of the temp dir it
# creates and cleans up on exit.

set -eu

GITHUB_REPO="raymondj99/sift"
BIN_NAME="sift"

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

note() {
    printf '%s\n' "$1"
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

# -------- Environment checks --------

# curl or wget is required for downloads.
if command -v curl >/dev/null 2>&1; then
    FETCH_CMD="curl -fsSL"
    FETCH_OUT="curl -fsSL -o"
elif command -v wget >/dev/null 2>&1; then
    FETCH_CMD="wget -qO-"
    FETCH_OUT="wget -qO"
else
    die "need curl or wget to download release archive"
fi

need_cmd tar
# sha256sum (Linux) or shasum (macOS) is required for integrity verification.
if command -v sha256sum >/dev/null 2>&1; then
    SHA256_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    SHA256_CMD="shasum -a 256"
else
    die "need sha256sum or shasum to verify downloads"
fi

# -------- Platform detection --------

os=$(uname -s)
arch=$(uname -m)
case "$os/$arch" in
    Darwin/arm64)          target="aarch64-apple-darwin" ;;
    Darwin/x86_64)         target="x86_64-apple-darwin" ;;
    Linux/x86_64)          target="x86_64-unknown-linux-gnu" ;;
    *)
        die "unsupported platform: $os/$arch. Build from source:
  cargo install --git https://github.com/${GITHUB_REPO} sift-cli"
        ;;
esac

# -------- Version resolution --------

version="${SIFT_VERSION:-}"
if [ -z "$version" ]; then
    # Extract tag_name from the "latest release" API — no jq required.
    version=$($FETCH_CMD "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" \
        | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)
    [ -n "$version" ] || die "could not resolve latest release tag"
fi

# -------- Install location --------

install_dir="${SIFT_INSTALL_DIR:-}"
if [ -z "$install_dir" ]; then
    # Prefer /usr/local/bin when the user already owns it (Homebrew layout on
    # macOS, admin Linux). Fall back to ~/.local/bin otherwise — no sudo.
    if [ -w /usr/local/bin ] 2>/dev/null; then
        install_dir="/usr/local/bin"
    else
        install_dir="$HOME/.local/bin"
        mkdir -p "$install_dir"
    fi
fi
[ -d "$install_dir" ] || die "install directory does not exist: $install_dir"
[ -w "$install_dir" ] || die "install directory not writable: $install_dir"

# -------- Download + verify --------

tarball="sift-${version}-${target}.tar.gz"
url="https://github.com/${GITHUB_REPO}/releases/download/${version}/${tarball}"
sums_url="https://github.com/${GITHUB_REPO}/releases/download/${version}/SHA256SUMS.txt"

tmp=$(mktemp -d 2>/dev/null || mktemp -d -t sift-install)
trap 'rm -rf "$tmp"' EXIT INT TERM

note "Downloading $url"
$FETCH_OUT "$tmp/$tarball" "$url" || die "download failed: $url"

$FETCH_OUT "$tmp/SHA256SUMS.txt" "$sums_url" || die "download failed: $sums_url"
# Match the tarball line literally (-F) so regex metacharacters in the
# filename (e.g. the dots in `.tar.gz`) don't match spuriously, then pick
# the row whose second field is exactly our tarball.
expected=$(grep -F -- " $tarball" "$tmp/SHA256SUMS.txt" \
    | awk -v t="$tarball" '$2 == t { print $1; exit }')
[ -n "$expected" ] || die "checksum not found for $tarball in SHA256SUMS.txt"
actual=$(cd "$tmp" && $SHA256_CMD "$tarball" | awk '{print $1}')
if [ "$expected" != "$actual" ]; then
    die "checksum mismatch for $tarball
  expected: $expected
  actual:   $actual"
fi

# -------- Extract + install --------

tar -xzf "$tmp/$tarball" -C "$tmp"
extracted_dir="$tmp/sift-${version}-${target}"
[ -x "$extracted_dir/$BIN_NAME" ] || die "binary not found in tarball"

install_path="$install_dir/$BIN_NAME"
cp "$extracted_dir/$BIN_NAME" "$install_path"
chmod 755 "$install_path"

# -------- Completions (best-effort) --------

install_completions() {
    [ -n "${SIFT_NO_COMPLETIONS:-}" ] && return 0

    # bash
    for dir in "/usr/local/share/bash-completion/completions" \
               "${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions"; do
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            cp "$extracted_dir/complete/sift.bash" "$dir/sift" 2>/dev/null && break
        fi
    done

    # zsh
    for dir in "/usr/local/share/zsh/site-functions" \
               "$HOME/.zsh/completions"; do
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            cp "$extracted_dir/complete/_sift" "$dir/_sift" 2>/dev/null && break
        fi
    done

    # fish — user-writable by convention when fish is installed
    if [ -d "$HOME/.config/fish" ]; then
        mkdir -p "$HOME/.config/fish/completions"
        cp "$extracted_dir/complete/sift.fish" "$HOME/.config/fish/completions/sift.fish" 2>/dev/null || true
    fi
}
install_completions

# -------- Post-install message --------

note ""
note "Installed $BIN_NAME $version to $install_path"
note "Next: sift scan .          # index the current directory"
note "      sift models download # enable semantic search (optional)"

if [ "$install_dir" = "$HOME/.local/bin" ] && [ -z "${SIFT_NO_MODIFY_PATH:-}" ]; then
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *)
            note ""
            note "Note: $HOME/.local/bin is not in your PATH."
            note '      Add: export PATH="$HOME/.local/bin:$PATH"'
            ;;
    esac
fi
