class SiftBin < Formula
  version "0.1.0"
  desc "Point at anything, search everything — fast file indexer and search tool"
  homepage "https://github.com/raymondj99/sift"

  if OS.mac?
    if Hardware::CPU.arm?
      url "https://github.com/raymondj99/sift/releases/download/#{version}/sift-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    else
      url "https://github.com/raymondj99/sift/releases/download/#{version}/sift-#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end
  elsif OS.linux?
    url "https://github.com/raymondj99/sift/releases/download/#{version}/sift-#{version}-x86_64-unknown-linux-musl.tar.gz"
    sha256 "PLACEHOLDER"
  end

  def install
    bin.install "sift"

    # Install shell completions if present
    bash_completion.install "complete/sift.bash" if File.exist? "complete/sift.bash"
    zsh_completion.install "complete/_sift" if File.exist? "complete/_sift"
    fish_completion.install "complete/sift.fish" if File.exist? "complete/sift.fish"
  end

  test do
    assert_match "sift #{version}", shell_output("#{bin}/vx --version")
  end
end
