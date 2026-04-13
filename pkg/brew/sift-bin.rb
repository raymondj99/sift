class SiftBin < Formula
  version "0.1.4"
  desc "Local semantic search engine — index 30+ formats, hybrid search, AI agent memory"
  homepage "https://github.com/raymondj99/sift"
  license "MIT"

  if OS.mac?
    if Hardware::CPU.arm?
      url "https://github.com/raymondj99/sift/releases/download/v#{version}/sift-v#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    else
      url "https://github.com/raymondj99/sift/releases/download/v#{version}/sift-v#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end
  elsif OS.linux?
    url "https://github.com/raymondj99/sift/releases/download/v#{version}/sift-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
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
    assert_match "sift #{version}", shell_output("#{bin}/sift --version")
  end
end
