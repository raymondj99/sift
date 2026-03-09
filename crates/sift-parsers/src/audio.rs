#[cfg(feature = "audio")]
use crate::traits::Parser;
#[cfg(feature = "audio")]
use sift_core::{ContentType, ParsedDocument, SiftResult};
#[cfg(feature = "audio")]
use std::collections::HashMap;
#[cfg(feature = "audio")]
use std::io::Cursor;
#[cfg(feature = "audio")]
use symphonia::core::formats::FormatOptions;
#[cfg(feature = "audio")]
use symphonia::core::io::MediaSourceStream;
#[cfg(feature = "audio")]
use symphonia::core::meta::MetadataOptions;
#[cfg(feature = "audio")]
use symphonia::core::probe::Hint;
#[cfg(feature = "audio")]
use tracing::debug;

/// Parser for audio files. Extracts metadata (duration, codec, sample rate,
/// channels, and ID3/Vorbis tags) into searchable text using `symphonia`.
///
/// This parser does not decode audio samples. It probes the container format
/// and reads whatever metadata the codec headers and tag blocks provide,
/// producing a text representation suitable for full-text and vector search.
#[cfg(feature = "audio")]
pub struct AudioParser;

#[cfg(feature = "audio")]
impl AudioParser {
    /// MIME types recognized as audio.
    const AUDIO_MIMES: &[&str] = &[
        "audio/mpeg",
        "audio/wav",
        "audio/flac",
        "audio/ogg",
        "audio/aac",
        "audio/mp4",
        "audio/x-wav",
    ];

    /// File extensions recognized as audio.
    const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "ogg", "aac", "m4a", "wma", "opus"];

    /// Map an extension to a friendly format label used in the text summary.
    fn format_label(extension: Option<&str>) -> &str {
        match extension.map(|e| e.to_lowercase()).as_deref() {
            Some("mp3") => "MP3",
            Some("wav") => "WAV",
            Some("flac") => "FLAC",
            Some("ogg") => "OGG",
            Some("aac") => "AAC",
            Some("m4a") => "M4A",
            Some("wma") => "WMA",
            Some("opus") => "Opus",
            _ => "Audio",
        }
    }

    /// Attempt to build a symphonia `Hint` from the MIME type and extension
    /// so the prober can narrow down which demuxer to try first.
    fn build_hint(mime_type: Option<&str>, extension: Option<&str>) -> Hint {
        let mut hint = Hint::new();
        if let Some(ext) = extension {
            hint.with_extension(&ext.to_lowercase());
        }
        if let Some(mime) = mime_type {
            hint.mime_type(mime);
        }
        hint
    }
}

#[cfg(feature = "audio")]
impl Parser for AudioParser {
    fn can_parse(&self, mime_type: Option<&str>, extension: Option<&str>) -> bool {
        if let Some(mime) = mime_type {
            let mime_lower = mime.to_lowercase();
            if Self::AUDIO_MIMES.iter().any(|m| mime_lower.starts_with(m)) {
                return true;
            }
        }
        if let Some(ext) = extension {
            let ext_lower = ext.to_lowercase();
            if Self::AUDIO_EXTENSIONS.contains(&ext_lower.as_str()) {
                return true;
            }
        }
        false
    }

    fn parse(
        &self,
        content: &[u8],
        mime_type: Option<&str>,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        let label = Self::format_label(extension);

        // Wrap the content bytes in a MediaSourceStream for symphonia.
        let cursor = Cursor::new(content.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
        let hint = Self::build_hint(mime_type, extension);

        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();

        let probe_result =
            symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts);

        match probe_result {
            Ok(mut probed) => self.build_from_probe(&mut probed, label, extension),
            Err(e) => {
                debug!(
                    error = %e,
                    ext = ?extension,
                    "Symphonia probe failed, falling back to basic metadata"
                );
                self.build_fallback(label, extension, content.len())
            }
        }
    }

    fn name(&self) -> &str {
        "audio"
    }
}

#[cfg(feature = "audio")]
impl AudioParser {
    /// Build a `ParsedDocument` from a successful symphonia probe.
    fn build_from_probe(
        &self,
        probed: &mut symphonia::core::probe::ProbeResult,
        label: &str,
        extension: Option<&str>,
    ) -> SiftResult<ParsedDocument> {
        let mut metadata = HashMap::new();
        let mut text_parts: Vec<String> = Vec::new();

        text_parts.push(format!("Audio file ({})", label));

        metadata.insert(
            "extension".to_string(),
            extension.unwrap_or("audio").to_string(),
        );

        // --- Track-level info (codec, sample rate, channels, duration) ---
        if let Some(track) = probed.format.default_track() {
            let codec_params = &track.codec_params;

            // Codec
            let codec_name = symphonia::core::codecs::CODEC_TYPE_NULL;
            let codec_str = if codec_params.codec != codec_name {
                let id = codec_params.codec;
                // Symphonia codec ids are integers; convert to a human-readable name.
                codec_id_to_name(id)
            } else {
                label.to_lowercase()
            };
            metadata.insert("codec".to_string(), codec_str.clone());

            // Sample rate
            if let Some(rate) = codec_params.sample_rate {
                metadata.insert("sample_rate".to_string(), rate.to_string());
                text_parts.push(format!("Sample rate: {} Hz", rate));
            }

            // Channels
            if let Some(channels) = codec_params.channels {
                let ch_count = channels.count();
                metadata.insert("channels".to_string(), ch_count.to_string());
                let ch_label = match ch_count {
                    1 => "Mono".to_string(),
                    2 => "Stereo".to_string(),
                    n => format!("{} channels", n),
                };
                text_parts.push(ch_label);
            }

            // Duration
            if let Some(n_frames) = codec_params.n_frames {
                if let Some(rate) = codec_params.sample_rate {
                    if rate > 0 {
                        let seconds = n_frames as f64 / rate as f64;
                        let formatted = format_duration(seconds);
                        metadata.insert("duration_seconds".to_string(), format!("{:.2}", seconds));
                        metadata.insert("duration".to_string(), formatted.clone());
                        text_parts.push(format!("Duration: {}", formatted));
                    }
                }
            }

            text_parts.push(format!("Codec: {}", codec_str));
        }

        // --- Tag metadata (title, artist, album, etc.) ---
        let mut title_tag: Option<String> = None;

        // Extract tags from the probe metadata (often ID3v2/Vorbis comments
        // attached before the first audio packet).
        if let Some(rev) = probed.metadata.get().as_ref().and_then(|m| m.current()) {
            extract_tags(rev.tags(), &mut metadata, &mut text_parts, &mut title_tag);
        }

        let text = text_parts.join(". ");

        Ok(ParsedDocument {
            text,
            title: title_tag,
            language: None,
            content_type: ContentType::Audio,
            metadata,
        })
    }

    /// Produce a minimal `ParsedDocument` when symphonia cannot probe the file.
    /// This ensures audio files are still indexed with at least extension-level
    /// information rather than causing a pipeline error.
    fn build_fallback(
        &self,
        label: &str,
        extension: Option<&str>,
        size_bytes: usize,
    ) -> SiftResult<ParsedDocument> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "extension".to_string(),
            extension.unwrap_or("audio").to_string(),
        );
        metadata.insert("size_bytes".to_string(), size_bytes.to_string());
        metadata.insert("parse_method".to_string(), "fallback".to_string());

        let text = format!(
            "Audio file ({}). Format could not be fully parsed. Size: {} bytes",
            label, size_bytes
        );

        Ok(ParsedDocument {
            text,
            title: None,
            language: None,
            content_type: ContentType::Audio,
            metadata,
        })
    }
}

/// Convert a symphonia codec type id to a human-readable name.
#[cfg(feature = "audio")]
fn codec_id_to_name(id: symphonia::core::codecs::CodecType) -> String {
    use symphonia::core::codecs;

    match id {
        codecs::CODEC_TYPE_MP3 => "mp3".to_string(),
        codecs::CODEC_TYPE_AAC => "aac".to_string(),
        codecs::CODEC_TYPE_FLAC => "flac".to_string(),
        codecs::CODEC_TYPE_VORBIS => "vorbis".to_string(),
        codecs::CODEC_TYPE_OPUS => "opus".to_string(),
        codecs::CODEC_TYPE_PCM_S16LE => "pcm_s16le".to_string(),
        codecs::CODEC_TYPE_PCM_S16BE => "pcm_s16be".to_string(),
        codecs::CODEC_TYPE_PCM_S24LE => "pcm_s24le".to_string(),
        codecs::CODEC_TYPE_PCM_S32LE => "pcm_s32le".to_string(),
        codecs::CODEC_TYPE_PCM_F32LE => "pcm_f32le".to_string(),
        codecs::CODEC_TYPE_PCM_F64LE => "pcm_f64le".to_string(),
        codecs::CODEC_TYPE_ALAC => "alac".to_string(),
        other => format!("codec({:?})", other),
    }
}

/// Extract recognized tags from a slice of symphonia tags into the metadata
/// map and text parts, setting the title if found.
#[cfg(feature = "audio")]
fn extract_tags(
    tags: &[symphonia::core::meta::Tag],
    metadata: &mut HashMap<String, String>,
    text_parts: &mut Vec<String>,
    title_tag: &mut Option<String>,
) {
    for tag in tags {
        let key = if let Some(std_key) = tag.std_key {
            format!("{:?}", std_key)
        } else {
            tag.key.to_string()
        };
        let value = tag.value.to_string();

        let key_lower = key.to_lowercase();
        if key_lower.contains("title") || key_lower == "tracktitle" {
            *title_tag = Some(value.clone());
            metadata.insert("title".to_string(), value.clone());
            text_parts.push(format!("Title: {}", value));
        } else if key_lower.contains("artist") {
            metadata.insert("artist".to_string(), value.clone());
            text_parts.push(format!("Artist: {}", value));
        } else if key_lower.contains("album") && !key_lower.contains("artist") {
            metadata.insert("album".to_string(), value.clone());
            text_parts.push(format!("Album: {}", value));
        } else if key_lower.contains("genre") {
            metadata.insert("genre".to_string(), value.clone());
            text_parts.push(format!("Genre: {}", value));
        } else if key_lower.contains("date") || key_lower.contains("year") {
            metadata.insert("year".to_string(), value.clone());
        }
    }
}

/// Format a duration in seconds into a human-readable string.
///
/// - Less than one hour: "M:SS" (e.g. "3:45")
/// - One hour or more:   "H:MM:SS" (e.g. "1:02:30")
#[cfg(feature = "audio")]
fn format_duration(seconds: f64) -> String {
    let total_secs = seconds.round() as u64;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;

    if h > 0 {
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        format!("{}:{:02}", m, s)
    }
}

#[cfg(test)]
#[cfg(feature = "audio")]
mod tests {
    use super::*;

    // ---- can_parse tests ----

    #[test]
    fn test_can_parse_audio_types() {
        let parser = AudioParser;

        // MIME type checks
        assert!(parser.can_parse(Some("audio/mpeg"), None));
        assert!(parser.can_parse(Some("audio/wav"), None));
        assert!(parser.can_parse(Some("audio/flac"), None));
        assert!(parser.can_parse(Some("audio/ogg"), None));
        assert!(parser.can_parse(Some("audio/aac"), None));
        assert!(parser.can_parse(Some("audio/mp4"), None));
        assert!(parser.can_parse(Some("audio/x-wav"), None));

        // Extension checks
        assert!(parser.can_parse(None, Some("mp3")));
        assert!(parser.can_parse(None, Some("wav")));
        assert!(parser.can_parse(None, Some("flac")));
        assert!(parser.can_parse(None, Some("ogg")));
        assert!(parser.can_parse(None, Some("aac")));
        assert!(parser.can_parse(None, Some("m4a")));
        assert!(parser.can_parse(None, Some("wma")));
        assert!(parser.can_parse(None, Some("opus")));

        // Case-insensitive
        assert!(parser.can_parse(None, Some("MP3")));
        assert!(parser.can_parse(None, Some("Flac")));
        assert!(parser.can_parse(Some("Audio/MPEG"), None));

        // Both provided
        assert!(parser.can_parse(Some("audio/mpeg"), Some("mp3")));
    }

    #[test]
    fn test_cannot_parse_non_audio() {
        let parser = AudioParser;

        assert!(!parser.can_parse(Some("text/plain"), None));
        assert!(!parser.can_parse(Some("image/png"), None));
        assert!(!parser.can_parse(Some("application/pdf"), None));
        assert!(!parser.can_parse(None, Some("txt")));
        assert!(!parser.can_parse(None, Some("rs")));
        assert!(!parser.can_parse(None, Some("jpg")));
        assert!(!parser.can_parse(None, None));
    }

    // ---- format_duration tests ----

    #[test]
    fn test_format_duration() {
        // Zero
        assert_eq!(format_duration(0.0), "0:00");

        // Seconds only
        assert_eq!(format_duration(5.0), "0:05");
        assert_eq!(format_duration(59.0), "0:59");

        // Minutes and seconds
        assert_eq!(format_duration(60.0), "1:00");
        assert_eq!(format_duration(61.0), "1:01");
        assert_eq!(format_duration(225.0), "3:45");
        assert_eq!(format_duration(599.0), "9:59");

        // Hours
        assert_eq!(format_duration(3600.0), "1:00:00");
        assert_eq!(format_duration(3750.0), "1:02:30");
        assert_eq!(format_duration(7384.0), "2:03:04");

        // Fractional seconds (should round)
        assert_eq!(format_duration(225.4), "3:45");
        assert_eq!(format_duration(225.6), "3:46");
    }

    // ---- fallback / error handling tests ----

    #[test]
    fn test_parse_fallback_on_invalid_content() {
        let parser = AudioParser;

        // Feed garbage bytes that are not a valid audio container.
        let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33];
        let doc = parser
            .parse(&garbage, Some("audio/mpeg"), Some("mp3"))
            .expect("fallback should not return an error");

        assert_eq!(doc.content_type, ContentType::Audio);
        assert!(doc.text.contains("Audio file"));
        assert!(doc.text.contains("MP3"));
        assert!(doc.text.contains("could not be fully parsed"));
        assert_eq!(
            doc.metadata.get("parse_method").map(|s| s.as_str()),
            Some("fallback")
        );
        assert_eq!(
            doc.metadata.get("size_bytes").map(|s| s.as_str()),
            Some("8")
        );
    }

    #[test]
    fn test_parse_fallback_unknown_extension() {
        let parser = AudioParser;

        let garbage = vec![0x00; 16];
        let doc = parser
            .parse(&garbage, Some("audio/x-wav"), Some("wav"))
            .expect("fallback should succeed for WAV with bad content");

        assert_eq!(doc.content_type, ContentType::Audio);
        assert!(doc.text.contains("WAV"));
    }

    #[test]
    fn test_parse_empty_content() {
        let parser = AudioParser;

        let doc = parser
            .parse(&[], Some("audio/flac"), Some("flac"))
            .expect("empty content should trigger fallback, not panic");

        assert_eq!(doc.content_type, ContentType::Audio);
        assert!(doc.text.contains("FLAC"));
        assert!(doc.text.contains("0 bytes"));
    }

    #[test]
    fn test_parser_name() {
        let parser = AudioParser;
        assert_eq!(parser.name(), "audio");
    }

    #[test]
    fn test_format_label_mapping() {
        assert_eq!(AudioParser::format_label(Some("mp3")), "MP3");
        assert_eq!(AudioParser::format_label(Some("wav")), "WAV");
        assert_eq!(AudioParser::format_label(Some("flac")), "FLAC");
        assert_eq!(AudioParser::format_label(Some("ogg")), "OGG");
        assert_eq!(AudioParser::format_label(Some("aac")), "AAC");
        assert_eq!(AudioParser::format_label(Some("m4a")), "M4A");
        assert_eq!(AudioParser::format_label(Some("wma")), "WMA");
        assert_eq!(AudioParser::format_label(Some("opus")), "Opus");
        assert_eq!(AudioParser::format_label(Some("xyz")), "Audio");
        assert_eq!(AudioParser::format_label(None), "Audio");
    }

    #[test]
    fn test_codec_id_to_name_known() {
        use symphonia::core::codecs;
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_MP3), "mp3");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_FLAC), "flac");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_AAC), "aac");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_VORBIS), "vorbis");
    }
}
