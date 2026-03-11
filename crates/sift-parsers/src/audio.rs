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
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
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
        let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());
        let hint = Self::build_hint(mime_type, extension);

        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();

        let probe_result =
            symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts);

        match probe_result {
            Ok(mut probed) => Self::build_from_probe(&mut probed, label, extension),
            Err(e) => {
                debug!(
                    error = %e,
                    ext = ?extension,
                    "Symphonia probe failed, falling back to basic metadata"
                );
                Self::build_fallback(label, extension, content.len())
            }
        }
    }

    fn name(&self) -> &'static str {
        "audio"
    }
}

#[cfg(feature = "audio")]
impl AudioParser {
    /// Build a `ParsedDocument` from a successful symphonia probe.
    fn build_from_probe(
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
                        let seconds = n_frames as f64 / f64::from(rate);
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
            tag.key.clone()
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

    #[test]
    fn test_codec_id_to_name_more() {
        use symphonia::core::codecs;
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_OPUS), "opus");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_PCM_S16LE), "pcm_s16le");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_PCM_S16BE), "pcm_s16be");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_PCM_S24LE), "pcm_s24le");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_PCM_S32LE), "pcm_s32le");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_PCM_F32LE), "pcm_f32le");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_PCM_F64LE), "pcm_f64le");
        assert_eq!(codec_id_to_name(codecs::CODEC_TYPE_ALAC), "alac");
    }

    #[test]
    fn test_codec_id_to_name_unknown() {
        use symphonia::core::codecs;
        let name = codec_id_to_name(codecs::CODEC_TYPE_NULL);
        assert!(name.starts_with("codec("));
    }

    #[test]
    fn test_build_hint() {
        // Both extension and MIME
        let hint = AudioParser::build_hint(Some("audio/mpeg"), Some("mp3"));
        // We can't easily inspect the Hint, but it should not panic
        let _ = hint;

        // Extension only
        let hint = AudioParser::build_hint(None, Some("wav"));
        let _ = hint;

        // MIME only
        let hint = AudioParser::build_hint(Some("audio/flac"), None);
        let _ = hint;

        // Neither
        let hint = AudioParser::build_hint(None, None);
        let _ = hint;
    }

    /// Construct a minimal valid WAV file (PCM, mono, 44100Hz, 16-bit)
    fn make_minimal_wav() -> Vec<u8> {
        let num_samples: u32 = 100;
        let num_channels: u16 = 1;
        let sample_rate: u32 = 44100;
        let bits_per_sample: u16 = 16;
        let byte_rate: u32 = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
        let block_align: u16 = num_channels * bits_per_sample / 8;
        let data_size: u32 = num_samples * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
        let file_size: u32 = 36 + data_size; // RIFF header size - 8 + data

        let mut wav = Vec::new();
        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        // fmt sub-chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // sub-chunk size for PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // audio format: PCM = 1
        wav.extend_from_slice(&num_channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());
        // data sub-chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        // Silence (all zeros)
        wav.extend(std::iter::repeat_n(0u8, data_size as usize));
        wav
    }

    #[test]
    fn test_parse_valid_wav() {
        let parser = AudioParser;
        let wav = make_minimal_wav();
        let doc = parser
            .parse(&wav, Some("audio/wav"), Some("wav"))
            .expect("WAV parsing should succeed");

        assert_eq!(doc.content_type, ContentType::Audio);
        assert!(doc.text.contains("Audio file"));
        assert!(doc.text.contains("WAV"));
        // Should have extracted codec/sample-rate info from the probe
        assert!(
            doc.metadata.contains_key("codec") || doc.metadata.contains_key("sample_rate"),
            "Expected probed metadata, got: {:?}",
            doc.metadata
        );
        // Should NOT be a fallback parse
        assert_ne!(
            doc.metadata.get("parse_method").map(|s| s.as_str()),
            Some("fallback")
        );
    }

    #[test]
    fn test_parse_wav_has_sample_rate() {
        let parser = AudioParser;
        let wav = make_minimal_wav();
        let doc = parser.parse(&wav, Some("audio/wav"), Some("wav")).unwrap();

        if let Some(rate) = doc.metadata.get("sample_rate") {
            assert_eq!(rate, "44100");
        }
    }

    #[test]
    fn test_parse_wav_has_channels() {
        let parser = AudioParser;
        let wav = make_minimal_wav();
        let doc = parser.parse(&wav, Some("audio/wav"), Some("wav")).unwrap();

        if let Some(ch) = doc.metadata.get("channels") {
            assert_eq!(ch, "1");
        }
        // Mono audio
        assert!(doc.text.contains("Mono"));
    }

    #[test]
    fn test_parse_fallback_no_extension() {
        let parser = AudioParser;
        let garbage = vec![0xDE, 0xAD];
        let doc = parser
            .parse(&garbage, Some("audio/ogg"), None)
            .expect("fallback should work");
        assert_eq!(doc.metadata.get("extension").unwrap(), "audio");
    }

    #[test]
    fn test_extract_tags_title() {
        use symphonia::core::meta::{StandardTagKey, Tag, Value};
        let mut metadata = HashMap::new();
        let mut text_parts = Vec::new();
        let mut title_tag = None;

        let tags = vec![Tag {
            std_key: Some(StandardTagKey::TrackTitle),
            key: "TrackTitle".to_string(),
            value: Value::from("My Song"),
        }];

        extract_tags(&tags, &mut metadata, &mut text_parts, &mut title_tag);
        assert_eq!(title_tag, Some("My Song".to_string()));
        assert_eq!(metadata.get("title").unwrap(), "My Song");
        assert!(text_parts.iter().any(|p| p.contains("Title: My Song")));
    }

    #[test]
    fn test_extract_tags_artist() {
        use symphonia::core::meta::{StandardTagKey, Tag, Value};
        let mut metadata = HashMap::new();
        let mut text_parts = Vec::new();
        let mut title_tag = None;

        let tags = vec![Tag {
            std_key: Some(StandardTagKey::Artist),
            key: "Artist".to_string(),
            value: Value::from("The Band"),
        }];

        extract_tags(&tags, &mut metadata, &mut text_parts, &mut title_tag);
        assert_eq!(metadata.get("artist").unwrap(), "The Band");
        assert!(text_parts.iter().any(|p| p.contains("Artist: The Band")));
        assert!(title_tag.is_none());
    }

    #[test]
    fn test_extract_tags_album() {
        use symphonia::core::meta::{StandardTagKey, Tag, Value};
        let mut metadata = HashMap::new();
        let mut text_parts = Vec::new();
        let mut title_tag = None;

        let tags = vec![Tag {
            std_key: Some(StandardTagKey::Album),
            key: "Album".to_string(),
            value: Value::from("Greatest Hits"),
        }];

        extract_tags(&tags, &mut metadata, &mut text_parts, &mut title_tag);
        assert_eq!(metadata.get("album").unwrap(), "Greatest Hits");
        assert!(text_parts
            .iter()
            .any(|p| p.contains("Album: Greatest Hits")));
    }

    #[test]
    fn test_extract_tags_genre() {
        use symphonia::core::meta::{StandardTagKey, Tag, Value};
        let mut metadata = HashMap::new();
        let mut text_parts = Vec::new();
        let mut title_tag = None;

        let tags = vec![Tag {
            std_key: Some(StandardTagKey::Genre),
            key: "Genre".to_string(),
            value: Value::from("Rock"),
        }];

        extract_tags(&tags, &mut metadata, &mut text_parts, &mut title_tag);
        assert_eq!(metadata.get("genre").unwrap(), "Rock");
        assert!(text_parts.iter().any(|p| p.contains("Genre: Rock")));
    }

    #[test]
    fn test_extract_tags_date() {
        use symphonia::core::meta::{StandardTagKey, Tag, Value};
        let mut metadata = HashMap::new();
        let mut text_parts = Vec::new();
        let mut title_tag = None;

        let tags = vec![Tag {
            std_key: Some(StandardTagKey::Date),
            key: "Date".to_string(),
            value: Value::from("2024"),
        }];

        extract_tags(&tags, &mut metadata, &mut text_parts, &mut title_tag);
        assert_eq!(metadata.get("year").unwrap(), "2024");
    }

    #[test]
    fn test_extract_tags_non_std_key() {
        use symphonia::core::meta::{Tag, Value};
        let mut metadata = HashMap::new();
        let mut text_parts = Vec::new();
        let mut title_tag = None;

        // Tag with no standard key, but key string contains "title"
        let tags = vec![Tag {
            std_key: None,
            key: "title".to_string(),
            value: Value::from("Custom Title"),
        }];

        extract_tags(&tags, &mut metadata, &mut text_parts, &mut title_tag);
        assert_eq!(title_tag, Some("Custom Title".to_string()));
    }

    #[test]
    fn test_extract_tags_album_artist_not_album() {
        use symphonia::core::meta::{StandardTagKey, Tag, Value};
        let mut metadata = HashMap::new();
        let mut text_parts = Vec::new();
        let mut title_tag = None;

        // "AlbumArtist" contains "album" and "artist" -- the artist branch
        // should match first because of order
        let tags = vec![Tag {
            std_key: Some(StandardTagKey::AlbumArtist),
            key: "AlbumArtist".to_string(),
            value: Value::from("Various Artists"),
        }];

        extract_tags(&tags, &mut metadata, &mut text_parts, &mut title_tag);
        // AlbumArtist contains "artist", so it should go to the artist branch
        assert_eq!(metadata.get("artist").unwrap(), "Various Artists");
    }

    #[test]
    fn test_parse_wav_has_duration() {
        let parser = AudioParser;
        let wav = make_minimal_wav();
        let doc = parser.parse(&wav, Some("audio/wav"), Some("wav")).unwrap();

        // The WAV has 100 samples at 44100Hz, so duration should be present
        if let Some(dur) = doc.metadata.get("duration_seconds") {
            let secs: f64 = dur.parse().unwrap();
            assert!(secs >= 0.0);
        }
        // The text should contain "Codec:"
        assert!(doc.text.contains("Codec:"));
    }

    /// Construct a stereo WAV to test the "Stereo" channel label.
    fn make_stereo_wav() -> Vec<u8> {
        let num_samples: u32 = 100;
        let num_channels: u16 = 2;
        let sample_rate: u32 = 44100;
        let bits_per_sample: u16 = 16;
        let byte_rate: u32 = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
        let block_align: u16 = num_channels * bits_per_sample / 8;
        let data_size: u32 = num_samples * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
        let file_size: u32 = 36 + data_size;

        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&num_channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.extend(std::iter::repeat_n(0u8, data_size as usize));
        wav
    }

    #[test]
    fn test_parse_stereo_wav() {
        let parser = AudioParser;
        let wav = make_stereo_wav();
        let doc = parser.parse(&wav, Some("audio/wav"), Some("wav")).unwrap();

        assert!(doc.text.contains("Stereo"));
        if let Some(ch) = doc.metadata.get("channels") {
            assert_eq!(ch, "2");
        }
    }
}
