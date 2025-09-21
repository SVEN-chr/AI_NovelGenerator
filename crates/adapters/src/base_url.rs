use once_cell::sync::Lazy;
use regex::Regex;

static VERSION_SUFFIX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"/v\d+$").unwrap());

pub fn check_base_url(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.ends_with('#') {
        return trimmed.trim_end_matches('#').to_string();
    }

    if !VERSION_SUFFIX_RE.is_match(trimmed) && !trimmed.contains("/v1") {
        let without_slash = trimmed.trim_end_matches('/');
        format!("{}/v1", without_slash)
    } else {
        trimmed.to_string()
    }
}

pub fn ensure_openai_base_url_has_v1(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if VERSION_SUFFIX_RE.is_match(trimmed) || trimmed.contains("/v1") {
        trimmed.to_string()
    } else {
        let without_slash = trimmed.trim_end_matches('/');
        format!("{}/v1", without_slash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_base_url_appends_v1_when_missing() {
        assert_eq!(
            check_base_url("https://example.com"),
            "https://example.com/v1"
        );
    }

    #[test]
    fn check_base_url_keeps_existing_version() {
        assert_eq!(
            check_base_url("https://example.com/v2"),
            "https://example.com/v2"
        );
    }

    #[test]
    fn check_base_url_respects_hash_suffix() {
        assert_eq!(
            check_base_url("https://example.com/#"),
            "https://example.com/"
        );
    }

    #[test]
    fn ensure_base_url_appends_v1() {
        assert_eq!(
            ensure_openai_base_url_has_v1("https://example.com"),
            "https://example.com/v1"
        );
    }
}
