use std::borrow::Cow;

/// 默认的知识片段字符数限制。
pub const DEFAULT_SEGMENT_CHAR_LIMIT: usize = 500;

/// 将原始文本拆分为不超过 `max_length` 字符的片段。
///
/// 拆分流程参考了 Python 版本的 NLTK + 回退策略：
/// 1. 优先按照中英文句号、问号、感叹号断句；
/// 2. 若无法断句，则按段落（换行）拆分；
/// 3. 最后兜底按照固定长度进行硬切分。
///
/// 返回的每个片段都会被 `trim`，并且不会为空。
pub fn split_text_segments(text: &str, max_length: usize) -> Vec<String> {
    if text.trim().is_empty() || max_length == 0 {
        return Vec::new();
    }

    let sentences = sentence_candidates(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }

        let sentence_len = trimmed.chars().count();
        if sentence_len > max_length {
            if !current.is_empty() {
                segments.push(current.trim().to_string());
                current.clear();
            }
            segments.extend(chunk_long_sentence(trimmed, max_length));
            continue;
        }

        let current_len = current.chars().count();
        let required = if current.is_empty() {
            sentence_len
        } else {
            current_len + 1 + sentence_len
        };

        if required > max_length && !current.is_empty() {
            segments.push(current.trim().to_string());
            current.clear();
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(trimmed);
    }

    if !current.is_empty() {
        segments.push(current.trim().to_string());
    }

    segments
}

fn sentence_candidates(text: &str) -> Vec<Cow<'_, str>> {
    let mut candidates = split_by_punctuation(text);
    if candidates.iter().all(|segment| segment.trim().is_empty()) {
        candidates = split_by_newline(text);
    }
    if candidates.iter().all(|segment| segment.trim().is_empty()) {
        candidates = split_by_length(text, DEFAULT_SEGMENT_CHAR_LIMIT)
            .into_iter()
            .map(Cow::Owned)
            .collect();
    }
    candidates
}

fn split_by_punctuation(text: &str) -> Vec<Cow<'_, str>> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();

    for (idx, ch) in chars.iter().enumerate() {
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？' | ';' | '；' | '…') {
            let end = idx + 1;
            if end > start {
                let slice: String = chars[start..end].iter().collect();
                sentences.push(Cow::Owned(slice));
            }
            start = end;
            while start < chars.len() && chars[start].is_whitespace() {
                start += 1;
            }
        }
    }

    if start < chars.len() {
        let slice: String = chars[start..].iter().collect();
        sentences.push(Cow::Owned(slice));
    }

    sentences
}

fn split_by_newline(text: &str) -> Vec<Cow<'_, str>> {
    text.split('\n')
        .map(|line| Cow::Owned(line.to_string()))
        .collect()
}

fn split_by_length(text: &str, max_length: usize) -> Vec<String> {
    if max_length == 0 {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut buffer = String::new();

    for ch in text.chars() {
        buffer.push(ch);
        if buffer.chars().count() == max_length {
            segments.push(buffer.clone());
            buffer.clear();
        }
    }

    if !buffer.trim().is_empty() {
        segments.push(buffer);
    }

    segments
}

fn chunk_long_sentence(sentence: &str, max_length: usize) -> Vec<String> {
    if max_length == 0 {
        return Vec::new();
    }

    let chars: Vec<char> = sentence.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + max_length).min(chars.len());
        let slice: String = chars[start..end].iter().collect();
        if !slice.trim().is_empty() {
            chunks.push(slice.trim().to_string());
        }
        start = end;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_handles_empty_input() {
        assert!(split_text_segments("   ", 500).is_empty());
        assert!(split_text_segments("text", 0).is_empty());
    }

    #[test]
    fn split_respects_max_length() {
        let text = "这是第一句。这是第二句，长度会超过限制，需要被拆分。";
        let segments = split_text_segments(text, 10);
        assert!(!segments.is_empty());
        assert!(segments.iter().all(|s| s.chars().count() <= 10));
    }

    #[test]
    fn split_falls_back_to_length_when_needed() {
        let text = "无标点且很长的段落需要被切割成多个部分用于向量库存储";
        let segments = split_text_segments(text, 8);
        assert!(segments.len() > 1);
        assert!(segments.iter().all(|s| s.chars().count() <= 8));
    }

    #[test]
    fn chunk_long_sentence_preserves_content() {
        let text = "abcdef";
        let chunks = chunk_long_sentence(text, 2);
        assert_eq!(chunks, vec!["ab", "cd", "ef"]);
    }
}
