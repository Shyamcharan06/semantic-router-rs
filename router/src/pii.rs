use regex::Regex;
use std::sync::OnceLock;

/// Demo-grade heuristics, not a production compliance tool: plain regexes
/// over a handful of common PII shapes (email, US phone/SSN, a 16-digit
/// card format, IPv4). No Luhn check, no international formats.
struct Detector {
    kind: &'static str,
    regex: Regex,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PiiFinding {
    pub kind: &'static str,
    pub matched: String,
}

fn detectors() -> &'static Vec<Detector> {
    static DETECTORS: OnceLock<Vec<Detector>> = OnceLock::new();
    DETECTORS.get_or_init(|| {
        vec![
            Detector {
                kind: "email",
                regex: Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap(),
            },
            Detector {
                kind: "ssn",
                regex: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            },
            Detector {
                kind: "credit_card",
                regex: Regex::new(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b").unwrap(),
            },
            Detector {
                kind: "phone",
                regex: Regex::new(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
                    .unwrap(),
            },
            Detector {
                kind: "ip_address",
                regex: Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap(),
            },
        ]
    })
}

/// Returns every PII match found, without modifying `text`.
pub fn scan(text: &str) -> Vec<PiiFinding> {
    detectors()
        .iter()
        .flat_map(|d| {
            d.regex.find_iter(text).map(move |m| PiiFinding {
                kind: d.kind,
                matched: m.as_str().to_string(),
            })
        })
        .collect()
}

/// Replaces every PII match with a `[REDACTED_<KIND>]` placeholder and
/// returns the redacted text alongside what was found.
pub fn redact(text: &str) -> (String, Vec<PiiFinding>) {
    let mut findings = Vec::new();
    let mut result = text.to_string();

    for d in detectors() {
        let matches: Vec<String> = d
            .regex
            .find_iter(&result)
            .map(|m| m.as_str().to_string())
            .collect();
        for m in matches {
            findings.push(PiiFinding {
                kind: d.kind,
                matched: m,
            });
        }
        let placeholder = format!("[REDACTED_{}]", d.kind.to_uppercase());
        result = d
            .regex
            .replace_all(&result, placeholder.as_str())
            .into_owned();
    }

    (result, findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_email() {
        let findings = scan("reach me at jane.doe@example.com please");
        assert!(
            findings
                .iter()
                .any(|f| f.kind == "email" && f.matched == "jane.doe@example.com")
        );
    }

    #[test]
    fn detects_ssn() {
        let findings = scan("my ssn is 123-45-6789");
        assert!(
            findings
                .iter()
                .any(|f| f.kind == "ssn" && f.matched == "123-45-6789")
        );
    }

    #[test]
    fn detects_credit_card() {
        let findings = scan("card number 4111 1111 1111 1111 expires soon");
        assert!(findings.iter().any(|f| f.kind == "credit_card"));
    }

    #[test]
    fn clean_text_has_no_findings() {
        let findings = scan("Write a Python function to sort a list");
        assert!(findings.is_empty());
    }

    #[test]
    fn redact_replaces_email_and_reports_finding() {
        let (redacted, findings) = redact("contact jane.doe@example.com about this");
        assert_eq!(redacted, "contact [REDACTED_EMAIL] about this");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, "email");
    }

    #[test]
    fn redact_handles_multiple_pii_kinds() {
        let (redacted, findings) =
            redact("email jane@example.com or call 555-123-4567, ssn 123-45-6789");
        assert!(redacted.contains("[REDACTED_EMAIL]"));
        assert!(redacted.contains("[REDACTED_SSN]"));
        assert!(findings.len() >= 2);
    }
}
