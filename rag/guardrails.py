"""Guardrails: PII detection, prompt injection defense, output toxicity filtering.

Three-layer security:
1. Input guardrails: Block prompt injection and detect PII before processing
2. Output guardrails: Filter toxic/harmful content from LLM responses
3. PII redaction: Detect and optionally redact sensitive information

All guards return a GuardrailResult with pass/fail and optional redacted content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    violations: list[str] = field(default_factory=list)
    redacted_text: str | None = None
    metadata: dict = field(default_factory=dict)


# ---- PII Detection ----

# Common PII patterns (US-focused, extensible)
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


class PIIDetector:
    """Detects and optionally redacts Personally Identifiable Information."""

    def __init__(self, patterns: dict[str, re.Pattern] | None = None, redact: bool = True):
        self.patterns = patterns or PII_PATTERNS
        self.redact = redact

    def check(self, text: str) -> GuardrailResult:
        violations = []
        redacted = text

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                violations.append(f"{pii_type}: {len(matches)} instance(s) detected")
                if self.redact:
                    redacted = pattern.sub(f"[{pii_type.upper()}_REDACTED]", redacted)

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            redacted_text=redacted if violations else None,
            metadata={"pii_types_found": [v.split(":")[0] for v in violations]},
        )


# ---- Prompt Injection Defense ----

# Known prompt injection patterns
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"(disregard|forget)\s+(all\s+)?(previous|above|prior|your)\s+", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+follow\s+(the|your)\s+(previous|original)\b", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if|though)\s+", re.IGNORECASE),
    re.compile(r"override\s+(your|the|all)\s+(instructions?|rules?|guidelines?)", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
    re.compile(r"```\s*system", re.IGNORECASE),
]

# Suspicious structural patterns (role markers in user input)
ROLE_INJECTION_PATTERNS = [
    re.compile(r"^(system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"###\s*(system|instruction|human|assistant)", re.IGNORECASE),
]


class PromptInjectionDetector:
    """Detects prompt injection attempts in user input."""

    def __init__(
        self,
        patterns: list[re.Pattern] | None = None,
        role_patterns: list[re.Pattern] | None = None,
        max_length: int = 5000,
    ):
        self.patterns = patterns or INJECTION_PATTERNS
        self.role_patterns = role_patterns or ROLE_INJECTION_PATTERNS
        self.max_length = max_length

    def check(self, text: str) -> GuardrailResult:
        violations = []

        # Length check
        if len(text) > self.max_length:
            violations.append(f"Input exceeds max length ({len(text)} > {self.max_length})")

        # Injection pattern matching
        for pattern in self.patterns:
            if pattern.search(text):
                violations.append(f"Injection pattern detected: {pattern.pattern[:50]}")

        # Role injection
        for pattern in self.role_patterns:
            if pattern.search(text):
                violations.append(f"Role injection detected: {pattern.pattern[:50]}")

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            metadata={"check_type": "prompt_injection"},
        )


# ---- Output Toxicity Filter ----

# Basic toxicity/harmful content indicators
TOXIC_PATTERNS = [
    re.compile(r"\b(kill|murder|harm|attack|bomb|weapon)\s+(yourself|them|people|someone)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+)(hack|steal|break\s+into|exploit)\b", re.IGNORECASE),
    re.compile(r"\b(racial\s+slur|hate\s+speech)\b", re.IGNORECASE),
]

# Content that shouldn't appear in RAG answers
HALLUCINATION_INDICATORS = [
    re.compile(r"as\s+an\s+AI\s+(language\s+)?model", re.IGNORECASE),
    re.compile(r"I\s+don'?t\s+have\s+access\s+to\s+(real-?time|current)", re.IGNORECASE),
    re.compile(r"my\s+(training|knowledge)\s+(data|cutoff)", re.IGNORECASE),
]


class OutputGuardrail:
    """Filters toxic, harmful, or hallucinated content from LLM output."""

    def __init__(
        self,
        toxic_patterns: list[re.Pattern] | None = None,
        hallucination_patterns: list[re.Pattern] | None = None,
    ):
        self.toxic_patterns = toxic_patterns or TOXIC_PATTERNS
        self.hallucination_patterns = hallucination_patterns or HALLUCINATION_INDICATORS

    def check(self, text: str) -> GuardrailResult:
        violations = []

        for pattern in self.toxic_patterns:
            if pattern.search(text):
                violations.append(f"Toxic content: {pattern.pattern[:50]}")

        for pattern in self.hallucination_patterns:
            if pattern.search(text):
                violations.append(f"Hallucination indicator: {pattern.pattern[:50]}")

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            metadata={"check_type": "output_toxicity"},
        )


# ---- Composite Guardrail ----

class GuardrailPipeline:
    """Orchestrates all guardrail checks (input and output)."""

    def __init__(
        self,
        enable_pii: bool = True,
        enable_injection: bool = True,
        enable_output: bool = True,
        pii_redact: bool = True,
    ):
        self.pii_detector = PIIDetector(redact=pii_redact) if enable_pii else None
        self.injection_detector = PromptInjectionDetector() if enable_injection else None
        self.output_guardrail = OutputGuardrail() if enable_output else None

    def check_input(self, text: str) -> GuardrailResult:
        """Run all input guardrails. Returns combined result."""
        all_violations = []
        redacted = None

        if self.injection_detector:
            result = self.injection_detector.check(text)
            if not result.passed:
                all_violations.extend(result.violations)

        if self.pii_detector:
            result = self.pii_detector.check(text)
            if not result.passed:
                all_violations.extend(result.violations)
                redacted = result.redacted_text

        return GuardrailResult(
            passed=len(all_violations) == 0,
            violations=all_violations,
            redacted_text=redacted,
            metadata={"stage": "input"},
        )

    def check_output(self, text: str) -> GuardrailResult:
        """Run output guardrails on LLM response."""
        if not self.output_guardrail:
            return GuardrailResult(passed=True)

        return self.output_guardrail.check(text)
