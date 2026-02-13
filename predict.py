#!/usr/bin/env python3
"""
Inference module for the trained DistilBERT email classifier.

Usage as module:
    from predict import EmailClassifierLocal, get_classifier

    clf = get_classifier()
    result = clf.predict(subject="Approved - Business Name", body="We are pleased to offer...")
    # {"response_type": "APPROVED", "confidence": 0.97, "classification_method": "distilbert"}

Usage as CLI:
    python predict.py "Subject here" "Body text here"
"""

import json
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "email_classifier"

# Thread-safe singleton
_instance: Optional["EmailClassifierLocal"] = None
_lock = threading.Lock()


class EmailClassifierLocal:
    """DistilBERT-based email classifier. Thread-safe for Flask/FastAPI."""

    def __init__(self, model_path: str = str(MODEL_DIR)):
        model_path = Path(model_path)
        meta_path = model_path / "meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run train.py first."
            )

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.labels = self.meta["labels"]
        self.max_length = self.meta.get("max_length", 512)
        self.model_version = self.meta.get("version", "v1")

        # Pick device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load model + tokenizer once
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
        self.model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

    # ── Evidence patterns (scanned for ALL classes, independently of BERT) ──

    APPROVAL_SIGNALS = [
        (r"(?:we(?:'re| are) )?pleased to offer", "pleased to offer", 3),
        (r"(?:amount|offer|approved for)[:\s]*\$\s*[0-9][0-9,]{2,}", "offer amount", 3),
        (r"clear to fund|ctf\b", "clear to fund", 3),
        (r"\bapproved?\b(?!.*(?:update|review now|active deals|portal))", "approved", 2),
        (r"(?:factor|buy rate)[^\d]{0,10}[0-9]\.\d{1,2}", "factor rate", 2),
        (r"(?:term|repay)[:\s]*\d+\s*(?:day|week|month)", "term offered", 2),
        (r"\bfunded?\b", "funded", 1),
        (r"green light", "green light", 2),
        (r"contingent to final underwriting", "contingent approval", 2),
        # MCA / structured offer patterns
        (r"(?:mca|funding|preliminary)\s+(?:funding\s+)?offer", "funding offer", 3),
        (r"here is (?:our|the|your)\s+.{0,20}offer", "here is offer", 3),
        (r"funding\s+factor\s+payback", "offer table", 3),
        (r"daily\s+(?:pmt|payment|pymt)", "daily payment", 2),
        (r"weekly\s+(?:pmt|payment|pymt)", "weekly payment", 2),
        (r"(?:we (?:would|can|are able to|want to) )?(?:fund|finance) this deal", "fund this deal", 3),
        (r"congratulations.{0,30}(?:deal|offer|approved|funded)", "congratulations", 3),
    ]

    DECLINE_SIGNALS = [
        (r"too many (?:advances|positions|mcas)", "too many advances", 2),
        (r"high (?:utilization|risk)", "high utilization/risk", 2),
        (r"low (?:revenue|deposits|balance)", "low revenue/deposits", 2),
        (r"negative (?:balance|days)", "negative balance", 2),
        (r"\bnsf\b|insufficient funds", "NSF", 2),
        (r"time in business", "time in business", 2),
        (r"bankruptcy|tax lien|default", "bankruptcy/lien/default", 2),
        (r"industry|sic code|prohibited", "industry", 1),
        (r"not a (?:good )?fit", "not a fit", 2),
        (r"(?:cannot|can't|won't|unable to) (?:fund|proceed|assist|offer|approve)", "unable to fund", 2),
        (r"we(?:'re| are) passing|passing on", "passing", 2),
        (r"\bdecline[d]?\b|\bdenied\b", "declined", 2),
        (r"\bno room\b", "no room", 2),
    ]

    STIPS_SIGNALS = [
        (r"(?:need|require|provide|send|missing).{0,30}(?:bank statement|tax return|void.{0,5}check|driver|license|proof)", "stips requested", 2),
        (r"(?:additional|more) (?:doc|info)", "more docs needed", 2),
        (r"stipulation", "stipulations", 2),
    ]

    # Granular stips document patterns for extraction
    STIPS_DOCUMENTS = [
        (r"(\d+)\s*(?:months?\s+)?bank\s+statements?", "{n} months bank statements"),
        (r"bank\s+statements?", "bank statements"),
        (r"tax\s+returns?", "tax returns"),
        (r"void(?:ed)?\s+check", "voided check"),
        (r"driver'?s?\s+licen[sc]e", "driver's license"),
        (r"proof\s+of\s+(?:ownership|address|identity|income)", "proof of {match}"),
        (r"(?:p\s*&\s*l|profit\s*(?:&|and)\s*loss)", "P&L statement"),
        (r"business\s+licen[sc]e", "business license"),
        (r"(?:articles?\s+of\s+)?(?:incorporation|organization)", "articles of incorporation"),
        (r"lease\s+agreement", "lease agreement"),
        (r"(?:credit\s+card|merchant)\s+(?:processing\s+)?statements?", "merchant processing statements"),
        (r"(?:accounts?\s+)?receiv(?:able|ing)", "accounts receivable"),
        (r"balance\s+sheet", "balance sheet"),
        (r"photo\s+(?:id|identification)", "photo ID"),
        (r"ein\s+(?:letter|verification)", "EIN verification"),
        (r"application|signed\s+(?:contract|agreement)", "signed application"),
    ]

    OTHER_SIGNALS = [
        (r"auto.?reply|out of office|automatic reply", "auto-reply", 2),
        (r"thank you for (?:your |the )?submission", "submission ack", 1),
        (r"new submission|received your", "submission confirmation", 1),
        (r"unsubscribe|newsletter|marketing", "marketing", 1),
        (r"active deals.*(?:past|for)\s*\d+\s*days", "portal summary", 2),
        (r"(?:view|edit) offer.*(?:copyright|all rights)", "portal notification", 2),
    ]

    @classmethod
    def _extract_evidence(cls, text: str) -> Dict:
        """
        Scan email text for signals of ALL classes, independent of model prediction.

        Returns:
            {
                "APPROVED":  {"score": int, "signals": ["pleased to offer", "$40k"]},
                "DECLINED":  {"score": int, "signals": [...]},
                "STIPS_REQUIRED": {"score": int, "signals": [...]},
                "OTHER":     {"score": int, "signals": [...]},
            }
        """
        t = text.lower()
        evidence = {
            "APPROVED": {"score": 0, "signals": []},
            "DECLINED": {"score": 0, "signals": []},
            "STIPS_REQUIRED": {"score": 0, "signals": []},
            "OTHER": {"score": 0, "signals": []},
        }

        for pattern, label_text, weight in cls.APPROVAL_SIGNALS:
            if re.search(pattern, t):
                evidence["APPROVED"]["score"] += weight
                evidence["APPROVED"]["signals"].append(label_text)

        for pattern, label_text, weight in cls.DECLINE_SIGNALS:
            if re.search(pattern, t):
                evidence["DECLINED"]["score"] += weight
                evidence["DECLINED"]["signals"].append(label_text)

        for pattern, label_text, weight in cls.STIPS_SIGNALS:
            if re.search(pattern, t):
                evidence["STIPS_REQUIRED"]["score"] += weight
                evidence["STIPS_REQUIRED"]["signals"].append(label_text)

        for pattern, label_text, weight in cls.OTHER_SIGNALS:
            if re.search(pattern, t):
                evidence["OTHER"]["score"] += weight
                evidence["OTHER"]["signals"].append(label_text)

        # Extract offer details if approval signals found
        m = re.search(r"\$\s*([0-9][0-9,]{2,})", t)
        if m:
            evidence["APPROVED"]["signals"].append(f"${m.group(1)}")
        m = re.search(r"(factor|buy rate)[^\d]*([0-9]\.\d{1,2})", t)
        if m:
            evidence["APPROVED"]["signals"].append(f"factor {m.group(2)}")
        m = re.search(r"(\d{1,3})\s*(day|week|month)", t)
        if m:
            evidence["APPROVED"]["signals"].append(f"{m.group(1)} {m.group(2)}s")

        # Deduplicate signals
        for k in evidence:
            seen = set()
            unique = []
            for s in evidence[k]["signals"]:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            evidence[k]["signals"] = unique

        return evidence

    @staticmethod
    def _derive_sentiment(response_type: str) -> str:
        return {
            "APPROVED": "positive",
            "DECLINED": "negative",
            "STIPS_REQUIRED": "neutral",
            "OTHER": "neutral",
        }.get(response_type, "neutral")

    @staticmethod
    def _extract_decline_reasons(evidence: Dict) -> str:
        signals = evidence.get("DECLINED", {}).get("signals", [])
        return "; ".join(signals) if signals else ""

    @classmethod
    def _extract_stips_requested(cls, text: str) -> List[str]:
        t = text.lower()
        stips = []
        seen = set()
        for pattern, template in cls.STIPS_DOCUMENTS:
            m = re.search(pattern, t)
            if m:
                if "{n}" in template:
                    doc = template.replace("{n}", m.group(1))
                elif "{match}" in template:
                    # Extract the last word(s) from the match for "proof of ..."
                    full = m.group(0)
                    suffix = full.split("proof of ")[-1] if "proof of " in full else full
                    doc = template.replace("{match}", suffix)
                else:
                    doc = template
                if doc not in seen:
                    seen.add(doc)
                    stips.append(doc)
        return stips

    @classmethod
    def _build_summary(cls, response_type: str, evidence: Dict,
                       decline_reason: str, stips_requested: List[str]) -> str:
        if response_type == "APPROVED":
            details = []
            for sig in evidence.get("APPROVED", {}).get("signals", []):
                if sig.startswith("$"):
                    details.append(f"offer of {sig}")
                elif sig.startswith("factor "):
                    details.append(f"factor rate {sig.split()[-1]}")
            if details:
                return f"Lender approved with {', '.join(details)}."
            return "Lender approved the deal."

        if response_type == "DECLINED":
            if decline_reason:
                return f"Lender declined — {decline_reason}."
            return "Lender declined the deal."

        if response_type == "STIPS_REQUIRED":
            if stips_requested:
                return f"Lender requested additional documentation: {', '.join(stips_requested)}."
            return "Lender requested additional documentation before making a decision."

        # OTHER
        other_sigs = evidence.get("OTHER", {}).get("signals", [])
        if other_sigs:
            return f"Informational/non-decision email — {'; '.join(other_sigs[:2])}."
        return "Informational or non-decision email."

    def predict(
        self,
        subject: str = "",
        body: str = "",
        snippet: str = "",
    ) -> Dict:
        """
        Classify an email. Returns dict matching the existing Claude classifier contract.

        Returns:
            {
                "response_type": "APPROVED" | "DECLINED" | "STIPS_REQUIRED" | "OTHER",
                "confidence": float (0.0 - 1.0),
                "classification_method": "distilbert",
                "reason": "evidence-based explanation",
                "evidence": {...},          # signals found for ALL classes
                "approval_flag": bool,      # True = approval signals found but BERT disagrees
                "all_scores": {"APPROVED": 0.02, "DECLINED": 0.01, ...}
            }
        """
        # Build input text (same format as training data)
        text_body = body.strip() if body.strip() else snippet.strip()
        subject = subject.strip()
        combined = f"Subject: {subject}\n\n{text_body}" if subject else text_body
        combined = combined[:2000]

        if not combined.strip():
            return {
                "response_type": "OTHER",
                "confidence": 0.0,
                "classification_method": "distilbert",
                "model_version": self.model_version,
                "reason": "empty email",
                "evidence": {},
                "approval_flag": False,
                "all_scores": {l: 0.0 for l in self.labels},
                "sentiment": "neutral",
                "decline_reason": "",
                "summary": "Empty email — no content to classify.",
                "stips_requested": [],
            }

        # Tokenize
        encodings = self.tokenizer(
            combined,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        label = self.labels[pred_idx]

        all_scores = {self.labels[i]: round(probs[i].item(), 4) for i in range(len(self.labels))}

        # Extract evidence from ALL classes
        full_text = f"{subject}\n{text_body}"
        evidence = self._extract_evidence(full_text)

        # ── APPROVAL SAFETY NET ──
        # If BERT says NOT approved but we see strong approval evidence,
        # flag it so the caller forces Claude API fallback.
        approval_score = evidence["APPROVED"]["score"]
        approval_flag = False
        if label != "APPROVED" and approval_score >= 4:
            approval_flag = True

        # Build reason from the predicted label's evidence
        pred_signals = evidence.get(label, {}).get("signals", [])
        if pred_signals:
            reason = "; ".join(pred_signals[:4])
        elif label == "OTHER":
            reason = "informational or non-decision email"
        else:
            reason = f"{label.lower()} language detected"

        # If flagged, append the contradicting approval evidence to reason
        if approval_flag:
            appr_sigs = evidence["APPROVED"]["signals"]
            reason += f" | BUT found approval: {'; '.join(appr_sigs[:3])}"

        # Extract enrichment fields
        sentiment = self._derive_sentiment(label)
        decline_reason = self._extract_decline_reasons(evidence) if label == "DECLINED" else ""
        stips_requested = self._extract_stips_requested(full_text) if label == "STIPS_REQUIRED" else []
        summary = self._build_summary(label, evidence, decline_reason, stips_requested)

        return {
            "response_type": label,
            "confidence": round(confidence, 4),
            "classification_method": "distilbert",
            "model_version": self.model_version,
            "reason": reason,
            "evidence": evidence,
            "approval_flag": approval_flag,
            "all_scores": all_scores,
            "sentiment": sentiment,
            "decline_reason": decline_reason,
            "summary": summary,
            "stips_requested": stips_requested,
        }


def get_classifier() -> EmailClassifierLocal:
    """Get or create singleton classifier instance (thread-safe)."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        # Double-check after acquiring lock
        if _instance is None:
            _instance = EmailClassifierLocal()
    return _instance


def reload_classifier(model_path: str = None) -> str:
    """
    Hot-swap the singleton with a new model version. Thread-safe.

    Args:
        model_path: Path to new model directory. Defaults to MODEL_DIR.

    Returns:
        The model_version string of the newly loaded model.
    """
    global _instance
    path = model_path or str(MODEL_DIR)
    new_clf = EmailClassifierLocal(model_path=path)
    with _lock:
        _instance = new_clf
    return new_clf.model_version


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <subject> [body]")
        sys.exit(1)

    subject = sys.argv[1]
    body = sys.argv[2] if len(sys.argv) > 2 else ""

    clf = get_classifier()
    result = clf.predict(subject=subject, body=body)

    print(json.dumps(result, indent=2))
