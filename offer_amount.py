# Copyright (c) 2026 CrocCRM. All rights reserved.
# Proprietary and confidential. See LICENSE for terms.

"""The approval amount, minus the one case where it is provably a fee.

This is deliberately the existing symbol-anchored extraction with a single
exception, not a replacement for it. A label-driven rewrite was measured
against all 14,124 stored approval bodies and changed 252 of them - 70% inside
quoted reply chains, one picking an Offer ID out of a bulk table, twenty-two
returning Lily Advance's payback instead of its advance. Breadth was the
defect, so this keeps today's answer everywhere it is not demonstrably wrong.

Loan23 is where it is demonstrably wrong. They put the dollar sign at the end
of the row *label* and never on a figure:

    PURCHASE PRICE
    30,500.00
    ...
    Origination fee, $
    2,135.00

Their mail contains exactly two "$" characters, both closing a fee label, so
the pattern could only ever return a fee. It did three times in ten days -
$1,190, $1,305 and $2,135 against advances of $7,000, $14,500 and $30,500 -
and each went to a rep as the approval.

Measured on the same 14,124 bodies, this changes 4 and leaves 14,120 byte for
byte identical. All four are Loan23, and all four are those fees becoming the
purchase price.
"""

import re
from typing import Optional

# Unchanged from production, so an email whose first match is not a fee gets
# exactly the answer it gets today.
_SYMBOL = re.compile(r"\$\s*([0-9][0-9,]{3,})")

# Loan23 does not put the symbol on its figures - it puts it at the end of the
# row *label*: "Origination fee, $" then the number on the next line, and
# "Broker's fee, $" likewise. Those two are the only "$" in the mail, so the
# symbol-anchored pattern can only ever return a fee.
#
# The test is therefore that the "$" terminates a fee label, not that a fee
# word appears somewhere nearby. A looser window skipped correct figures whose
# column header happened to contain the word - "Funding Amount Factor Fee
# Payback" above the real number - and the next match down was usually the
# payback or the daily payment, which is worse than what it replaced.
# The gaps are [ \t] and not \s: the "$" has to sit on the same line as the
# fee word it belongs to. SmartStep ends a column header with "... Commission
# CC or TDA Account Setup Fee" and puts $73,000.00 - the purchase price - on
# the next line; treating that as a fee row skipped it and returned $100,739,
# the total repayment. Loan23's "Origination fee, $" keeps the symbol on the
# label's own line, which is the shape being matched here.
_FEE_LABEL_END = re.compile(
    r"(?:origination|broker'?s?|commission|ucc|processing|service|setup)?"
    # \Z, not $: in Python "$" also matches just before a trailing newline,
    # which let "... Account Setup Fee\n" count as ending at the symbol.
    r"[ \t]*(?:fee|charge)s?[ \t]*[:,\-]?[ \t]*\Z",
    re.I,
)

# Only consulted when every "$" in the email is against a fee. Loan23 is the
# case: its figures carry no symbol at all and its only two "$" characters are
# in the "Origination fee, $" and "Broker's fee, $" labels.
_ADVANCE_LABEL = re.compile(
    r"(?:purchase[d]?\s+price|advance[d]?\s+amount|(?<!net )funding\s+amount)"
    r"[^0-9]{0,40}"
    r"([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{1,2})?|[0-9]{4,}(?:\.[0-9]{1,2})?)",
    re.I | re.S,
)

_LABEL_LOOKBEHIND = 45


def _to_int(raw: str) -> Optional[int]:
    try:
        return int(float(raw.replace(",", "")))
    except (TypeError, ValueError):
        return None


def advance_amount(text: Optional[str]) -> Optional[int]:
    if not text:
        return None

    saw_fee = False
    for m in _SYMBOL.finditer(text):
        preceding = text[max(0, m.start() - _LABEL_LOOKBEHIND):m.start()]
        if _FEE_LABEL_END.search(preceding):
            saw_fee = True
            continue
        return _to_int(m.group(1))

    # Every "$" in the mail was against a fee, or there were none at all.
    if saw_fee:
        m = _ADVANCE_LABEL.search(text)
        if m:
            return _to_int(m.group(1))
    return None
