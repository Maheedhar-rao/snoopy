# Copyright (c) 2026 CrocCRM. All rights reserved.
# Proprietary and confidential. See LICENSE for terms.

"""
The advance out of a funder's approval email, by label rather than by symbol.

The regex this replaces looked for a dollar sign followed by a number. Loan23
prints the dollar sign in the row *label* and never on the figures:

    PURCHASE PRICE
    30,500.00
    ...
    Origination fee, $
    2,135.00

Its emails contain exactly two "$" characters, both against fee rows, so a
symbol-anchored pattern could only ever return a fee. It did, three times in ten
days - $1,190, $1,305, $2,135 against advances of $7,000, $14,500 and $30,500 -
and a rep was emailed each one. A different fee each time, because the pattern
needs four characters of digits and commas, so a fee under $1,000 misses and it
falls through to the next "$" down the page.

So: find the advance by the name the funder gives it, and treat a figure sitting
against a fee label as unusable. When neither yields anything, return None. A
wrong amount is worse than a missing one - nothing downstream can tell a wrong
one is wrong, and both the arithmetic checks and the rep's email read better
with a gap than with a fee in the advance's place.
"""

from __future__ import annotations

import re
from typing import Optional

# Ordered: the first label that produces a number wins. "purchase price" leads
# because Loan23 also prints NET FUNDING AMOUNT further down, which is the
# advance after the origination fee and the UCC - not what was approved.
#
# (label, how far the figure may sit from it). A column header in a term sheet
# can be a long way from its first value - SmartStep runs "Purchase Price Factor
# Rate (Yield) Upsell Specified Percentage Specified Amount Duration (Months)
# Commission CC or TDA Account Setup Fee" before printing one - but a phrase
# that also occurs in prose must stay close to its figure, or "approved for
# funding. Review your offer and request your contract in minutes." swallows a
# sentence and returns whatever digit it lands on.
_ADVANCE_LABELS = (
    (r"purchase[d]?\s+price", 200),
    (r"advance[d]?\s+amount", 200),
    (r"(?<!net )funding\s+amount", 200),
    (r"funded\s+amount", 200),
    (r"approval\s+amount", 60),
    (r"approved\s+(?:for|amount)", 25),
)

# Nothing numeric may sit between a label and its figure.
_GAP = r"[^0-9]{{0,{0}}}"

# An advance is thousands: either comma-grouped, or four or more bare digits.
# Without this floor the patterns below match the "5" in "5 business days" and
# the symbol fallback matches the "$5" in a fee schedule - both of which this
# returned before the floor was added, in place of a $35,000 approval.
_FIGURE = r"([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{1,2})?|[0-9]{4,}(?:\.[0-9]{1,2})?)"

_LABELLED = tuple(
    re.compile(label + _GAP.format(gap) + _FIGURE, re.I | re.S)
    for label, gap in _ADVANCE_LABELS
)

# What the funder deducts or pays out, never what it advances. Checked against
# the text immediately preceding a figure, which is where the row's own label
# sits in every table layout seen so far.
_FEE_LABEL = re.compile(
    r"origination|broker|commission|\bucc\b|\bwire\b|setup|"
    r"processing|service charge|\bfee\b|\bfees\b",
    re.I,
)

_SYMBOL = re.compile(r"\$\s*" + _FIGURE)

# How far back to look for the label that belongs to a figure.
_LABEL_LOOKBEHIND = 60


def _to_int(raw: str) -> Optional[int]:
    try:
        return int(float(raw.replace(",", "")))
    except (TypeError, ValueError):
        return None


def advance_amount(text: Optional[str]) -> Optional[int]:
    """The advance in dollars, or None when the email does not clearly state one."""
    if not text:
        return None

    for pattern in _LABELLED:
        m = pattern.search(text)
        if m:
            value = _to_int(m.group(1))
            if value:
                return value

    # No label the funder named. A bare "$50,000 approved" is still worth
    # reading, but skip any figure whose own label calls it a fee.
    for m in _SYMBOL.finditer(text):
        preceding = text[max(0, m.start() - _LABEL_LOOKBEHIND):m.start()]
        if _FEE_LABEL.search(preceding):
            continue
        value = _to_int(m.group(1))
        if value:
            return value

    return None
