# Copyright (c) 2026 CrocCRM. All rights reserved.
# Proprietary and confidential. See LICENSE for terms.

"""The advance, not the fee, out of a funder's approval email."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from offer_amount import advance_amount  # noqa: E402


# Loan23's real layout: the dollar sign is in the row *label* and never on a
# figure, so the whole email contains exactly two of them - both against fees.
LOAN23 = """Kickin Wingz & Thingz
PURCHASE PRICE
30,500.00
30,500.00
30,500.00
Purchased amount of future receipts (RTR)
45,445.00
45,445.00
45,445.00
Origination fee, %
7.00%
7.00%
7.00%
Origination fee, $
2,135.00
2,135.00
2,135.00
UCC
400.00
NET FUNDING AMOUNT
27,965.00
DAILY PAYMENT
649.22
Term (business days)
70
Broker's fee, $
4,575.00
"""


def test_loan23_returns_the_purchase_price_not_a_fee():
    """
    Three Loan23 approvals in ten days were stored as $1,190, $1,305 and $2,135
    against advances of $7,000, $14,500 and $30,500, and a rep was emailed each
    one. Every figure a dollar-sign match can reach in these emails is a fee.
    """
    assert advance_amount(LOAN23) == 30500

    # Not the RTR, and not the net funding amount - that is the advance after
    # the origination fee and the UCC, so it is not what was approved.
    assert advance_amount(LOAN23) not in (45445, 27965, 2135, 4575)


def test_the_label_wins_over_a_dollar_sign_further_up():
    body = "Origination fee, $\n1,305.00\nPURCHASE PRICE\n14,500.00\n"
    assert advance_amount(body) == 14500


@pytest.mark.parametrize("body,expected", [
    ("Advance Amount: 20,000.00 Factor Rate: 1.49999", 20000),   # no symbol at all
    ("Purchased Price\n$25,000.00\nPurchased Amount\n$37,500.00", 25000),
    ("You have been approved for $50,000 today", 50000),
    ("Approval Amount: $12,750.00", 12750),
])
def test_advances_the_funder_names(body, expected):
    assert advance_amount(body) == expected


def test_a_figure_beside_a_fee_label_is_not_an_advance():
    # With no advance label anywhere, a fee is still refused rather than
    # returned - a wrong amount is worse than a missing one.
    assert advance_amount("Origination fee, $\n2,135.00\n") is None
    assert advance_amount("Broker's fee, $ 4,575.00") is None


def test_prose_does_not_drag_a_stray_digit_in():
    """
    "approved for funding. Review your offer and request your contract in
    minutes." once matched, then swallowed the sentence and returned the first
    digit it met - 14, in place of a $35,000 approval.
    """
    body = ("You're approved for funding. Review your offer and request your "
            "contract in 5 minutes.\nEverest Business Funding\nAmount: $35,000")
    assert advance_amount(body) == 35000


def test_single_digits_and_small_change_are_never_an_advance():
    assert advance_amount("$5 processing charge applies") is None
    assert advance_amount("approved for 12 months") is None


def test_nothing_to_read():
    assert advance_amount(None) is None
    assert advance_amount("") is None
    assert advance_amount("Thanks, we'll take a look.") is None
