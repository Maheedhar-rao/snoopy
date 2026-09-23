# Copyright (c) 2026 CrocCRM. All rights reserved.
# Proprietary and confidential. See LICENSE for terms.

"""The approval amount, and the one case where it is provably a fee."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from offer_amount import advance_amount  # noqa: E402


# Loan23's real layout. Two "$" in the whole mail, both closing a fee label,
# and the figures themselves carry no symbol at all.
LOAN23 = """Kickin Wingz & Thingz
PURCHASE PRICE
30,500.00
Purchased amount of future receipts (RTR)
45,445.00
Origination fee, %
7.00%
Origination fee, $
2,135.00
NET FUNDING AMOUNT
27,965.00
Broker's fee, $
4,575.00
"""

# SmartStep runs a column header that ends in the word "Fee" and puts the
# purchase price on the next line. That figure is not a fee and must survive.
SMARTSTEP = """PRE-QUALIFIED
Purchase Price Factor Rate (Yield) Specified Amount Commission CC or TDA Account Setup Fee
$73,000.00
1.38
$100,739.00
"""


def test_loan23_returns_the_purchase_price_not_a_fee():
    assert advance_amount(LOAN23) == 30500
    assert advance_amount(LOAN23) not in (2135, 4575, 45445, 27965)


def test_a_fee_word_ending_a_column_header_is_not_a_fee_row():
    """The symbol must be on the fee label's own line. Treating a header that
    merely ends in "Fee" as one skipped $73,000 and returned $100,739, the
    total repayment."""
    assert advance_amount(SMARTSTEP) == 73000


def test_the_first_ordinary_figure_still_wins():
    assert advance_amount("You are approved for $50,000 today") == 50000
    assert advance_amount("Approved: $12,750.00 at 1.49") == 12750


def test_a_fee_is_skipped_but_a_later_real_figure_is_not():
    body = "Origination fee, $\n1,305.00\nApproved amount $14,500.00\n"
    assert advance_amount(body) == 14500


def test_nothing_readable():
    assert advance_amount(None) is None
    assert advance_amount("") is None
    assert advance_amount("Thanks, we'll take a look.") is None
    # No symbol anywhere is unchanged behaviour: this returned nothing before.
    assert advance_amount("Advance Amount: 20,000.00") is None
