from decimal import ROUND_HALF_UP, Context, Decimal, localcontext

from common.contracts import Receipt

MONEY_QUANTUM = Decimal("0.01")
MONEY_ROUNDING = ROUND_HALF_UP
ITEM_AMOUNT_MISMATCH = "item_amount_mismatch"
ITEMS_TOTAL_MISMATCH = "items_total_mismatch"


def _validate_arithmetic(receipt: Receipt, *, extraction_complete: bool) -> None:
    """Append arithmetic warnings without changing extracted values.

    Args:
        receipt: Parsed fields and items, retaining any existing extraction warnings.
        extraction_complete: Whether row coverage and receipt structure justify summation.
    """
    values = [
        value
        for item in receipt.items
        for value in (item.quantity, item.price, item.sum)
        if value is not None
    ]
    if receipt.total is not None:
        values.append(receipt.total)
    # Reserve precision for exact products and sums before rounding to money.
    precision = max(
        28, sum(len(value.as_tuple().digits) + abs(value.adjusted()) + 2 for value in values) + 2
    )
    with localcontext(Context(prec=precision, rounding=MONEY_ROUNDING)):
        warnings: list[str] = []
        if any(
            _money(item.quantity * item.price) != _money(item.sum)
            for item in receipt.items
            if item.quantity is not None and item.price is not None
        ):
            warnings.append(ITEM_AMOUNT_MISMATCH)
        if extraction_complete and receipt.total is not None and receipt.items:
            items_total = sum((item.sum for item in receipt.items), Decimal("0"))
            if _money(items_total) != _money(receipt.total):
                warnings.append(ITEMS_TOTAL_MISMATCH)
    receipt.warnings.extend(warning for warning in warnings if warning not in receipt.warnings)


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=MONEY_ROUNDING)
