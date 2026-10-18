from dataclasses import dataclass

from common.contracts import Receipt
from pipeline import MonthlySummary


@dataclass(frozen=True)
class Texts:
    """Localized user-facing bot messages and command descriptions.

    Attributes:
        start: Greeting sent for the start command.
        help: Usage guidance sent for the help command.
        unknown: Reply sent for unsupported messages.
        error: Reply sent when receipt processing fails.
        unreadable: Reply sent when nothing useful was found on a photo.
        month_empty: Reply sent when no receipts were saved this month.
        month_spending: Label for the monthly spending.
        receipt_count: Label for the number of receipts.
        store: Label for the store name.
        date: Label for the purchase date.
        total: Label for the receipt total.
        not_found: Placeholder for a field that was not recognized.
        help_command: Localized help command description.
        month_command: Localized monthly report command description.
    """

    start: str
    help: str
    unknown: str
    error: str
    unreadable: str
    month_empty: str
    month_spending: str
    receipt_count: str
    store: str
    date: str
    total: str
    not_found: str
    help_command: str
    month_command: str


RU = Texts(
    start="Привет! Пришлите фото чека, и я посчитаю покупки по категориям.\nСписок команд: /help",
    help=(
        "Отправьте фото чека, чтобы сохранить покупку. "
        "Если отправить его файлом, качество будет выше.\n"
        "/month — расходы за текущий месяц\n"
        "/help — эта справка"
    ),
    unknown="Я понимаю только фото чеков. Список команд: /help",
    error="Не получилось обработать чек. Попробуйте ещё раз позже.",
    unreadable="Не удалось распознать чек. Сфотографируйте его ровно и при хорошем освещении.",
    month_empty="В этом месяце чеков пока нет.",
    month_spending="Расходы за",
    receipt_count="Чеков",
    store="Магазин",
    date="Дата",
    total="Сумма",
    not_found="не распознано",
    help_command="Справка",
    month_command="Расходы за месяц",
)

EN = Texts(
    start="Hi! Send me a photo of a receipt and I will sort the purchases by category.\n"
    "Commands: /help",
    help=(
        "Send a photo of a receipt to save the purchase. "
        "Sending it as a file keeps the quality higher.\n"
        "/month — spending for the current month\n"
        "/help — this message"
    ),
    unknown="I only understand receipt photos. Commands: /help",
    error="Could not process the receipt. Please try again later.",
    unreadable="Could not read the receipt. Take a straight photo in good light.",
    month_empty="No receipts this month yet.",
    month_spending="Spending for",
    receipt_count="Receipts",
    store="Store",
    date="Date",
    total="Total",
    not_found="not recognized",
    help_command="Help",
    month_command="Spending this month",
)

TEXTS_BY_LANGUAGE = {"ru": RU, "en": EN}
DEFAULT_TEXTS = EN


def texts_for(language_code: str | None) -> Texts:
    """Select localized texts for a Telegram language code.

    Args:
        language_code: Telegram language code, optionally including a region.

    Returns:
        Matching localized texts, or the default English texts.
    """
    if not language_code:
        return DEFAULT_TEXTS
    language = language_code.split("-")[0].lower()
    return TEXTS_BY_LANGUAGE.get(language, DEFAULT_TEXTS)


def format_receipt(receipt: Receipt, texts: Texts) -> str:
    """Render a parsed receipt as a user-facing message.

    Args:
        receipt: Parsed receipt.
        texts: Localized labels.

    Returns:
        Message with the store, date and total of the receipt.
    """
    store = receipt.store or texts.not_found
    date = receipt.datetime.strftime("%d.%m.%Y %H:%M") if receipt.datetime else texts.not_found
    total = f"{receipt.total:.2f}" if receipt.total is not None else texts.not_found
    return f"{texts.store}: {store}\n{texts.date}: {date}\n{texts.total}: {total}"


def format_monthly_summary(summary: MonthlySummary, texts: Texts) -> str:
    """Render monthly spending as a user-facing message.

    Args:
        summary: Spending summary for one month.
        texts: Localized labels.

    Returns:
        Message with the month total and the number of receipts.
    """
    if summary.receipt_count == 0:
        return texts.month_empty
    return (
        f"{texts.month_spending} {summary.month:%m.%Y}: {summary.total:.2f}\n"
        f"{texts.receipt_count}: {summary.receipt_count}"
    )
