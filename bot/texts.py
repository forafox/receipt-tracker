from dataclasses import dataclass

from common.contracts import Receipt
from pipeline import MonthlySummary

# Warning added by the extraction module when item sums do not add up to the total.
ITEMS_TOTAL_MISMATCH = "items_total_mismatch"


@dataclass(frozen=True)
class Texts:
    """Localized user-facing bot messages and command descriptions.

    Attributes:
        start: Greeting sent for the start command.
        help: Usage guidance sent for the help command.
        unknown: Reply sent for unsupported messages.
        error: Reply sent when receipt processing fails.
        unreadable: Reply sent when nothing useful was found on a photo.
        duplicate: Question asked when the receipt looks already saved.
        duplicate_save: Button that saves a suspected duplicate anyway.
        duplicate_skip: Button that drops a suspected duplicate.
        duplicate_saved: Heading of the reply after a duplicate was saved anyway.
        duplicate_skipped: Reply after a suspected duplicate was dropped.
        duplicate_expired: Reply when the suspected duplicate is no longer available.
        file_too_large: Reply sent when an image file is too large to download.
        recognition_unavailable: Reply sent when text recognition is temporarily failing.
        month_empty: Reply sent when no receipts were saved this month.
        month_spending: Label for the monthly spending.
        history_empty: Reply sent when the user has no saved receipts.
        history_title: Heading of the recent receipts list.
        undo_done: Heading of the reply about a deleted receipt.
        undo_empty: Reply sent when there is nothing to delete.
        receipt_count: Label for the number of receipts.
        store: Label for the store name.
        date: Label for the purchase date.
        total: Label for the receipt total.
        items: Heading of the purchased items list.
        items_mismatch: Note shown when item sums do not add up to the total.
        not_found: Placeholder for a field that was not recognized.
        help_command: Localized help command description.
        month_command: Localized monthly report command description.
        history_command: Localized recent receipts command description.
        undo_command: Localized undo command description.
    """

    start: str
    help: str
    unknown: str
    error: str
    unreadable: str
    duplicate: str
    duplicate_save: str
    duplicate_skip: str
    duplicate_saved: str
    duplicate_skipped: str
    duplicate_expired: str
    file_too_large: str
    recognition_unavailable: str
    month_empty: str
    month_spending: str
    history_empty: str
    history_title: str
    undo_done: str
    undo_empty: str
    receipt_count: str
    store: str
    date: str
    total: str
    items: str
    items_mismatch: str
    not_found: str
    help_command: str
    month_command: str
    history_command: str
    undo_command: str


RU = Texts(
    start="Привет! Пришлите фото чека, и я посчитаю покупки по категориям.\nСписок команд: /help",
    help=(
        "Отправьте фото чека, чтобы сохранить покупку. "
        "Если отправить его файлом, качество будет выше.\n"
        "/month — расходы за текущий месяц\n"
        "/history — последние чеки\n"
        "/undo — удалить последний сохранённый чек\n"
        "/help — эта справка"
    ),
    unknown="Я понимаю только фото чеков. Список команд: /help",
    error="Не получилось обработать чек. Попробуйте ещё раз позже.",
    unreadable="Не удалось распознать чек. Сфотографируйте его ровно и при хорошем освещении.",
    duplicate="Похоже, этот чек уже сохранён. Сохранить его ещё раз?",
    duplicate_save="Сохранить ещё раз",
    duplicate_skip="Не сохранять",
    duplicate_saved="Чек сохранён ещё раз:",
    duplicate_skipped="Хорошо, повторно чек не сохранён.",
    duplicate_expired="Этот чек уже не найти. Пришлите фото ещё раз.",
    file_too_large="Файл больше 20 МБ, такой бот скачать не может. Пришлите фото поменьше.",
    recognition_unavailable="Распознавание временно недоступно. Попробуйте через несколько минут.",
    month_empty="В этом месяце чеков пока нет.",
    month_spending="Расходы за",
    history_empty="Сохранённых чеков пока нет.",
    history_title="Последние чеки:",
    undo_done="Удалён чек:",
    undo_empty="Удалять нечего, сохранённых чеков нет.",
    receipt_count="Чеков",
    store="Магазин",
    date="Дата",
    total="Сумма",
    items="Покупки",
    items_mismatch=(
        "Сумма покупок не сходится с итогом. "
        "Если чек распознан неверно, удалите его командой /undo."
    ),
    not_found="не распознано",
    help_command="Справка",
    month_command="Расходы за месяц",
    history_command="Последние чеки",
    undo_command="Удалить последний чек",
)

EN = Texts(
    start="Hi! Send me a photo of a receipt and I will sort the purchases by category.\n"
    "Commands: /help",
    help=(
        "Send a photo of a receipt to save the purchase. "
        "Sending it as a file keeps the quality higher.\n"
        "/month — spending for the current month\n"
        "/history — recent receipts\n"
        "/undo — delete the last saved receipt\n"
        "/help — this message"
    ),
    unknown="I only understand receipt photos. Commands: /help",
    error="Could not process the receipt. Please try again later.",
    unreadable="Could not read the receipt. Take a straight photo in good light.",
    duplicate="This receipt seems to be saved already. Save it again?",
    duplicate_save="Save again",
    duplicate_skip="Don't save",
    duplicate_saved="Receipt saved again:",
    duplicate_skipped="OK, the receipt was not saved again.",
    duplicate_expired="This receipt is no longer available. Send the photo again.",
    file_too_large="The file is larger than 20 MB and cannot be downloaded. Send a smaller image.",
    recognition_unavailable="Recognition is temporarily unavailable. Try again in a few minutes.",
    month_empty="No receipts this month yet.",
    month_spending="Spending for",
    history_empty="No saved receipts yet.",
    history_title="Recent receipts:",
    undo_done="Deleted receipt:",
    undo_empty="Nothing to delete, there are no saved receipts.",
    receipt_count="Receipts",
    store="Store",
    date="Date",
    total="Total",
    items="Items",
    items_mismatch=(
        "The items do not add up to the total. "
        "If the receipt was read incorrectly, delete it with /undo."
    ),
    not_found="not recognized",
    help_command="Help",
    month_command="Spending this month",
    history_command="Recent receipts",
    undo_command="Delete the last receipt",
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
        Message with the store, date and total of the receipt, followed by its items if any.
    """
    store = receipt.store or texts.not_found
    date = receipt.datetime.strftime("%d.%m.%Y %H:%M") if receipt.datetime else texts.not_found
    total = f"{receipt.total:.2f}" if receipt.total is not None else texts.not_found
    lines = [f"{texts.store}: {store}", f"{texts.date}: {date}", f"{texts.total}: {total}"]
    if receipt.items:
        lines.append(f"\n{texts.items}:")
        lines.extend(f"- {item.name}: {item.sum:.2f}" for item in receipt.items)
    if ITEMS_TOTAL_MISMATCH in receipt.warnings:
        lines.append(f"\n{texts.items_mismatch}")
    return "\n".join(lines)


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


def format_history(receipts: list[Receipt], texts: Texts) -> str:
    """Render recent receipts as a user-facing list.

    Args:
        receipts: Receipts ordered from the newest to the oldest.
        texts: Localized labels.

    Returns:
        One line per receipt with its date, store and total.
    """
    if not receipts:
        return texts.history_empty
    lines = [texts.history_title]
    for receipt in receipts:
        date = receipt.datetime.strftime("%d.%m.%Y") if receipt.datetime else texts.not_found
        total = f"{receipt.total:.2f}" if receipt.total is not None else texts.not_found
        lines.append(f"{date}, {receipt.store or texts.not_found}, {total}")
    return "\n".join(lines)
