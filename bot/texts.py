from dataclasses import dataclass


@dataclass(frozen=True)
class Texts:
    start: str
    help: str
    unknown: str
    error: str
    help_command: str
    month_command: str


RU = Texts(
    start="Привет! Пришлите фото чека, и я посчитаю покупки по категориям.\nСписок команд: /help",
    help=(
        "Отправьте фото чека, чтобы сохранить покупку.\n"
        "/month — расходы за текущий месяц\n"
        "/help — эта справка"
    ),
    unknown="Я понимаю только фото чеков. Список команд: /help",
    error="Не получилось обработать чек. Попробуйте ещё раз позже.",
    help_command="Справка",
    month_command="Расходы за месяц",
)

EN = Texts(
    start="Hi! Send me a photo of a receipt and I will sort the purchases by category.\n"
    "Commands: /help",
    help=(
        "Send a photo of a receipt to save the purchase.\n"
        "/month — spending for the current month\n"
        "/help — this message"
    ),
    unknown="I only understand receipt photos. Commands: /help",
    error="Could not process the receipt. Please try again later.",
    help_command="Help",
    month_command="Spending this month",
)

TEXTS_BY_LANGUAGE = {"ru": RU, "en": EN}
DEFAULT_TEXTS = EN


def texts_for(language_code: str | None) -> Texts:
    if not language_code:
        return DEFAULT_TEXTS
    language = language_code.split("-")[0].lower()
    return TEXTS_BY_LANGUAGE.get(language, DEFAULT_TEXTS)
