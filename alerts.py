from alerting import AlertPayload, TelegramNotifier


class TelegramAlerter(TelegramNotifier):
    def __init__(self, session, bot_token: str, chat_id: str, enabled: bool) -> None:
        super().__init__(session=session, bot_token=bot_token if enabled else None, chat_id=chat_id if enabled else None)
