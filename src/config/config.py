import logging
from dataclasses import dataclass
from .utils import get_env_variable
from dotenv import load_dotenv


MAX_TELEGRAM_MESSAGE_LEN: int = 4095
logger: logging.Logger = logging.getLogger(__name__)

@dataclass
class TelegramBot:
    token: str
    debug_mode: bool = False


@dataclass
class Config:
    tg_bot: TelegramBot
    ALLOWED_USERS: list
    base_url: str


def load_config() -> Config:
    load_dotenv()

    tg_bot: TelegramBot = TelegramBot(
        token=get_env_variable("BOT_TOKEN"),
        debug_mode=get_env_variable("DEBUG") == "1",
    )

    ALLOWED_USERS = get_env_variable("ALLOWED_USERS")
    base_url = get_env_variable("BASE_URL")

    return Config(
        tg_bot=tg_bot,
        ALLOWED_USERS=ALLOWED_USERS,
        base_url=base_url,
    )
