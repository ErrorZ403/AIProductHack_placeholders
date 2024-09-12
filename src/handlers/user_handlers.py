import logging
from aiogram import Bot, F, Router
# from aiogram.types import Message
import aiohttp
from collections import defaultdict, deque
from typing import List

from src.models import TelegramDialogManager, DictDialogStorage, Message, Role
from src.config import configs

router = Router()
dialog_manager = TelegramDialogManager(DictDialogStorage())


def is_user_allowed(username: str) -> bool:
    return username in configs.ALLOWED_USERS


def is_mentioned(message: Message, bot_username: str) -> bool:
    return f"@{bot_username}" in message.text


def is_reply_to_bot(message: Message, bot_id: int) -> bool:
    return message.reply_to_message and message.reply_to_message.from_user.id == bot_id


async def fetch_chat_answer(messages: List[Message], uid_project: str) -> str:
    async with aiohttp.ClientSession() as session:
        logging.debug(messages)
        messages_dict = [message.dict() for message in messages]
        
        for message in messages_dict:
            message['role'] = message['role'].value
        
        payload = messages_dict
        params = {"uid_project": uid_project}

        logging.debug(f"Payload: {payload}")
        logging.debug(f"Params: {params}")
        
        async with session.post(f"{configs.base_url}", json=payload, params=params) as response:
            if response.status != 200:
                logging.error(f"Failed to fetch chat answer: {response.status}")
                try:
                    error_detail = await response.json()
                    logging.error(f"Error detail: {error_detail}")
                except Exception as e:
                    logging.error(f"Failed to parse error detail: {e}")
                return "Извините, произошла ошибка при обработке вашего запроса"
            
            data = await response.json()
            return data


@router.message(lambda message: is_mentioned(message, "enji_test_chat_bot") or is_reply_to_bot(message, message.bot.id))
async def process_text_message(message: Message, bot: Bot):
    username = message.from_user.username
    user_id = message.from_user.id
    if not is_user_allowed(username):
        await message.answer("У вас нет доступа к этому боту")
        return
    if not message.text:
        await message.reply(text="Скажите что-то")
    else:
        await dialog_manager.add_message(message.from_user.id, message.chat.id, message.text)
        messages = await dialog_manager.get_chat(message.from_user.id, message.chat.id)
        answer = await fetch_chat_answer(messages, uid_project="45")
        answer = answer[:300]
        await dialog_manager.add_message(message.from_user.id, message.chat.id, message.text, Role.SYSTEM)
        await message.reply(text=answer)


