import logging
from .base import BaseChat, DialogStorage, Role, Message

logger = logging.getLogger(__name__)


class Chat(BaseChat):
    def add_message(self, text: str, role: Role = Role.USER, removable: bool = True):
        self.messages.append(Message(content=text, role=role, removable=removable))
        if len(self.messages) >= 2:
            self._trim_context()

    def _trim_context(self):
        if not self.messages:
            return

        return self.messages[-10:]

    def get_massages(self):
        return self.messages


class DictDialogStorage(DialogStorage):
    chats: dict[tuple[int, int], Chat] = dict()

    async def add_chat(self, chat: Chat):
        self.chats[(chat.user_id, chat.chat_id)] = chat

    async def get_chat(self, user_id: int, chat_id: int) -> Chat:
        try:
            return self.chats[(user_id, chat_id)]
        except KeyError:
            raise Exception(
                f"Chat for {(user_id, chat_id)} doesn't exist.",
            )

    def is_chat_exists(self, user_id: int, chat_id: int) -> bool:
        return (user_id, chat_id) in self.chats


class DialogManager:
    def __init__(self, dialog_storage: DialogStorage):
        self.dialog_storage: DialogStorage = dialog_storage

    async def add_chat(self, chat: Chat):
        await self.dialog_storage.add_chat(chat)

    async def get_chat(self, user_id: int, chat_id: int) -> Chat:
        return await self.dialog_storage.get_chat(user_id, chat_id)



## основной
class TelegramDialogManager(DialogManager):
    dialog_storage: DictDialogStorage
    
    async def get_or_create_chat(self, user_id: int, chat_id: int) -> Chat:
        if self.dialog_storage.is_chat_exists(user_id, chat_id):
            chat = await super().get_chat(user_id, chat_id)
        else:
            chat = Chat(user_id=user_id, chat_id=chat_id)
            await self.add_chat(chat)
        return chat
    
    async def add_message(self, user_id: int, chat_id: int, text, role: Role = Role.USER):
        chat = await self.get_or_create_chat(user_id=user_id, chat_id=chat_id)
        chat.add_message(text, role)

    async def get_chat(self, user_id: int, chat_id: int):
        chat = await self.get_or_create_chat(user_id=user_id, chat_id=chat_id)
        return chat.get_massages()
