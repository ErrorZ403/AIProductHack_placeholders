from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    SYSTEM = "model"
    ASSISTANT = "model"


class Message(BaseModel):
    content: str
    role: Role
    removable: bool = True


class BaseChat(BaseModel):
    user_id: int
    chat_id: int
    messages: list[Message] = Field(default_factory=list)
    max_tokens: int = 3000


class DialogStorage(ABC):
    @abstractmethod
    async def add_chat(self, chat: BaseChat):
        raise NotImplementedError

    @abstractmethod
    async def get_chat(self, user_id: int, chat_id: int) -> BaseChat:
        raise NotImplementedError
