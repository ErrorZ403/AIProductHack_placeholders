from enum import Enum

from pydantic import BaseModel
from pydantic import Field


class Role(str, Enum):
    USER = 'user'
    SYSTEM = 'model'
    ASSISTANT = 'model'


class Message(BaseModel):
    content: str
    role: Role
    removable: bool = True


class BaseAiChat(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    max_tokens: int = 3000
