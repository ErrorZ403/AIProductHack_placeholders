import logging

from config import configs
from database.utils import get_optim_k
from dialog_processor.base import BaseAiChat
from dialog_processor.base import Message
from dialog_processor.base import Role
from ml.llm_api import complete
from ml.llm_api import get_tokens_count


chat_model = configs.chat_model
logger = logging.getLogger(__name__)


class AiChat(BaseAiChat):
    max_context_window: int = chat_model.chatbot.max_context_len
    max_free_context_window: int = chat_model.chatbot.max_free_context_len

    def add_message(self, text: str, role: Role = Role.USER, removable: bool = True) -> None:
        """Add a message to the chat."""
        self.messages.append(Message(content=text, role=role, removable=removable))
        if len(self.messages) >= self._minimum_messages_for_trimming():
            self._trim_context()

    def model_post_init(self, __context=None) -> None:
        """Initialize the model with system messages."""
        self.add_message(
            chat_model.chatbot.description + 'Отвечай в формате MARKDOWN.',
            Role.SYSTEM,
            removable=False,
        )
        ans = (
            '"""Ответьте на вопрос, опираясь только на следующий контекст. Если вы не можете ответить на вопрос, опираясь на контекст, пожалуйста, ответьте «Я не знаю»'
            ': \n'
        )
        self.add_message(ans, Role.USER, removable=False)

    @staticmethod
    def _get_message_tokens_num(message: Message) -> int:
        """Calculate the number of tokens in a message."""
        message_as_str = f'{message.content}{message.role}'
        return get_tokens_count(message_as_str) + 3

    def _trim_context(self) -> None:
        """Trim the context if it exceeds the maximum token count."""
        if not self.messages:
            return

        system_message_1 = self.messages.pop(0)
        system_message_2 = self.messages.pop(0)
        tokens = self._get_message_tokens_num(system_message_1) + self._get_message_tokens_num(system_message_2)

        remaining_messages_to_keep = 2  # Minimum number of messages to retain
        for i, message in enumerate(reversed(self.messages)):
            tokens += self._get_message_tokens_num(message)

            if tokens >= self.max_free_context_window:
                remaining_messages_to_keep = 2
                break

            if tokens >= self.max_context_window:
                wall = len(self.messages) - i
                self.messages = self.messages[wall:]
                break

        if tokens >= self.max_free_context_window:
            self.messages = self.messages[-remaining_messages_to_keep:]

        self.messages.insert(0, system_message_2)
        self.messages.insert(0, system_message_1)

    async def _generate_bot_answer(self) -> str:
        """Generate the bot's response using the chat model."""
        prompt = [{'role': m.role, 'parts': [m.content]} for m in self.messages]
        answer = await complete(prompt)
        self.add_message(answer, Role.ASSISTANT)
        return answer

    async def get_answer(self, text: str, db_manager: object) -> str:
        """Generate a bot answer based on user input and context."""
        final_res = await get_optim_k(text, db_manager)
        prompt = '/n'.join(final_res)

        self.add_message(prompt, Role.SYSTEM)

        self.add_message(text, Role.USER)
        answer = await self._generate_bot_answer()
        logger.debug('AiChat state: %s', self)
        return answer

    def __len__(self) -> int:
        """Calculate the total number of tokens in the chat."""
        if not self.messages:
            return 0

        return sum(self._get_message_tokens_num(message) for message in self.messages)

    @staticmethod
    def _minimum_messages_for_trimming() -> int:
        """Return the minimum number of messages required to trigger context trimming."""
        return 2
