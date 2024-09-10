import logging
from langchain_community.llms import VLLM
from config import configs
from ml.utils import escape_markdown

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)

# Initialize the VLLM with Qwen2 model configuration
llm = VLLM(
    model="Qwen-2",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)

def get_tokens_count(messages: list) -> int:
    try:
        token_count = llm.get_num_tokens(messages)
    except Exception as e:
        logger.exception(f"Error while counting tokens: {e}")
        raise
    return token_count


async def complete(messages: list) -> str:
    try:
        # Generate content using the Qwen2 model
        completion = await llm.agenerate(messages)
    except Exception as e:
        logger.exception(f"Error while generating completion: {e}")
        raise
    return escape_markdown(completion['text'].strip()) or ''
