import logging
from typing import List

import boto3
from fastapi import APIRouter
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from celery_worker import delete_bd_task
from celery_worker import update_bd_task
from database.faiss import get_pgvector
from dialog_processor.base import Message
from config import configs
from dialog_processor.models import AiChat

app = FastAPI()
router = APIRouter()

logging.basicConfig(
    level=logging.INFO if configs.debug_mode == '0' else logging.DEBUG,
    format='[{asctime} {levelname}] {message}',
    style='{'
)

s3_client = boto3.client('s3')


class S3ObjectParams(BaseModel):
    s3_bucket: str
    s3_key: str
    uid_project: str


async def async_get_chat_completion(messages: List[dict]) -> str:
    try:
        messages = [Message(**msg) for msg in messages]
        client = await get_pgvector()

        chat = AiChat()
        chat.model_post_init()
        chat.messages.extend(messages[:-1])
        return await chat.get_answer(messages[-1].content, client)
    except Exception as e:
        logging.exception(
            f'Error in get_ans_task. '
            f'Exception: {e}, '
            f'Messages: {messages}, '
        )
        raise


@router.post('/chat_completion', response_model=str)
async def get_chat_answer(messages: List[Message]) -> str:
    try:
        logging.debug(f'Received messages: {messages}')
        messages_dict = [message.dict() for message in messages]
        logging.debug(f'Messages dict: {messages_dict}')
        return await async_get_chat_completion(messages_dict)
    except Exception as e:
        logging.exception(f'Error in get_ans: {e}, messages: {messages}')
        raise HTTPException(status_code=500, detail=f'An error occurred: {e}, messages: {messages}') from e


@router.post('/update')
async def update_bd(body:S3ObjectParams) -> dict:
    s3_bucket = body.s3_bucket
    s3_key = body.s3_key
    logging.info(f"update_bd called with parameters: s3_bucket={s3_bucket}, s3_key={s3_key}")
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        update_bd_task.delay(s3_bucket, s3_key)
        return {'status': 'task executed'}
    except s3_client.exceptions.NoSuchKey:
        logging.exception(f'S3 Key {s3_key} not found in bucket {s3_bucket}')
        raise HTTPException(status_code=404, detail=f'S3 Key {s3_key} not found')
    except s3_client.exceptions.NoSuchBucket:
        logging.exception(f'S3 Bucket {s3_bucket} not found')
        raise HTTPException(status_code=404, detail=f'S3 Bucket {s3_bucket} not found')
    except AssertionError as e:
        logging.exception(f'AssertionError in async_update_bd_task: {e}')
        return {'error': f'AssertionError: {e}'}
    except Exception as e:
        logging.exception(f'Error in async_update_bd_task: {e}')
        raise HTTPException(status_code=500, detail=f'Internal Server Error: {e}') from e


@router.get('/delete')
async def delete_bd() -> dict:
    try:
        delete_bd_task.delay()
        return {'status': 'task executed'}
    except Exception as e:
        logging.exception(f'Error in delete: {e}')
        raise HTTPException(status_code=500, detail=str(e)) from e


app.include_router(router)
