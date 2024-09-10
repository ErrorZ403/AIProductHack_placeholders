import asyncio
import datetime
import logging
from typing import Any
from typing import List

from celery import Celery
from celery.schedules import crontab

from config import configs
from data_processing.bytes_processing import process_zipfile
from data_processing.get_chunks import DataProcessor
from database.faiss import get_pgvector

import boto3

app = Celery('copilot')
app.conf.broker_url = configs.celery_broker_url
app.conf.result_backend = configs.celery_result_backend
app.autodiscover_tasks()

s3_client = boto3.client('s3')

logging.basicConfig(level=logging.INFO)


@app.on_after_configure.connect
def setup_periodic_tasks(sender: Any, **kwargs) -> None:
    sender.add_periodic_task(
        crontab(minute=0, hour=0),
        delete_bd_task.s(),
        name='Remove BD everyday at midnight',
    )


async def async_update_bd_task(s3_bucket: str, s3_key: str, uid_project: str) -> str:
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        data = response['Body'].read()



        for chunk in chunks:
            await client.add_document(**chunk)

        s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)
        logging.info('DB updated')
        return 'DB updated'
    except Exception as e:
        logging.exception(
            f'Error in async_update_bd_task: {e}, s3_bucket: {s3_bucket}, s3_key: {s3_key}, uid_project: {uid_project}')
        raise


@app.task
def update_bd_task(s3_bucket: str, s3_key: str, uid_project: str) -> str:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(async_update_bd_task(s3_bucket, s3_key, uid_project))


async def async_delete_bd_task() -> str:
    two_months_ago = None
    try:
        client = await get_pgvector()
        today = datetime.date.today()
        two_months_ago = today - datetime.timedelta(days=60)

        await client.delete_document_by_date(two_months_ago)

        return f'Deleted chunks from DB by date {two_months_ago}'
    except Exception as e:
        logging.exception(f'Error in async_delete_bd_task: {e}, date: {two_months_ago}')
        raise


@app.task
def delete_bd_task() -> str:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(async_delete_bd_task())
