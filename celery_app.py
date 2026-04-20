from celery import Celery
from config import REDIS_URL, CELERY_MAX_WORKERS, CELERY_TASK_TIMEOUT

celery_app = Celery(
    "bullet",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],   # explicitly import tasks module so worker registers them
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Results
    result_expires=3600,          # keep results in Redis for 1 hour
    result_extended=True,         # store task meta (start time, worker, etc.)

    # Concurrency — how many analyses run in parallel
    worker_concurrency=CELERY_MAX_WORKERS,

    # Windows fix — prefork pool doesn't work on Windows due to multiprocessing limits
    # Use gevent (pip install gevent) for production, solo for single-process dev
    worker_pool="gevent",

    # Timeouts
    task_soft_time_limit=CELERY_TASK_TIMEOUT,         # raises exception at limit
    task_time_limit=CELERY_TASK_TIMEOUT + 60,         # hard kill 60s after soft

    # Reliability
    task_acks_late=True,          # only ack after task completes, not on receive
    task_reject_on_worker_lost=True,  # requeue if worker dies mid-task

    # Prevent one slow task from blocking the queue
    task_routes={
        "tasks.analyse_script": {"queue": "analysis"},
    },
)
