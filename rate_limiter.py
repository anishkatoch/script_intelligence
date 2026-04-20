import redis
import logging
from config import REDIS_URL, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW

logger = logging.getLogger(__name__)

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def check_rate_limit(session_id: str) -> tuple[bool, int, int]:
    """
    Check if session is within rate limit.

    Returns:
        (allowed, current_count, remaining)
        allowed=False means the request should be blocked.
    """
    key = f"rate_limit:{session_id}"
    try:
        r = _get_redis()
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()

        # Set expiry on first request
        if count == 1:
            r.expire(key, RATE_LIMIT_WINDOW)
            ttl = RATE_LIMIT_WINDOW

        remaining = max(0, RATE_LIMIT_MAX - count)
        allowed = count <= RATE_LIMIT_MAX

        if not allowed:
            logger.warning(f"Rate limit exceeded for session {session_id[:8]}... — {count}/{RATE_LIMIT_MAX} in window")

        return allowed, count, remaining

    except redis.RedisError as e:
        # If Redis is down, fail open — don't block users
        logger.error(f"Redis error during rate limit check: {e}")
        return True, 0, RATE_LIMIT_MAX


def get_usage(session_id: str) -> dict:
    """Return current usage stats for a session."""
    key = f"rate_limit:{session_id}"
    try:
        r = _get_redis()
        count = int(r.get(key) or 0)
        ttl = r.ttl(key)
        return {
            "count": count,
            "limit": RATE_LIMIT_MAX,
            "remaining": max(0, RATE_LIMIT_MAX - count),
            "resets_in_seconds": ttl if ttl > 0 else 0,
        }
    except redis.RedisError:
        return {"count": 0, "limit": RATE_LIMIT_MAX, "remaining": RATE_LIMIT_MAX, "resets_in_seconds": 0}
