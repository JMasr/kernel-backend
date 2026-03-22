from arq.connections import RedisSettings

from kernel_backend.config import Settings


def make_redis_settings(settings: Settings) -> RedisSettings:
    return RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        ssl=settings.REDIS_SSL,
        conn_timeout=10,
    )
