from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_dotenv(path: str | Path = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()

        key, separator, value = line.partition("=")
        if not separator:
            continue

        cleaned_key = key.strip()
        cleaned_value = value.strip().strip('"').strip("'")
        if cleaned_key and cleaned_key not in os.environ:
            os.environ[cleaned_key] = cleaned_value


def _env(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if value is None:
        return default

    if isinstance(default, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


@dataclass(slots=True)
class Settings:
    app_name: str = "real-time-news-alert-system"
    environment: str = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    redis_url: str = "redis://redis:6379/0"
    redis_stream: str = "news_stream"
    redis_dlq_stream: str = "news_stream:dlq"
    redis_consumer_group: str = "alert-workers"
    redis_consumer_name: str = "worker-1"
    redis_recent_window_seconds: int = 3600
    redis_message_idempotency_ttl: int = 86400
    redis_consumer_batch_size: int = 50
    redis_consumer_max_retries: int = 5
    redis_consumer_retry_backoff_seconds: float = 0.5

    postgres_dsn: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/news_alerts"

    newsapi_key: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_news_url: str = "https://data.alpaca.markets/v1beta1/news"
    sec_rss_url: str = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom"
    newsapi_url: str = "https://newsapi.org/v2/top-headlines"
    news_poll_interval_seconds: float = 30.0
    http_timeout_seconds: float = 15.0
    max_fetch_retries: int = 3

    dedup_similarity_threshold: float = 0.92
    dedup_neighborhood_size: int = 200
    dedup_ttl_seconds: int = 86400
    embedding_cache_ttl_seconds: int = 86400

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier_min_confidence: float = 0.30
    classifier_keyword_threshold: float = 0.55
    classifier_embedding_threshold: float = 0.42
    classifier_other_max_confidence: float = 0.45

    portfolio_match_cache_ttl_seconds: int = 3600

    webhook_timeout_seconds: float = 10.0
    delivery_retry_attempts: int = 3
    delivery_retry_base_delay_seconds: float = 0.5
    delivery_retry_max_delay_seconds: float = 8.0
    delivery_failure_stream: str = "notification_delivery_failures"
    delivery_webhook_rate_limit_per_second: float = 20.0
    delivery_slack_rate_limit_per_second: float = 1.0
    delivery_email_rate_limit_per_second: float = 10.0
    delivery_replay_group: str = "delivery-replay-workers"
    delivery_replay_consumer_name: str = "delivery-replay-1"
    delivery_replay_batch_size: int = 25
    delivery_replay_max_attempts: int = 5
    smtp_host: str = "localhost"
    smtp_port: int = 25
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_address: str = "alerts@example.com"
    smtp_use_tls: bool = False

    pipeline_concurrency: int = 10


DEFAULT_SETTINGS = Settings()


def load_settings() -> Settings:
    _load_dotenv()
    return Settings(
        app_name=_env("APP_NAME", DEFAULT_SETTINGS.app_name),
        environment=_env("ENVIRONMENT", DEFAULT_SETTINGS.environment),
        log_level=_env("LOG_LEVEL", DEFAULT_SETTINGS.log_level),
        api_host=_env("API_HOST", DEFAULT_SETTINGS.api_host),
        api_port=_env("API_PORT", DEFAULT_SETTINGS.api_port),
        redis_url=_env("REDIS_URL", DEFAULT_SETTINGS.redis_url),
        redis_stream=_env("REDIS_STREAM", DEFAULT_SETTINGS.redis_stream),
        redis_dlq_stream=_env("REDIS_DLQ_STREAM", DEFAULT_SETTINGS.redis_dlq_stream),
        redis_consumer_group=_env("REDIS_CONSUMER_GROUP", DEFAULT_SETTINGS.redis_consumer_group),
        redis_consumer_name=_env("REDIS_CONSUMER_NAME", DEFAULT_SETTINGS.redis_consumer_name),
        redis_recent_window_seconds=_env("REDIS_RECENT_WINDOW_SECONDS", DEFAULT_SETTINGS.redis_recent_window_seconds),
        redis_message_idempotency_ttl=_env("REDIS_MESSAGE_IDEMPOTENCY_TTL", DEFAULT_SETTINGS.redis_message_idempotency_ttl),
        redis_consumer_batch_size=_env("REDIS_CONSUMER_BATCH_SIZE", DEFAULT_SETTINGS.redis_consumer_batch_size),
        redis_consumer_max_retries=_env("REDIS_CONSUMER_MAX_RETRIES", DEFAULT_SETTINGS.redis_consumer_max_retries),
        redis_consumer_retry_backoff_seconds=_env(
            "REDIS_CONSUMER_RETRY_BACKOFF_SECONDS", DEFAULT_SETTINGS.redis_consumer_retry_backoff_seconds
        ),
        postgres_dsn=_env("POSTGRES_DSN", DEFAULT_SETTINGS.postgres_dsn),
        newsapi_key=_env("NEWSAPI_KEY", DEFAULT_SETTINGS.newsapi_key),
        alpaca_api_key=_env("ALPACA_API_KEY", DEFAULT_SETTINGS.alpaca_api_key),
        alpaca_api_secret=_env("ALPACA_API_SECRET", DEFAULT_SETTINGS.alpaca_api_secret),
        alpaca_news_url=_env("ALPACA_NEWS_URL", DEFAULT_SETTINGS.alpaca_news_url),
        sec_rss_url=_env("SEC_RSS_URL", DEFAULT_SETTINGS.sec_rss_url),
        newsapi_url=_env("NEWSAPI_URL", DEFAULT_SETTINGS.newsapi_url),
        news_poll_interval_seconds=_env("NEWS_POLL_INTERVAL_SECONDS", DEFAULT_SETTINGS.news_poll_interval_seconds),
        http_timeout_seconds=_env("HTTP_TIMEOUT_SECONDS", DEFAULT_SETTINGS.http_timeout_seconds),
        max_fetch_retries=_env("MAX_FETCH_RETRIES", DEFAULT_SETTINGS.max_fetch_retries),
        dedup_similarity_threshold=_env("DEDUP_SIMILARITY_THRESHOLD", DEFAULT_SETTINGS.dedup_similarity_threshold),
        dedup_neighborhood_size=_env("DEDUP_NEIGHBORHOOD_SIZE", DEFAULT_SETTINGS.dedup_neighborhood_size),
        dedup_ttl_seconds=_env("DEDUP_TTL_SECONDS", DEFAULT_SETTINGS.dedup_ttl_seconds),
        embedding_cache_ttl_seconds=_env("EMBEDDING_CACHE_TTL_SECONDS", DEFAULT_SETTINGS.embedding_cache_ttl_seconds),
        embedding_model_name=_env("EMBEDDING_MODEL_NAME", DEFAULT_SETTINGS.embedding_model_name),
        classifier_min_confidence=_env("CLASSIFIER_MIN_CONFIDENCE", DEFAULT_SETTINGS.classifier_min_confidence),
        classifier_keyword_threshold=_env("CLASSIFIER_KEYWORD_THRESHOLD", DEFAULT_SETTINGS.classifier_keyword_threshold),
        classifier_embedding_threshold=_env("CLASSIFIER_EMBEDDING_THRESHOLD", DEFAULT_SETTINGS.classifier_embedding_threshold),
        classifier_other_max_confidence=_env("CLASSIFIER_OTHER_MAX_CONFIDENCE", DEFAULT_SETTINGS.classifier_other_max_confidence),
        portfolio_match_cache_ttl_seconds=_env(
            "PORTFOLIO_MATCH_CACHE_TTL_SECONDS", DEFAULT_SETTINGS.portfolio_match_cache_ttl_seconds
        ),
        webhook_timeout_seconds=_env("WEBHOOK_TIMEOUT_SECONDS", DEFAULT_SETTINGS.webhook_timeout_seconds),
        delivery_retry_attempts=_env("DELIVERY_RETRY_ATTEMPTS", DEFAULT_SETTINGS.delivery_retry_attempts),
        delivery_retry_base_delay_seconds=_env(
            "DELIVERY_RETRY_BASE_DELAY_SECONDS", DEFAULT_SETTINGS.delivery_retry_base_delay_seconds
        ),
        delivery_retry_max_delay_seconds=_env("DELIVERY_RETRY_MAX_DELAY_SECONDS", DEFAULT_SETTINGS.delivery_retry_max_delay_seconds),
        delivery_failure_stream=_env("DELIVERY_FAILURE_STREAM", DEFAULT_SETTINGS.delivery_failure_stream),
        delivery_webhook_rate_limit_per_second=_env(
            "DELIVERY_WEBHOOK_RATE_LIMIT_PER_SECOND", DEFAULT_SETTINGS.delivery_webhook_rate_limit_per_second
        ),
        delivery_slack_rate_limit_per_second=_env(
            "DELIVERY_SLACK_RATE_LIMIT_PER_SECOND", DEFAULT_SETTINGS.delivery_slack_rate_limit_per_second
        ),
        delivery_email_rate_limit_per_second=_env(
            "DELIVERY_EMAIL_RATE_LIMIT_PER_SECOND", DEFAULT_SETTINGS.delivery_email_rate_limit_per_second
        ),
        delivery_replay_group=_env("DELIVERY_REPLAY_GROUP", DEFAULT_SETTINGS.delivery_replay_group),
        delivery_replay_consumer_name=_env("DELIVERY_REPLAY_CONSUMER_NAME", DEFAULT_SETTINGS.delivery_replay_consumer_name),
        delivery_replay_batch_size=_env("DELIVERY_REPLAY_BATCH_SIZE", DEFAULT_SETTINGS.delivery_replay_batch_size),
        delivery_replay_max_attempts=_env("DELIVERY_REPLAY_MAX_ATTEMPTS", DEFAULT_SETTINGS.delivery_replay_max_attempts),
        smtp_host=_env("SMTP_HOST", DEFAULT_SETTINGS.smtp_host),
        smtp_port=_env("SMTP_PORT", DEFAULT_SETTINGS.smtp_port),
        smtp_username=_env("SMTP_USERNAME", DEFAULT_SETTINGS.smtp_username),
        smtp_password=_env("SMTP_PASSWORD", DEFAULT_SETTINGS.smtp_password),
        smtp_from_address=_env("SMTP_FROM_ADDRESS", DEFAULT_SETTINGS.smtp_from_address),
        smtp_use_tls=_env("SMTP_USE_TLS", DEFAULT_SETTINGS.smtp_use_tls),
        pipeline_concurrency=_env("PIPELINE_CONCURRENCY", DEFAULT_SETTINGS.pipeline_concurrency),
    )


settings = load_settings()
