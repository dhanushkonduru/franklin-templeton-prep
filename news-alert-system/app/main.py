from __future__ import annotations

import logging

from app.api import create_app
from app.config import settings
from app.lifecycle import application_lifespan
from app.logging_config import setup_logging


setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = create_app(lifespan=application_lifespan)
