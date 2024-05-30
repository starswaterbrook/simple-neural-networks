import logging

import pytest


@pytest.fixture(autouse=True)
def setup_logger():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    yield
