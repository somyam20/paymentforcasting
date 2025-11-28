import asyncio
import hashlib
import hmac
import string
from config.settings import ALIAS_SALT, ALIAS_LENGTH
from src.utils.db import get_alias, upsert_alias
import logging

logger = logging.getLogger(__name__)

ALPHABET = string.ascii_letters + string.digits

def _num_to_base(n: int, alphabet: str = ALPHABET) -> str:
    base = len(alphabet)
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(alphabet[r])
    return ''.join(reversed(out)) or alphabet[0]


async def make_alias(project_name: str, customer_key: str) -> str:
    try:
        # --- WRAP BLOCKING GET ---
        existing = await asyncio.to_thread(get_alias, project_name, customer_key)
        if existing:
            logger.debug(
                "make_alias: existing alias for project=%s customer=%s -> %s",
                project_name,
                customer_key,
                existing
            )
            return existing

        # PURE CPU → fast enough to run sync
        digest = hmac.new(ALIAS_SALT.encode(), customer_key.encode(), hashlib.sha256).hexdigest()
        num = int(digest[:16], 16)
        alias = _num_to_base(num)[:ALIAS_LENGTH]

        # --- WRAP BLOCKING UPSERT ---
        await asyncio.to_thread(upsert_alias, project_name, customer_key, alias)

        logger.info(
            "make_alias: created alias for project=%s customer=%s -> %s",
            project_name,
            customer_key,
            alias
        )
        return alias

    except Exception as e:
        logger.exception(
            "make_alias: failed for project=%s customer=%s : %s",
            project_name,
            customer_key,
            e
        )
        raise
