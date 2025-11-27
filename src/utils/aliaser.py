import hashlib
import hmac
import string
from config.settings import ALIAS_SALT, ALIAS_LENGTH
from src.utils.db import get_alias, upsert_alias




ALPHABET = string.ascii_letters + string.digits




def _num_to_base(n: int, alphabet: str = ALPHABET) -> str:
    base = len(alphabet)
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(alphabet[r])
    return ''.join(reversed(out)) or alphabet[0]




def make_alias(project_name: str, customer_key: str) -> str:
    existing = get_alias(project_name, customer_key)
    if existing:
        return existing


    digest = hmac.new(ALIAS_SALT.encode(), customer_key.encode(), hashlib.sha256).hexdigest()
    # convert part of digest to int and base62 it
    num = int(digest[:16], 16)
    alias = _num_to_base(num)[:ALIAS_LENGTH]


    # ensure uniqueness by upserting (DB primary key ensures no dup for same customer key)
    upsert_alias(project_name, customer_key, alias)
    return alias