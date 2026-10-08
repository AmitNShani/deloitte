"""Shared HTTP session with retries for all public data sources."""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config

_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))),
)


def session() -> requests.Session:
    return _session


def get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", config.HTTP_TIMEOUT)
    resp = _session.get(url, **kwargs)
    resp.raise_for_status()
    return resp
