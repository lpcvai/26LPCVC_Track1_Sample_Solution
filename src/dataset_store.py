from datetime import datetime, timezone, timedelta
from typing import Any

import qai_hub

from dataset_registry import DatasetInfo
from utils import DATASETS


def try_resolve_cached_dataset_id(*, key: str, cache: bool = True, cache_write: bool = True) -> str | None:
    """
    Resolve a dataset id from datasets.json by cache key.

    Returns dataset_id if present and valid, else None.
    """
    if not cache or not key:
        return None

    DATASETS.load()
    existing = DATASETS.find_by_key(key)
    if existing is None:
        return None

    now = datetime.now(timezone.utc)
    if existing.expiration_time is not None and existing.expiration_time > (now + timedelta(minutes=5)):
        return existing.dataset_id

    try:
        remote = qai_hub.get_dataset(existing.dataset_id)
        if remote.is_expired():
            return None
        exp = getattr(remote, "expiration_time", None)
        if exp is not None:
            if getattr(exp, "tzinfo", None) is None:
                exp = exp.replace(tzinfo=timezone.utc)
            else:
                exp = exp.astimezone(timezone.utc)
        refreshed = DatasetInfo(
            dataset_id=remote.dataset_id,
            name=getattr(remote, "name", None),
            expiration_time=exp,
            key=existing.key,
            kind=existing.kind,
            meta=existing.meta,
        )
        if cache_write:
            DATASETS.upsert(refreshed)
        return existing.dataset_id
    except Exception:
        return None


def get_or_upload_dataset(
    data: Any,
    *,
    name: str | None = None,
    key: str | None = None,
    kind: str | None = None,
    meta: dict[str, Any] | None = None,
    cache: bool = True,
    cache_write: bool = True,
    cacheable: bool = True,
) -> str:
    """
    Upload a dataset to QAI Hub and record it in datasets.json.

    If key is provided and cache is True, reuse an existing non-expired dataset with the same key
    (and refresh its expiration_time from Hub) instead of re-uploading.

    If cacheable is False, the dataset will never be looked up or written to the cache, regardless
    of cache/cache_write. This is intended for run-specific uploads (for example: model outputs).
    """
    # Loading prunes expired/invalid entries as a side effect (requested behavior).
    DATASETS.load()

    if cache and cacheable and key:
        cached_id = try_resolve_cached_dataset_id(key=key, cache=cache, cache_write=(cache_write and cacheable))
        if cached_id is not None:
            return cached_id

    ds = qai_hub.upload_dataset(data, name=name) if name is not None else qai_hub.upload_dataset(data)
    exp = getattr(ds, "expiration_time", None)
    if exp is not None:
        if getattr(exp, "tzinfo", None) is None:
            exp = exp.replace(tzinfo=timezone.utc)
        else:
            exp = exp.astimezone(timezone.utc)
    info = DatasetInfo(
        dataset_id=ds.dataset_id,
        name=getattr(ds, "name", name),
        expiration_time=exp,
        key=key,
        kind=kind,
        meta=meta,
    )
    if cache_write and cacheable:
        DATASETS.upsert(info)
    return ds.dataset_id
