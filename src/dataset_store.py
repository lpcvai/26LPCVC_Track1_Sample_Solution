from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Any

import qai_hub

from dataset_registry import DatasetInfo
from utils import DATASETS


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
        existing = DATASETS.find_by_key(key)
        if existing is not None:
            # Prefer trusting local expiration_time to avoid API calls that can fail and
            # cause unnecessary re-uploads. Only revalidate against Hub when missing or
            # close to expiry.
            now = _utc_now()
            if existing.expiration_time is not None and existing.expiration_time > (now + timedelta(minutes=5)):
                return existing.dataset_id

            try:
                remote = qai_hub.get_dataset(existing.dataset_id)
                if not remote.is_expired():
                    refreshed = DatasetInfo(
                        dataset_id=remote.dataset_id,
                        name=getattr(remote, "name", None),
                        expiration_time=_to_utc(getattr(remote, "expiration_time", None)),
                        key=existing.key,
                        kind=existing.kind,
                        meta=existing.meta,
                    )
                    if cache_write and cacheable:
                        DATASETS.upsert(refreshed)
                    return existing.dataset_id
            except Exception:
                # If lookup fails, fall through to upload a new dataset.
                pass

    ds = qai_hub.upload_dataset(data, name=name) if name is not None else qai_hub.upload_dataset(data)
    info = DatasetInfo(
        dataset_id=ds.dataset_id,
        name=getattr(ds, "name", name),
        expiration_time=_to_utc(getattr(ds, "expiration_time", None)),
        key=key,
        kind=kind,
        meta=meta,
    )
    if cache_write and cacheable:
        DATASETS.upsert(info)
    return ds.dataset_id
