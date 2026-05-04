from __future__ import annotations

import os
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

def _debug_enabled() -> bool:
    v = os.getenv("QAI_DATASET_CACHE_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


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

    debug = _debug_enabled()
    if debug:
        print(
            f"[dataset-cache] request key={key!r} kind={kind!r} cache={cache} cache_write={cache_write} cacheable={cacheable}"
        )

    if cache and cacheable and key:
        existing = DATASETS.find_by_key(key)
        if debug:
            if existing is None:
                print("[dataset-cache] lookup: MISS (no matching key in datasets.json)")
            else:
                print(
                    f"[dataset-cache] lookup: HIT id={existing.dataset_id} expires={existing.expiration_time} name={existing.name!r}"
                )
        if existing is not None:
            # Prefer trusting local expiration_time to avoid API calls that can fail and
            # cause unnecessary re-uploads. Only revalidate against Hub when missing or
            # close to expiry.
            now = _utc_now()
            if existing.expiration_time is not None and existing.expiration_time > (now + timedelta(minutes=5)):
                if debug:
                    print("[dataset-cache] reuse: local expiration OK; returning cached dataset id")
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
                    if debug:
                        print("[dataset-cache] reuse: revalidated against Hub; returning cached dataset id")
                    return existing.dataset_id
            except Exception as e:
                # If lookup fails, fall through to upload a new dataset.
                if debug:
                    print(f"[dataset-cache] reuse: Hub get_dataset failed ({type(e).__name__}: {e}); will upload new")
                pass

    if debug and (not key or not cache or not cacheable):
        reason = []
        if not cache:
            reason.append("cache disabled")
        if not cacheable:
            reason.append("cacheable=False")
        if not key:
            reason.append("no key")
        print(f"[dataset-cache] bypass cache: {', '.join(reason) if reason else 'unknown'}; will upload new")

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
        if debug:
            print(f"[dataset-cache] upload: wrote to cache id={ds.dataset_id} expires={info.expiration_time}")
    elif debug:
        print(f"[dataset-cache] upload: NOT written to cache (cache_write={cache_write}, cacheable={cacheable}) id={ds.dataset_id}")
    return ds.dataset_id
