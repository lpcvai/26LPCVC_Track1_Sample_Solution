import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    # We persist ISO-8601. If it's naive, treat it as UTC to make comparisons deterministic.
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _dt_to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    name: str | None
    expiration_time: datetime | None
    # Stable cache identity for the dataset contents, derived from HF dataset metadata
    # plus our own upload parameters.
    key: str | None = None
    kind: str | None = None  # e.g. "image_batch", "text", "calib_image", "calib_text"
    meta: dict[str, Any] | None = None

    def is_expired(self, *, now: datetime | None = None) -> bool:
        if self.expiration_time is None:
            return False
        n = now or _utc_now()
        return n >= self.expiration_time

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "name": self.name,
            "expiration_time": _dt_to_iso(self.expiration_time),
            "key": self.key,
            "kind": self.kind,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "DatasetInfo":
        ds_id = obj.get("id")
        if not isinstance(ds_id, str) or not ds_id.strip():
            raise ValueError("missing/invalid dataset id")

        name = obj.get("name")
        if name is not None and not isinstance(name, str):
            name = str(name)

        exp = _parse_dt(obj.get("expiration_time"))

        key = obj.get("key")
        if key is not None and not isinstance(key, str):
            key = str(key)
        kind = obj.get("kind")
        if kind is not None and not isinstance(kind, str):
            kind = str(kind)
        meta = obj.get("meta")
        if meta is not None and not isinstance(meta, dict):
            meta = None

        return DatasetInfo(
            dataset_id=ds_id.strip(),
            name=name.strip() if isinstance(name, str) else name,
            expiration_time=exp,
            key=key.strip() if isinstance(key, str) else key,
            kind=kind.strip() if isinstance(kind, str) else kind,
            meta=meta,
        )


class DatasetRegistry:
    """
    Small local registry for uploaded Hub datasets. This is intentionally conservative:
    - We treat the saved expiration time as the source of truth for pruning.
    - We prune expired/invalid entries on load (as requested).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, DatasetInfo] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            self._save()
            return

        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            # If the file is corrupt, start fresh rather than blocking runs.
            raw = {}

        items = raw.get("datasets") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            items = []

        now = _utc_now()
        cleaned: dict[str, DatasetInfo] = {}
        changed = False
        for item in items:
            if not isinstance(item, dict):
                changed = True
                continue
            try:
                info = DatasetInfo.from_json(item)
            except Exception:
                changed = True
                continue
            # If we can't identify what this dataset is, it's not safe/useful to keep.
            if not info.key:
                changed = True
                continue
            if info.is_expired(now=now):
                changed = True
                continue
            cleaned[info.dataset_id] = info

        self._data = cleaned
        if changed:
            self._save()

    def _save(self) -> None:
        payload = {"datasets": [info.to_json() for info in self._data.values()]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def upsert(self, info: DatasetInfo) -> None:
        self._data[info.dataset_id] = info
        self._save()

    def find_by_key(self, key: str) -> DatasetInfo | None:
        now = _utc_now()
        candidates = [info for info in self._data.values() if info.key == key and not info.is_expired(now=now)]
        if not candidates:
            return None
        # Prefer permanent datasets (expiration_time=None), otherwise the one that expires last.
        best: DatasetInfo | None = None
        best_exp: datetime | None = None
        for c in candidates:
            exp = c.expiration_time or datetime.max.replace(tzinfo=timezone.utc)
            if best is None or exp > best_exp:
                best = c
                best_exp = exp
        return best
