import json
from pathlib import Path

from constants import JOB_IDS_JSON


class JobIds:
    """
    Class to manage JobIds data and dynamically update the corresponding json file
    """
    path: Path
    data: dict

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.loader()

    def loader(self):
        """
        Loads JobIds data from json file
        Creates and add null values if the file does not exist
        """
        if self.path.exists():
            with self.path.open("r") as f:
                self.data = json.load(f)
                return
        self.data = {
            "text": {"compiled_id": None, "dataset_id": None},
            "image": {"compiled_id": None, "dataset_id": None}
        }
        self.save()

    def save(self):
        """
        Saves JobIds data to json file
        """
        with self.path.open("w") as f:
            json.dump(self.data, f, indent=2)

    def __getitem__(self, key):
        outer, inner = key
        return self.data[outer][inner]

    def __setitem__(self, key, value):
        outer, inner = key
        self.data[outer][inner] = value
        self.save()

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __bool__(self):
        return bool(self.data)

    def __iter__(self):
        return iter(self.data.items())

    def __len__(self):
        return len(self.data)


JOB_IDS = JobIds(JOB_IDS_JSON)


