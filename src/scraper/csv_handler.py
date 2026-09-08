import csv
from pathlib import Path


class CSVWriter:
    """Append rows to a CSV file, writing the header only when the file is new."""

    def __init__(self, path: Path, headers: list[str], buffer_size: int = 12):
        self.path = Path(path)
        self.headers = headers
        self.buffer_size = buffer_size
        self.buffer: list[list[str]] = []
        self.file = None
        self.writer = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self.file = open(self.path, mode="a", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(self.headers)

    def write_row(self, row: list[str]) -> None:
        if any(row):
            self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        self.writer.writerows(self.buffer)
        self.buffer = []
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.flush()
            self.file.close()
            self.file = None
