#!/usr/bin/env python3
"""Chẩn đoán: chỉ ra CHÍNH XÁC cụm 6-từ nào trong một file đang khớp
fingerprint của bộ brief riêng.

AN TOÀN: script này chỉ đọc file CỦA BẠN (scripts/selfeval.py — công khai,
do bạn hoặc người soạn đề viết) và tập hash đã có sẵn trên máy bạn
(tests/private_fingerprints.json). Nó không đọc, không cần, và không thể
suy ra nội dung brief riêng — salted SHA-256 là hàm một chiều, nên biết
hash không cho phép đảo ngược ra câu gốc. Việc in ra "câu khớp" ở đây là
in nguyên văn từ CHÍNH FILE CỦA BẠN, không phải từ brief riêng.

Chạy:
    python3 find_leak.py scripts/selfeval.py

Không commit file này. Xoá sau khi dùng xong.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
FINGERPRINTS_PATH = REPO_ROOT / "tests" / "private_fingerprints.json"

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFC", str(text)).casefold())


def _shingles_with_spans(text: str, size: int):
    matches = list(_WORD_RE.finditer(unicodedata.normalize("NFC", text).casefold()))
    if not matches:
        return
    n = len(matches)
    if n <= size:
        shingle = " ".join(m.group(0) for m in matches)
        yield shingle, matches[0].start(), matches[-1].end()
        return
    for i in range(n - size + 1):
        window = matches[i : i + size]
        shingle = " ".join(m.group(0) for m in window)
        yield shingle, window[0].start(), window[-1].end()


def _fingerprint(shingle: str, salt: str) -> str:
    return hashlib.sha256((salt + "\x1f" + shingle).encode("utf-8")).hexdigest()[:16]


def main():
    if len(sys.argv) != 2:
        print("Dùng: python3 find_leak.py <đường-dẫn-file-cần-soi>")
        sys.exit(1)

    target = REPO_ROOT / sys.argv[1]
    if not target.exists():
        print(f"Không tìm thấy file: {target}")
        sys.exit(1)

    if not FINGERPRINTS_PATH.exists():
        print(f"Không tìm thấy {FINGERPRINTS_PATH} — cần file này để so khớp.")
        sys.exit(1)

    data = json.loads(FINGERPRINTS_PATH.read_text(encoding="utf-8"))
    salt = data["salt"]
    size = data["shingle"]
    tier_all = set(data["tier_all"])
    tier_prose = set(data.get("tier_prose", []))

    original = target.read_text(encoding="utf-8")
    hits = []
    for shingle, start, end in _shingles_with_spans(original, size):
        fp = _fingerprint(shingle, salt)
        which = None
        if fp in tier_all:
            which = "tier_all"
        elif fp in tier_prose:
            which = "tier_prose"
        if which:
            line_no = original.count("\n", 0, start) + 1
            hits.append((line_no, which, shingle, start, end))

    if not hits:
        print(f"Không tìm thấy khớp nào trong {target} (có thể đã sửa xong).")
        return

    print(f"Tìm thấy {len(hits)} chỗ khớp trong {target}:\n")
    for line_no, which, shingle, start, end in hits:
        context_start = max(0, start - 40)
        context_end = min(len(original), end + 40)
        context = original[context_start:context_end].replace("\n", " ⏎ ")
        print(f"  Dòng ~{line_no}  [{which}]")
        print(f"    cụm khớp (đã chuẩn hoá): {shingle!r}")
        print(f"    ngữ cảnh trong file gốc: ...{context}...")
        print()


if __name__ == "__main__":
    main()