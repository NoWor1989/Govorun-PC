#!/usr/bin/env python3
"""
Загрузчик ONNX-модели GigaAM v2 для Govorun PC.

Скачивает официальную сборку sherpa-onnx + Сбер (формат NeMo CTC, ~330 МБ),
распаковывает и кладёт нужные файлы в ./models/

Если URL устарел — см. свежие модели на:
    https://github.com/k2-fsa/sherpa-onnx/releases  (теги asr-models)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/russian.html
"""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# Официальный архив с GigaAM v2 (CTC) от sherpa-onnx
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2"
)
ARCHIVE_NAME = "gigaam.tar.bz2"

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download(url: str, dst: Path) -> None:
    print(f"⬇  Качаю {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                f.write(chunk)
                bar.update(len(chunk))


def extract(archive: Path, out: Path) -> Path:
    print(f"📦 Распаковываю {archive.name}...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(out)
    # внутри архива одна папка — возвращаем её путь
    for item in out.iterdir():
        if item.is_dir() and item.name.startswith("sherpa-onnx-"):
            return item
    raise RuntimeError("Не нашёл папку с моделью внутри архива")


def find_files(folder: Path) -> tuple[Path, Path]:
    """Ищем model.onnx (или model.int8.onnx) и tokens.txt"""
    onnx = None
    for cand in ("model.int8.onnx", "model.onnx"):
        p = folder / cand
        if p.exists():
            onnx = p
            break
    if onnx is None:
        # рекурсивно
        onnx_files = list(folder.rglob("*.onnx"))
        if not onnx_files:
            raise RuntimeError("ONNX-файл не найден")
        onnx = onnx_files[0]

    tokens = folder / "tokens.txt"
    if not tokens.exists():
        tokens_files = list(folder.rglob("tokens.txt"))
        if not tokens_files:
            raise RuntimeError("tokens.txt не найден")
        tokens = tokens_files[0]
    return onnx, tokens


def main() -> int:
    archive = MODELS_DIR / ARCHIVE_NAME
    target_model  = MODELS_DIR / "model.onnx"
    target_tokens = MODELS_DIR / "tokens.txt"

    if target_model.exists() and target_tokens.exists():
        print(f"✅ Модель уже на месте: {target_model}")
        return 0

    try:
        if not archive.exists():
            download(MODEL_URL, archive)
        else:
            print(f"📦 Архив уже скачан: {archive.name}")

        extracted = extract(archive, MODELS_DIR)
        onnx, tokens = find_files(extracted)

        shutil.copy2(onnx,   target_model)
        shutil.copy2(tokens, target_tokens)

        # чистим за собой
        shutil.rmtree(extracted, ignore_errors=True)
        archive.unlink(missing_ok=True)

        print(f"\n✅ Готово.\n   {target_model}\n   {target_tokens}")
        return 0

    except Exception as e:
        print(f"\n[!] Ошибка: {e}", file=sys.stderr)
        print(
            "\nЕсли URL устарел — скачайте свежий архив GigaAM вручную:\n"
            "  https://github.com/k2-fsa/sherpa-onnx/releases (тег asr-models)\n"
            "распакуйте и положите model.onnx + tokens.txt в папку ./models/",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
