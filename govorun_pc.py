#!/usr/bin/env python3
"""
Govorun PC — голосовой ввод на русском для Windows.
Офлайн, на основе GigaAM v2 (Сбер) + sherpa-onnx.

Использование:
    python govorun_pc.py               # запуск с иконкой в трее
    python govorun_pc.py --device 2    # выбрать микрофон
    python govorun_pc.py --list-devices

Требования:
    pip install -r requirements.txt
    python download_models.py          # один раз, ~330 МБ
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Перенаправляем кэши PyTorch/HuggingFace в домашнюю папку
# (до любых импортов torch/transformers)
_CACHE = Path.home() / ".cache" / "govorun"
os.environ.setdefault("HF_HOME",            str(_CACHE / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE",  str(_CACHE / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME",          str(_CACHE / "torch"))

import numpy as np
import sounddevice as sd
import sherpa_onnx
import keyboard
import pyperclip
import pystray
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Конфигурация (изменяется через настройки)
# ============================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
MODELS_DIR  = SCRIPT_DIR / "models"
MODEL_PATH  = MODELS_DIR / "model.onnx"
TOKENS_PATH = MODELS_DIR / "tokens.txt"

SAMPLE_RATE     = 16_000
FEATURE_DIM     = 80
MIN_DURATION_SEC = 0.3

HOTKEY: str       = "alt+x"
INPUT_DEVICE: int | None = None


# ============================================================
# Иконки трея (рисуются через Pillow)
# ============================================================
def _make_icon(recording: bool) -> Image.Image:
    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Фон: серый в покое, красный при записи
    bg = (210, 40, 40, 255) if recording else (60, 60, 60, 255)
    draw.ellipse([2, 2, 62, 62], fill=bg)

    # Тело микрофона
    draw.rounded_rectangle([24, 10, 40, 36], radius=8, fill="white")

    # Дуга-подставка
    draw.arc([18, 28, 46, 50], start=180, end=0, fill="white", width=3)

    # Ножка
    draw.line([32, 50, 32, 57], fill="white", width=3)

    # Основание
    draw.line([24, 57, 40, 57], fill="white", width=3)

    # Мигающая точка при записи
    if recording:
        draw.ellipse([46, 4, 60, 18], fill=(255, 80, 80, 255))

    return img


ICON_IDLE      = _make_icon(False)
ICON_RECORDING = _make_icon(True)


# ============================================================
# Запись звука
# ============================================================
class Recorder:
    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.frames: list[np.ndarray] = []
        self.recording = False
        self.stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[!] sounddevice: {status}")
        with self._lock:
            if self.recording:
                self.frames.append(indata.copy())

    def start(self) -> None:
        with self._lock:
            self.frames = []
            self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=INPUT_DEVICE,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        with self._lock:
            self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self._lock:
            if not self.frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.frames).flatten().astype(np.float32)


# ============================================================
# Распознавание
# ============================================================
def load_recognizer() -> sherpa_onnx.OfflineRecognizer:
    if not MODEL_PATH.exists() or not TOKENS_PATH.exists():
        print(f"[!] Модель не найдена в {MODELS_DIR}", file=sys.stderr)
        print("    Запустите:  python download_models.py", file=sys.stderr)
        sys.exit(1)
    print("⏳ Загружаю GigaAM...")
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(MODEL_PATH),
        tokens=str(TOKENS_PATH),
        num_threads=4,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method="greedy_search",
    )
    print("✅ Модель готова.\n")
    return rec


def recognize(rec: sherpa_onnx.OfflineRecognizer, audio: np.ndarray) -> str:
    if audio.size / SAMPLE_RATE < MIN_DURATION_SEC:
        return ""
    stream = rec.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    rec.decode_stream(stream)
    return stream.result.text.strip()


# ============================================================
# Пунктуация (ONNX, без PyTorch)
# ============================================================
_punct_model = None

def load_punct_model() -> bool:
    global _punct_model
    try:
        from punctuators.models import PunctCapSegModelONNX
        print("⏳ Загружаю модель пунктуации (ONNX)...")
        _punct_model = PunctCapSegModelONNX.from_pretrained(
            "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
        )
        print("✅ Пунктуация готова.\n")
        return True
    except ImportError:
        print("⚠️  punctuators не установлен — пунктуация отключена.\n")
        return False
    except Exception as e:
        print(f"⚠️  Не удалось загрузить модель пунктуации: {e}\n")
        return False


def restore_punctuation(text: str) -> str:
    if not text:
        return text
    if _punct_model is not None:
        try:
            results = _punct_model.infer([text])
            return "".join(results[0]) if results else text
        except Exception as e:
            print(f"[!] Ошибка пунктуации: {e}")
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


# ============================================================
# Вставка текста
# ============================================================
def paste_text(text: str) -> None:
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")


# ============================================================
# Контроллер: Alt+X → старт/стоп
# ============================================================
class Controller:
    def __init__(self, rec: sherpa_onnx.OfflineRecognizer):
        self.rec      = rec
        self.recorder = Recorder()
        self._busy      = False
        self._busy_lock = threading.Lock()
        self.on_state_change: callable = lambda recording: None  # колбэк для трея

    def toggle(self, _event=None) -> None:
        with self._busy_lock:
            if self._busy:
                return
        if self.recorder.recording:
            with self._busy_lock:
                self._busy = True
            threading.Thread(target=self._process, daemon=True).start()
        else:
            print(f"🎤 Запись... ({HOTKEY.upper()} — остановить)")
            self.recorder.start()
            self.on_state_change(True)

    def _process(self) -> None:
        self.on_state_change(False)
        audio = self.recorder.stop()
        dur   = audio.size / SAMPLE_RATE
        rms   = float(np.sqrt(np.mean(audio ** 2))) if audio.size > 0 else 0.0
        print(f"⏹  Записано {dur:.1f}с, уровень: {rms:.4f}, распознаю...")
        try:
            text = recognize(self.rec, audio)
        except Exception as e:
            print(f"[!] Ошибка распознавания: {e}")
            text = ""
        if text:
            text = restore_punctuation(text)
            print(f"📝 {text}")
            paste_text(text)
            print("✅ Скопировано и вставлено\n")
        else:
            print("(тишина или слишком короткая запись)\n")
        with self._busy_lock:
            self._busy = False


# ============================================================
# Окно настроек (tkinter)
# ============================================================
class SettingsWindow:
    def __init__(self, root: tk.Tk, ctrl: Controller, tray: "TrayApp"):
        self.ctrl = ctrl
        self.tray = tray
        self.win  = tk.Toplevel(root)
        self.win.title("Govorun — настройки")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self._build()
        # Центрируем окно
        self.win.update_idletasks()
        w, h = self.win.winfo_width(), self.win.winfo_height()
        x = (self.win.winfo_screenwidth()  - w) // 2
        y = (self.win.winfo_screenheight() - h) // 2
        self.win.geometry(f"{w}x{h}+{x}+{y}")

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        f   = tk.Frame(self.win, padx=16, pady=12)
        f.pack(fill="both", expand=True)

        # --- Горячая клавиша ---
        tk.Label(f, text="Горячая клавиша:", anchor="w").grid(
            row=0, column=0, sticky="w", **pad)
        self._hotkey_var = tk.StringVar(value=HOTKEY)
        tk.Entry(f, textvariable=self._hotkey_var, width=20).grid(
            row=0, column=1, sticky="ew", **pad)

        # --- Микрофон ---
        tk.Label(f, text="Микрофон:", anchor="w").grid(
            row=1, column=0, sticky="w", **pad)
        devices     = sd.query_devices()
        self._dev_names = [
            d["name"] for d in devices if d["max_input_channels"] > 0
        ]
        self._dev_ids = [
            i for i, d in enumerate(devices) if d["max_input_channels"] > 0
        ]
        current_name = (
            devices[INPUT_DEVICE]["name"]
            if INPUT_DEVICE is not None
            else devices[sd.default.device[0]]["name"]
        )
        self._device_var = tk.StringVar(value=current_name)
        ttk.Combobox(
            f,
            textvariable=self._device_var,
            values=self._dev_names,
            state="readonly",
            width=30,
        ).grid(row=1, column=1, sticky="ew", **pad)

        # --- Пунктуация ---
        self._punct_var = tk.BooleanVar(value=_punct_model is not None)
        tk.Checkbutton(
            f, text="Восстанавливать пунктуацию", variable=self._punct_var
        ).grid(row=2, columnspan=2, sticky="w", **pad)

        # --- Кнопки ---
        btn_frame = tk.Frame(f)
        btn_frame.grid(row=3, columnspan=2, pady=(8, 0))
        tk.Button(btn_frame, text="Применить", width=12,
                  command=self._apply).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Отмена",    width=12,
                  command=self.win.destroy).pack(side="left", padx=4)

    def _apply(self) -> None:
        global HOTKEY, INPUT_DEVICE, _punct_model

        # Горячая клавиша
        new_hotkey = self._hotkey_var.get().strip().lower()
        if new_hotkey and new_hotkey != HOTKEY:
            try:
                keyboard.remove_hotkey(HOTKEY)
                keyboard.add_hotkey(new_hotkey, self.ctrl.toggle, suppress=True)
                HOTKEY = new_hotkey
                print(f"⌨️  Хоткей изменён на [{HOTKEY.upper()}]")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось установить хоткей:\n{e}")
                return

        # Микрофон
        chosen_name = self._device_var.get()
        if chosen_name in self._dev_names:
            idx = self._dev_ids[self._dev_names.index(chosen_name)]
            INPUT_DEVICE = idx
            print(f"🎙  Микрофон: [{idx}] {chosen_name}")

        # Пунктуация
        want_punct = self._punct_var.get()
        if want_punct and _punct_model is None:
            load_punct_model()
        elif not want_punct:
            _punct_model = None

        self.tray.update_tooltip()
        self.win.destroy()


# ============================================================
# Оверлей записи — маленький бейдж поверх всех окон
# ============================================================
class RecordingOverlay:
    W, H     = 120, 36
    BG       = "#1e1e1e"
    DOT_ON   = "#ff3b3b"
    DOT_OFF  = "#7a1010"
    FG       = "#ffffff"

    def __init__(self, root: tk.Tk):
        self._root   = root
        self._win: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None
        self._dot_id: int | None = None
        self._blink_on = True
        self._blink_job = None

    def show(self) -> None:
        if self._win and self._win.winfo_exists():
            return

        win = tk.Toplevel(self._root)
        win.overrideredirect(True)          # без рамки и заголовка
        win.attributes("-topmost", True)    # поверх всех окон
        win.attributes("-alpha", 0.88)
        win.resizable(False, False)

        # Позиция: правый нижний угол над панелью задач
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x  = sw - self.W - 16
        y  = sh - self.H - 52
        win.geometry(f"{self.W}x{self.H}+{x}+{y}")

        c = tk.Canvas(win, width=self.W, height=self.H,
                      bg=self.BG, highlightthickness=0)
        c.pack()

        # Скруглённый фон
        r = 10
        c.create_arc(0,        0,        2*r, 2*r, start=90,  extent=90,  fill=self.BG, outline=self.BG)
        c.create_arc(self.W-2*r, 0,      self.W, 2*r, start=0, extent=90, fill=self.BG, outline=self.BG)
        c.create_rectangle(r, 0, self.W-r, self.H, fill=self.BG, outline=self.BG)
        c.create_rectangle(0, r, self.W, self.H-r, fill=self.BG, outline=self.BG)

        # Мигающая точка
        dot_x, dot_y = 16, self.H // 2
        self._dot_id = c.create_oval(
            dot_x - 6, dot_y - 6, dot_x + 6, dot_y + 6,
            fill=self.DOT_ON, outline=""
        )

        # Текст
        c.create_text(dot_x + 14, dot_y, anchor="w",
                      text="● REC", fill=self.FG,
                      font=("Segoe UI", 11, "bold"))

        self._win    = win
        self._canvas = c
        self._blink_on = True
        self._blink()

    def hide(self) -> None:
        if self._blink_job:
            self._root.after_cancel(self._blink_job)
            self._blink_job = None
        if self._win and self._win.winfo_exists():
            self._win.destroy()
        self._win    = None
        self._canvas = None
        self._dot_id = None

    def _blink(self) -> None:
        if self._canvas and self._dot_id:
            color = self.DOT_ON if self._blink_on else self.DOT_OFF
            self._canvas.itemconfig(self._dot_id, fill=color)
            self._blink_on = not self._blink_on
            self._blink_job = self._root.after(500, self._blink)


# ============================================================
# Трей
# ============================================================
class TrayApp:
    def __init__(self, ctrl: Controller, root: tk.Tk):
        self.ctrl    = ctrl
        self.root    = root
        self.overlay = RecordingOverlay(root)
        self._icon: pystray.Icon | None = None
        ctrl.on_state_change = self._on_state_change

    def _on_state_change(self, recording: bool) -> None:
        if self._icon is not None:
            self._icon.icon  = ICON_RECORDING if recording else ICON_IDLE
            self._icon.title = "🎤 Запись идёт..." if recording else f"Govorun  [{HOTKEY.upper()}]"
        # Оверлей — в главном потоке tkinter
        if recording:
            self.root.after(0, self.overlay.show)
        else:
            self.root.after(0, self.overlay.hide)

    def update_tooltip(self) -> None:
        if self._icon:
            self._icon.title = f"Govorun  [{HOTKEY.upper()}]"

    def _open_settings(self) -> None:
        # Открываем окно в главном потоке tkinter
        self.root.after(0, lambda: SettingsWindow(self.root, self.ctrl, self))

    def _quit(self) -> None:
        if self._icon:
            self._icon.stop()
        self.root.after(0, self.root.quit)

    def run(self) -> None:
        menu = pystray.Menu(
            pystray.MenuItem("Настройки", lambda icon, item: self._open_settings()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход",     lambda icon, item: self._quit()),
        )
        self._icon = pystray.Icon(
            name="govorun",
            icon=ICON_IDLE,
            title=f"Govorun  [{HOTKEY.upper()}]",
            menu=menu,
        )
        # Трей — в отдельном потоке, tkinter — в главном
        threading.Thread(target=self._icon.run, daemon=True).start()
        self.root.mainloop()


# ============================================================
# Утилиты
# ============================================================
def print_devices() -> None:
    print("\n📋 Доступные устройства ввода:")
    devices   = sd.query_devices()
    default_in = sd.default.device[0]
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ◄ дефолт" if i == default_in else ""
            print(f"   [{i}] {d['name']}{marker}")
    print()


# ============================================================
# Точка входа
# ============================================================
def main() -> None:
    global INPUT_DEVICE

    parser = argparse.ArgumentParser(description="Голосовой ввод на ПК (офлайн, GigaAM)")
    parser.add_argument("--device",       type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-punct",     action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    print_devices()

    if args.device is not None:
        INPUT_DEVICE = args.device
        print(f"🎙  Выбрано устройство [{INPUT_DEVICE}]: {sd.query_devices(INPUT_DEVICE)['name']}\n")
    else:
        default_in = sd.default.device[0]
        dev_name   = sd.query_devices(default_in)["name"] if default_in >= 0 else "неизвестно"
        print(f"🎙  Используется дефолтное устройство [{default_in}]: {dev_name}\n")

    if not args.no_punct:
        load_punct_model()

    rec  = load_recognizer()
    ctrl = Controller(rec)

    keyboard.add_hotkey(HOTKEY, ctrl.toggle, suppress=True)
    print(f"🎯 [{HOTKEY.upper()}] — старт/стоп записи.")
    print("   Настройки и выход — иконка в системном трее.\n")

    # Скрытый tkinter root + трей
    root = tk.Tk()
    root.withdraw()

    tray = TrayApp(ctrl, root)
    tray.run()


if __name__ == "__main__":
    main()
