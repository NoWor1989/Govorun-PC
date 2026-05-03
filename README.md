# Govorun PC

Голосовой ввод на русском для **Windows / Linux / macOS**. Полностью офлайн,
звук никуда не уходит. Аналог Android-приложения [govorun-lite](https://github.com/amidexe/govorun-lite),
только под десктоп.

Под капотом:
* **GigaAM v2** — модель распознавания речи от Сбера (MIT)
* **sherpa-onnx** — кросс-платформенный ONNX-рантайм (Apache 2.0)

## Что получится

Зажимаете **F8** (или нажимаете **Ctrl+Shift+Space**) — диктуете —
распознанный текст вставляется туда, где курсор. Работает в любом
приложении: Telegram, браузер, Word, IDE — куда угодно.

## Установка

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Модель (~330 МБ, один раз)
python download_models.py

# 3. Запуск
python govorun_pc.py            # toggle: Ctrl+Shift+Space — старт/стоп
python govorun_pc.py --hold     # push-to-talk: зажать F8
```

## Особенности по платформам

### Windows
Работает «из коробки». Если горячие клавиши не ловятся — запустите
терминал от имени администратора (библиотека `keyboard` иногда требует
повышенных прав для глобальных хуков).

### Linux
Библиотека `keyboard` читает `/dev/input/*`, поэтому **запускать нужно
с `sudo`** или добавить пользователя в группу `input`:
```bash
sudo usermod -a -G input $USER
# перелогиниться
```
Альтернатива без рута — заменить `keyboard` на `pynput` (см. ниже).

Также понадобится `xdotool` для вставки в X11 / `wtype` для Wayland —
их использует `pyautogui`/`pyperclip`:
```bash
sudo apt install xdotool xclip      # X11
sudo apt install wtype wl-clipboard # Wayland
```

### macOS
Нужно дать разрешения в **Системные настройки → Конфиденциальность**:
* **Доступ к микрофону** — терминалу/IDE
* **Доступ к универсальному управлению** (Accessibility) — для
  глобальных горячих клавиш и эмуляции `Cmd+V`
* **Мониторинг ввода** (Input Monitoring) — для библиотеки `keyboard`

## Настройка

Все параметры — в верхней части `govorun_pc.py`:

| Переменная | Что меняет |
|---|---|
| `HOTKEY_TOGGLE`   | комбинация для toggle-режима |
| `HOTKEY_HOLD`     | клавиша для push-to-talk |
| `SAMPLE_RATE`     | частота дискретизации (16 кГц для GigaAM, не трогайте) |
| `MIN_DURATION_SEC`| ниже этой длительности запись игнорируется |

## Если что-то не работает

**«Модель не найдена»** — запустите `python download_models.py`. Если
загрузка падает (URL устарел), скачайте свежий архив GigaAM вручную:
https://github.com/k2-fsa/sherpa-onnx/releases (тег `asr-models`),
распакуйте и положите `model.onnx` + `tokens.txt` в папку `./models/`.

**Микрофон не пишет** — проверьте, что система видит вход:
```python
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Текст не вставляется** — проверьте, что `pyperclip` работает в вашей
системе (`python -c "import pyperclip; pyperclip.copy('тест')"`). На
Linux нужен `xclip` или `wl-clipboard`.

**На Linux хочу без `sudo`** — замените `keyboard` на `pynput`:
```python
from pynput import keyboard as kb
def on_press(key):
    if key == kb.Key.f8: ctrl.press()
def on_release(key):
    if key == kb.Key.f8: ctrl.release()
with kb.Listener(on_press=on_press, on_release=on_release) as l:
    l.join()
```

## Что можно докрутить

* **VAD с автостопом** — добавить Silero VAD, чтобы запись сама
  останавливалась после паузы (как в оригинальном Govorun).
* **Системный трей** — `pystray` + иконка состояния.
* **Стриминговое распознавание** — sherpa-onnx умеет онлайн-режим
  (модели streaming-zipformer), и текст будет появляться по мере речи,
  а не в конце.
* **Замена hotkey без перезапуска** — мини-GUI на `tkinter`.

## Лицензия

MIT — делайте что хотите.
