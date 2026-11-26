# AVSS: аудиовизуальное разделение речи

<p align="center">
  <a href="#ссылки">Ссылки</a> •
  <a href="#о-проекте">О проекте</a> •
  <a href="#установка">Установка</a> •
  <a href="#лицензия">Лицензия</a>
</p>

## Ссылки

- [Отчёт](final_report.pdf) (исходник `main.tex`)
- Логи обучения в CometML:
  - [Andrew Teplov](https://www.comet.com/andrew-teplov/avss-teplov)
  - [Nikita Romanenko](https://www.comet.com/niki-romanenko/avss/view/new/panels)
- [Модель](https://huggingface.co/romanenko02/dprnn_video/tree/main) — скачивается скриптом `uv run python download_model.py` в `ckpt/model_best.pth` вместе с энкодером для обучения `ckpt/encoder_ce_model.pth`

## О проекте

Репозиторий содержит модели и конфиги для разделения речи двух дикторов во временной области по одноканальной смеси и видео области рта каждого диктора.

Аудиомодели:

- **Conv-TasNet** - свёрточный TCN-разделитель во временной области;
- **DPRNN** - dual-path RNN для моделирования длинного контекста;
- **SepFormer** - dual-path модель на трансформерах.

Видеоэнкодеры для Conv-TasNet и DPRNN:

- **Conv3D** - трёхмерная свёртка по соседним кадрам области рта;
- **ViT** - покадровый Vision Transformer.

Финальная модель - **DPRNN + ViT** с вспомогательной функцией потерь, которая сопоставляет видео диктора с его чистой речью.

| SI-SNR | SI-SNRi | SDRi | PESQ | STOI |
|---|---|---|---|---|
| 12.69 | 12.69 | 13.09 | 2.21 | 0.91 |

Метрики посчитаны на валидационной выборке.

## Установка

Зависимости описаны в `pyproject.toml` и зафиксированы в `uv.lock`.

```bash
git clone https://github.com/ndromanenko/avss.git
cd avss
pip install uv
uv python install 3.12
uv sync --python 3.12 --locked
```

## Использование

### Данные

Папка с данными должна иметь формат CustomDir:

```
DATA_DIR
├── audio
│   ├── mix   # смеси FirstSpeakerID_SecondSpeakerID.wav
│   ├── s1    # эталоны, могут отсутствовать
│   └── s2
└── mouths    # SpeakerID.npz
```

### Обучение

Полный запуск обучения финальной модели:

```bash
# 1. окружение
git clone https://github.com/ndromanenko/avss.git
cd avss
pip install uv
uv python install 3.12
uv sync --python 3.12 --locked

# 2. чекпоинты в ckpt/
uv run python download_model.py

# 3. логин в CometML
uv run comet login

# 4. обучение
uv run train_separation.py -cn final_model
```

- `download_model.py` сохраняет в `ckpt/` 2 файла:
  - `model_best.pth` - финальная модель;
  - `encoder_ce_model.pth` - Conv-TasNet, обученный 20 эпох только по звуку. Его замороженный энкодер нужен вспомогательной функции потерь, чтобы получать признаки чистых дорожек. Путь задаётся параметром `trainer.audio_encoder_checkpoint`.
- Все гиперпараметры финальной модели собраны в `src/configs/final_model.yaml`, их можно менять через Hydra, например `trainer.n_epochs=50`.

### Инференс

Инференс показан в `demo.ipynb`: загрузите ноутбук в Google Colab или Kaggle с GPU и запускайте ячейки по порядку. Ноутбук клонирует репозиторий, ставит зависимости, скачивает чекпоинт и тестовые данные, запускает инференс, считает метрики и замеряет эффективность.

Те же шаги из командной строки:

```bash
# чекпоинт в ckpt/model_best.pth
uv run python download_model.py

# разделение: предсказания сохраняются в predictions/test/s1 и predictions/test/s2
# с теми же именами, что и смеси
uv run inference.py \
    datasets.test.dir=<DATA_DIR> \
    inferencer.from_pretrained=ckpt/model_best.pth \
    inferencer.save_path=predictions \
    'inferencer.device_tensors=[mix,targets,video]'
```

Если в `<DATA_DIR>` нет эталонов `audio/s1`, `audio/s2`, используйте `'inferencer.device_tensors=[mix,video]'`.

### Метрики

Скрипт считает SI-SNR, SI-SNRi, SDRi, PESQ и STOI по эталонным и предсказанным дорожкам:

```bash
uv run calc_metrics.py target_dir=<DATA_DIR>/audio pred_dir=predictions/test
```

`target_dir` должен содержать `mix`, `s1`, `s2`, а `pred_dir` - `s1`, `s2`.

### Эффективность

Скрипт замеряет число параметров, размер весов на диске, MACs, пиковую память GPU и время инференса на 1 батче из 1 смеси и видео обоих дикторов:

```bash
uv run measure_efficiency.py checkpoint=ckpt/model_best.pth data_dir=<DATA_DIR>
```

| Параметры | Размер на диске | MACs | Пиковая память GPU | Время на батч (Kaggle, T4) |
|---|---|---|---|---|
| 4.45 M | 17.04 MB | 57.38 G | 395.2 MB | 97.18 ms |

## Лицензия

[MIT](LICENSE)
