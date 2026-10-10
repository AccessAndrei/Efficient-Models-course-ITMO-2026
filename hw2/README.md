# ДЗ-2: Super-Convergence

Воспроизведены сравнения LeNet/MNIST из таблицы 2 и ResNet-56/CIFAR-10 из рисунка 1a статьи [Smith и Topin](https://arxiv.org/pdf/1708.07120). Для обучения использована GPU NVIDIA RTX PRO 6000 Blackwell Server Edition, 94,97 GiB видеопамяти. Вычисления выполнены в FP32, с PyTorch 2.10.0 и CUDA 12.8.

## Материалы для сдачи

- [Отчёт, 8 страниц](report.pdf).
- [Презентация, 17 слайдов, PDF](presentation.pdf) для показа.
- [Редактируемая презентация PPTX](presentation.pptx).
- Код обучения: `models.py`, `optim.py`, `data.py`, `train.py`.
- Исходные журналы и конфигурации: `results/`.

PDF презентации соответствует проверенному рендеру. PPTX содержит редактируемые тексты, графики и таблицы; LibreOffice может иначе отображать его scatter-графики.

## Результат

| Задача | Baseline | Циклический режим | Обновления baseline / cycle |
|---|---:|---:|---:|
| CIFAR-10 | 90,54% | 91,95% | 80 000 / 10 000 |
| MNIST | 99,02 ± 0,05% | 99,13 ± 0,04% | 10 000 / 1407 |

Здесь указана финальная test accuracy. MNIST: среднее ± выборочное SD по трём seeds; CIFAR-10: один seed. Дополнительный MNIST-baseline на 1407 шагах получил 98,63 ± 0,05%. Все 11 основных обучений завершены; два пилота по 200 шагов хранятся отдельно.

Записанное время обучения CIFAR-10: 100,85 минуты baseline и 12,61 минуты CLR. Это длительности полных протоколов; загрузка данных, оценка и запись файлов в них не входят. Подробные сравнения и ограничения приведены в отчёте.

## Повторить обучение

Команды выполняются из корня репозитория в окружении Python с поддержкой CUDA:

```bash
python -m pip install -r hw2/requirements.txt pytest==8.3.5
python -m pytest hw2 -q
for mode in cifar_baseline cifar_cycle; do
  python hw2/train.py "$mode" --device cuda --seed 42 --output-dir hw2/results/repeat
done
for mode in mnist_baseline mnist_cycle mnist_short; do
  for seed in 42 43 44; do
    python hw2/train.py "$mode" --device cuda --seed "$seed" --output-dir hw2/results/repeat
  done
done
```

Новые запуски сохраняются в `hw2/results/repeat`, отдельно от приложенных результатов. В каждом запуске записываются параметры, версии, хеши исходников и временные отметки. Для продолжения незавершённого запуска предусмотрен `--resume`; требуются checkpoint и то же окружение. На Mac доступен `--device mps`; время и численная траектория могут отличаться.

## Пересчитать таблицы и графики

```bash
python -m pip install -r hw2/requirements-report.txt
python hw2/analyze.py
python hw2/build_report.py
```

`analyze.py` читает исходные CSV и создаёт `results/analysis.json`, `results/summary.csv` и шесть графиков в `results/figures/`. `build_report.py` собирает зафиксированный отчёт по этим экспериментам. На macOS используется системный PT Serif; на других платформах — DejaVu Serif из Matplotlib. Каталог DejaVu Serif и DejaVu Sans Mono можно явно задать через `REPORT_FONT_DIR`. При изменении результатов текст выводов требует пересмотра.

## Что отличается от оригинала

Сохранены авторская архитектура ResNet, бюджеты шагов, общий batch, формы LR и Caffe momentum. На одной GPU BatchNorm нормализует восемь групп по 125 изображений. Для MNIST конечный LR `1e-5` выбран явно, потому что статья не уточняет его. CIFAR test set проверяется один раз целиком. Реализация перенесена из Caffe в PyTorch; точное совпадение чисел и побитовая воспроизводимость не заявляются.

При сравнении CIFAR меняются одновременно LR и BN decay. Один seed CIFAR не оценивает межзапусковую устойчивость; преимущества отдельных компонентов не измерялись отдельно.
