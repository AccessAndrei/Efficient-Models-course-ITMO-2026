"""Build the fixed reproduction report from the recorded experiment summaries."""

import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).parent
WIDTH = letter[0] - 112


def register_fonts() -> None:
    """Prefer PT Serif; allow a portable DejaVu Serif font directory override."""
    serif_collection = Path("/System/Library/Fonts/Supplemental/PTSerif.ttc")
    override = os.environ.get("REPORT_FONT_DIR")
    if serif_collection.is_file() and not override:
        for name, index in (("Body", 0), ("Body-Bold", 3), ("Body-Italic", 1)):
            pdfmetrics.registerFont(
                TTFont(name, str(serif_collection), subfontIndex=index)
            )
        pdfmetrics.registerFont(
            TTFont("Mono", "/System/Library/Fonts/Menlo.ttc", subfontIndex=0)
        )
        pdfmetrics.registerFont(
            TTFont("Math", "/System/Library/Fonts/Supplemental/Times New Roman.ttf")
        )
    else:
        if override:
            font_dir = Path(override)
        else:
            import matplotlib

            font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for name, filename in (
            ("Body", "DejaVuSerif.ttf"),
            ("Body-Bold", "DejaVuSerif-Bold.ttf"),
            ("Body-Italic", "DejaVuSerif-Italic.ttf"),
            ("Mono", "DejaVuSansMono.ttf"),
            ("Math", "DejaVuSerif.ttf"),
        ):
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bold", italic="Body-Italic"
    )


register_fonts()
STYLES = getSampleStyleSheet()
for name, options in (
    ("TextRu", dict(fontSize=10.6, leading=13.2, spaceAfter=8, alignment=TA_JUSTIFY)),
    ("TitleRu", dict(fontSize=16.5, leading=21, alignment=TA_CENTER, spaceAfter=9)),
    (
        "SubtitleRu",
        dict(fontSize=11.2, leading=14.4, alignment=TA_CENTER, spaceAfter=12),
    ),
    (
        "HeadingRu",
        dict(
            fontName="Body-Bold",
            fontSize=12.4,
            leading=16,
            spaceBefore=10,
            spaceAfter=9,
            keepWithNext=True,
        ),
    ),
    (
        "SubheadingRu",
        dict(
            fontName="Body-Bold",
            fontSize=11.2,
            leading=14,
            spaceBefore=9,
            spaceAfter=8,
            keepWithNext=True,
        ),
    ),
    (
        "BulletRu",
        dict(
            fontSize=10.6,
            leading=13.2,
            leftIndent=16,
            firstLineIndent=0,
            bulletIndent=5,
            spaceAfter=5,
        ),
    ),
    ("CellRu", dict(fontSize=10.1, leading=12.5)),
    (
        "CaptionRu",
        dict(
            fontSize=9.8,
            leading=12.5,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=10,
        ),
    ),
    ("SmallRu", dict(fontSize=9.4, leading=12, spaceAfter=7)),
    (
        "EquationRu",
        dict(
            fontSize=11.2, leading=16, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10
        ),
    ),
    ("CodeRu", dict(fontName="Mono", fontSize=8.6, leading=12.5, spaceAfter=7)),
):
    STYLES.add(
        ParagraphStyle(
            name, **{"fontName": "Body", "textColor": colors.black, **options}
        )
    )


def inline_math(text: str) -> str:
    """Use a matching serif fallback for symbols absent from PT Serif."""
    return text.translate(
        {ord(char): f'<font name="Math">{char}</font>' for char in "ηλ→"}
    )


def p(text: str, style: str = "TextRu") -> Paragraph:
    return Paragraph(inline_math(text), STYLES[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(inline_math(text), STYLES["BulletRu"], bulletText="•")


def number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def accuracy(analysis: dict, experiment: str) -> str:
    stats = analysis["groups"][experiment]["final_accuracy_percent"]
    result = number(stats["mean"])
    if stats["n"] > 1:
        result += " ± " + number(stats["sample_sd"])
    return result + "%"


def grid(rows: list[list[str]], widths: list[float]) -> Table:
    cells = [[p(text, "CellRu") for text in row] for row in rows]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
            ]
        )
    )
    table.spaceAfter = 10
    return table


def figure(name: str, caption: str, ratio: float = 4.5 / 11) -> KeepTogether:
    return KeepTogether(
        [
            Spacer(1, 5),
            Image(
                str(ROOT / "results" / "figures" / f"{name}.png"),
                width=WIDTH,
                height=WIDTH * ratio,
            ),
            p(caption, "CaptionRu"),
        ]
    )


def opening(analysis: dict) -> list[Flowable]:
    return [
        Spacer(1, 20),
        p("Отчёт о воспроизведении: «Super-Convergence»", "TitleRu"),
        p("Smith, Topin, 2018 · LeNet / MNIST и ResNet-56 / CIFAR-10", "SubtitleRu"),
        p("Коротко", "HeadingRu"),
        bullet(
            "<b>Основной эффект повторился.</b> На CIFAR-10 получено 91,95% за 10 000 обновлений против 90,54% за 80 000 у baseline. Циклический режим оказался лучше на 1,41 п. п. при восьмикратно меньшем числе шагов; в статье - 92,4% против пиковых 91,2%."
        ),
        bullet(
            f"<b>На MNIST результат такой же по направлению.</b> 1cycle дал {accuracy(analysis, 'mnist_cycle')} против {accuracy(analysis, 'mnist_baseline')} у длинного baseline. Короткий baseline с теми же 1407 шагами получил {accuracy(analysis, 'mnist_short')}. Средние и выборочные SD рассчитаны по трём сидам."
        ),
        bullet(
            "<b>Время сократилось вместе с бюджетом.</b> Обучение ResNet-56 заняло 12,61 мин вместо 100,85 мин. Отношение 8,00× относится к полным протоколам, а не к скорости одного шага или достижению одинаковой точности."
        ),
        bullet(
            "<b>Числа статьи не совпали точно.</b> На CIFAR-10 проведён один запуск на режим, на MNIST - три. Реализация перенесена из Caffe в PyTorch с сохранением архитектуры, формы SGD и BN по группам; недостающий конечный LR для MNIST явно задан равным 10<super>−5</super>."
        ),
        p("1. Что проверяется", "HeadingRu"),
        p(
            "В статье предлагается сначала повышать learning rate до большого значения, затем снижать его. Выбраны два сравнения с полными бюджетами обучения. Нумерация рисунков и таблиц ниже относится к arXiv:1708.07120v3 [1]."
        ),
        grid(
            [
                ["№", "Утверждение", "Где в статье", "Значение в статье"],
                [
                    "У1",
                    "ResNet-56 можно обучить за меньшее число шагов без потери точности",
                    "Рис. 1a; введение",
                    "10 000 шагов: 92,4%; baseline около 80 000: пик 91,2%",
                ],
                [
                    "У2",
                    "На LeNet/MNIST 1cycle даёт лучшую точность за 12 эпох вместо 85",
                    "Табл. 2",
                    "99,25 ± 0,03% против 99,03 ± 0,04%",
                ],
            ],
            [25, 233, 85, 157],
        ),
        p(
            "<b>Дополнительные вопросы.</b> В1: что даёт 1cycle на MNIST при одинаковом числе обновлений? В2: сокращается ли время до первой записанной точности 90% на CIFAR-10? Они отделяют итог длинного обучения от сравнения при равном бюджете и общей целевой метрике."
        ),
    ]


def method() -> list[Flowable]:
    return [
        p("2. Как устроен метод", "HeadingRu"),
        bullet(
            "<b>Learning rate (LR)</b> - множитель при обновлении весов. В простейшем SGD: w<sub>t+1</sub> = w<sub>t</sub> − η<sub>t</sub>g<sub>t</sub>, где g - градиент потерь, η - LR. Маленький шаг замедляет обучение, слишком большой может сделать его неустойчивым."
        ),
        bullet(
            "<b>CLR</b> - циклический LR: сначала линейный рост, затем линейное снижение. В выбранном опыте CIFAR-10 один цикл занимает всё обучение: 0,1 → 3 → 0,1 за 10 000 обновлений."
        ),
        bullet(
            "<b>1cycle</b> - один цикл с дополнительным снижением LR ниже начального значения в конце. На MNIST рост и спад занимают по пять эпох, заключительный участок - две эпохи. Одновременно momentum меняется в обратную сторону."
        ),
        bullet(
            "<b>Momentum</b> сохраняет часть предыдущего обновления, а <b>weight decay</b> добавляет к градиенту штраф за величину весов. <b>BatchNorm (BN)</b> нормализует активации по статистикам группы примеров и накапливает статистики для тестирования."
        ),
        p(
            "<b>Почему это может работать.</b> Авторы объясняют эффект регуляризацией от большого LR: такая динамика мешает слишком точно подстраиваться под train set. LR нужно согласовывать с weight decay и другими источниками регуляризации; финальное снижение шага позволяет уточнить решение [1, раздел 3 и приложение A.2]."
        ),
        p(
            "В LR range test авторы постепенно повышают LR и проверяют, где сохраняется качество. Этот тест не повторялся, диапазон не подбирался: использованы опубликованные настройки. Причинную роль большого LR эти сравнения не изолируют."
        ),
        figure(
            "schedules",
            "Рис. 1. Реализованные расписания LR. Для CIFAR-10 использован треугольный CLR из авторской конфигурации; для MNIST - цикл с заключительным спадом.",
            6.5 / 11,
        ),
        p(
            "<b>Шаг и эпоха.</b> Шаг - обновление весов по батчу; эквивалент эпох = шаги × batch / размер train. Для CIFAR-10: 10 000 шагов × 1000 / 50 000 = 200 эпох."
        ),
    ]


def protocol() -> list[Flowable]:
    return [
        p("3. Постановка", "HeadingRu"),
        p("3.1. Данные, модели и расписания", "SubheadingRu"),
        p(
            "Настройки CIFAR-10 сверены с архитектурой и solver-файлами репозитория авторов [2], MNIST - с таблицей 2 статьи и примером LeNet из Caffe [3]. Все обучающие и тестовые примеры сохранены; сокращения выборок и бюджетов не было."
        ),
        grid(
            [
                ["Компонент", "MNIST", "CIFAR-10"],
                [
                    "Train / test",
                    "60 000 / 10 000 изображений",
                    "50 000 / 10 000 изображений",
                ],
                [
                    "Модель",
                    "Caffe LeNet; 431 080 параметров",
                    "Авторская ResNet-56; 808 682 параметра",
                ],
                ["Batch", "512", "1000; BN по 8 группам × 125"],
                [
                    "Baseline",
                    "10 000 шагов ≈ 85,33 эпохи; LR 0,01 · (1 + 10<super>−4</super>t)<super>−0,75</super>",
                    "80 000 шагов = 1600 эпох; LR 0,35, деление на 10 после 50 000 и 70 000 шагов",
                ],
                [
                    "Быстрый режим",
                    "1407 шагов ≈ 12,006 эпохи; LR 0,01 → 0,1 → 0,01 → 10<super>−5</super>",
                    "10 000 шагов = 200 эпох; LR 0,1 → 3 → 0,1; полупериод 5000 шагов",
                ],
                [
                    "Momentum",
                    "0,9 у baseline; 0,95 → 0,8 → 0,95 у 1cycle",
                    "0,9 у обоих режимов",
                ],
                ["Weight decay", "5 × 10<super>−4</super>", "10<super>−4</super>"],
                [
                    "BN moving-average fraction",
                    "BN отсутствует",
                    "0,999 у baseline; 0,95 у CLR",
                ],
                [
                    "Входы и аугментации",
                    "Пиксели / 256; без аугментаций",
                    "RGB 0…255 минус среднее train-изображение; случайное отражение слева направо после центрирования",
                ],
            ],
            [109, 187, 204],
        ),
        p(
            "<b>LeNet.</b> Две свёртки с max pooling, полносвязный слой на 500 нейронов и выход на 10 классов. ReLU стоит только после первого полносвязного слоя; добавлять её после свёрток означало бы изменить модель Caffe."
        ),
        p(
            "<b>ResNet-56.</b> В сети 27 остаточных блоков. Переходный блок уменьшает пространственный размер, а shortcut использует average pooling. Нулевые каналы дописываются после сложения ветвей и ReLU. Сохранён авторский вариант архитектуры. Дополнительное кадрирование с padding не добавлено."
        ),
        p("3.2. Что пришлось перенести из Caffe", "SubheadingRu"),
        p(
            "В оптимизаторе Caffe LR умножает градиент до помещения в буфер momentum. Это существенно при меняющемся LR. Для одной группы параметров обновление имеет вид:"
        ),
        p(
            "v<sub>t</sub> = μ<sub>t</sub>v<sub>t−1</sub> + η<sub>t</sub>(g<sub>t</sub> + λw<sub>t</sub>), &nbsp; w<sub>t+1</sub> = w<sub>t</sub> − v<sub>t</sub>.",
            "EquationRu",
        ),
        p(
            "Перенесены эта формула, инициализация и множители LR/weight decay для групп весов и смещений. Обычная замена на torch.optim.SGD без сверки формулы дала бы другое накопление прошлых обновлений."
        ),
    ]


def differences() -> list[Flowable]:
    return [
        p("3.3. Что совпадает по смыслу и что отличается", "SubheadingRu"),
        p(
            "<b>Восемь групп BN на одной GPU.</b> В авторском запуске общий batch 1000 распределялся по восьми GPU: нормализация считалась локально по 125 изображениям. Сохранены восемь таких групп на одном устройстве, общие обучаемые веса BN и статистики первой группы для inference. Накопленные статистики нормируются как в Caffe, включая начальное накопление массы."
        ),
        p(
            "CPU-тесты сопоставляют сгруппированную реализацию с восемью последовательными микробатчами: выходы, градиенты и статистики совпадают с точностью вычислений. Это проверка реализации групп, а не доказательство побитового совпадения с исходным многопроцессным Caffe."
        ),
        grid(
            [
                ["В оригинале", "В воспроизведении", "Что это означает"],
                [
                    "Caffe/CUDA 7, несколько GPU",
                    "PyTorch/CUDA 12.8, одна GPU; размер группы BN сохранён",
                    "Численные ядра и генераторы случайных чисел различаются",
                ],
                [
                    "Конечный LR для хвоста MNIST не уточнён",
                    "Линейное снижение до 10<super>−5</super>",
                    "Явное допущение; точное расписание авторов восстановить нельзя",
                ],
                [
                    "В solver CIFAR задано 25 000 тестовых примеров с повторениями",
                    "Один проход по всем 10 000 тестовым изображениям",
                    "Иная процедура усреднения test accuracy",
                ],
                [
                    "Табл. 5: stride 2 первой свёртки; в коде: stride 1",
                    "По рабочему коду: stride 1",
                    "Приоритет авторской реализации при расхождении источников",
                ],
                [
                    "Несколько задач и архитектур",
                    "Два выбранных сравнения; 3 сида MNIST и 1 сид CIFAR-10",
                    "Вывод ограничен этой постановкой; разброс CIFAR не оценён",
                ],
            ],
            [151, 182, 167],
        ),
        p("3.4. Запуски, метрики и окружение", "SubheadingRu"),
        p(
            "На MNIST выполнены три режима с seeds 42, 43, 44: длинный baseline, 1cycle и короткий baseline на 1407 шагов с тем же inverse-LR, что у длинного. Внутри каждого сида начальные веса у режимов одинаковы. На CIFAR-10 выполнено по одному основному запуску с seed 42. Итого 11 обучений; ещё два пилота CIFAR по 200 шагов хранятся отдельно и в основные кривые не входят."
        ),
        p(
            "<b>Метрика</b> - accuracy на полном test set после последнего обновления, без выбора лучшего checkpoint. Промежуточные оценки CIFAR записаны каждые 100 шагов, MNIST - каждые 118 шагов и в конце. Для MNIST указаны среднее и выборочное SD между запусками (ddof=1), а не стандартная ошибка среднего."
        ),
        p(
            "<b>Оборудование:</b> NVIDIA RTX PRO 6000 Blackwell Server Edition, 94,97 GiB доступной VRAM. Python 3.13.15, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, CUDA 12.8, cuDNN 9.10.2, драйвер 580.82.07. Все вычисления FP32; AMP и TF32 отключены. Данные обходятся последовательно с переходом к началу, без случайного перемешивания."
        ),
    ]


def cifar_results(analysis: dict) -> list[Flowable]:
    return [
        p("4. Результаты", "HeadingRu"),
        p("4.1. ResNet-56 на CIFAR-10: главное сравнение", "SubheadingRu"),
        p(
            "<b>Постановка:</b> полный train set, batch 1000, 80 000 шагов baseline и 10 000 шагов CLR. Настройки из раздела 3, один запуск каждого режима с seed 42. Основная метрика - финальная точность на 10 000 тестовых изображений."
        ),
        grid(
            [
                ["Показатель", "Baseline", "CLR"],
                ["Шагов / эпох", "80 000 / 1600", "10 000 / 200"],
                [
                    "Финальная точность",
                    f"<b>{accuracy(analysis, 'cifar_baseline')}</b>",
                    f"<b>{accuracy(analysis, 'cifar_cycle')}</b>",
                ],
                [
                    "Лучшая записанная точность",
                    "90,63%; шаг 73 400",
                    "91,95%; шаг 10 000",
                ],
                ["Точность при 10 000 шагах", "53,43%", "91,95%"],
                [
                    "Финальный train / test loss",
                    "0,00037 / 0,38883",
                    "0,00057 / 0,31676",
                ],
            ],
            [219, 140, 141],
        ),
        figure(
            "cifar_accuracy",
            "Рис. 2. Полные кривые и первые 10 000 шагов крупнее. Отрезки соединяют записанные оценки без сглаживания.",
        ),
        bullet(
            "<b>У1 воспроизвелось качественно.</b> CLR завершил обучение с преимуществом 1,41 п. п. и в восемь раз меньшим числом обновлений. При одинаковых 10 000 шагах разница составляет 38,52 п. п., но длинный baseline на этом шаге ещё не дошёл до своего первого снижения LR."
        ),
        bullet(
            "<b>Точность ниже опубликованной.</b> Авторы указывают пик baseline 91,2%, поэтому сравнивать его нужно с полученным пиком 90,63%: разница −0,57 п. п. CLR дал 91,95% против авторских 92,4%: −0,45 п. п. Финальные 90,54% baseline не подменяют его лучший результат."
        ),
        bullet(
            "<b>Разница относится к двум протоколам целиком.</b> Помимо LR меняется moving-average fraction BN: 0,999 против 0,95. Поэтому большой разрыв в начале кривых нельзя объяснить только расписанием шага. Подбор оптимального baseline для 10 000 обновлений и отдельные абляции BN не проводились."
        ),
        p(
            "Оба режима почти обнулили тренировочную потерю, но CLR имеет меньшую тестовую потерю. Это согласуется с направлением авторского объяснения, однако одного такого сравнения недостаточно, чтобы отдельно доказать механизм регуляризации большим LR."
        ),
    ]


def mnist_results(analysis: dict) -> list[Flowable]:
    runs = {(run["experiment"], run["seed"]): run for run in analysis["runs"]}
    rows = [["Режим", "Seed 42", "Seed 43", "Seed 44", "Среднее ± SD"]]
    for name, label in (
        ("mnist_baseline", "Baseline, 10 000"),
        ("mnist_cycle", "1cycle, 1407"),
        ("mnist_short", "Baseline, 1407"),
    ):
        rows.append(
            [
                label,
                *[
                    number(runs[name, seed]["final_accuracy_percent"])
                    for seed in (42, 43, 44)
                ],
                accuracy(analysis, name),
            ]
        )
    return [
        p("4.2. LeNet на MNIST: длинный и короткий baseline", "SubheadingRu"),
        p(
            "<b>Постановка:</b> batch 512, seeds 42, 43, 44. Длинный baseline содержит 10 000 обновлений (85,33 эквивалентных эпохи), 1cycle и короткий baseline - по 1407 (12,006 эпохи). Короткий baseline - дополнительный контроль В1; соответствующей отдельной строки в таблице 2 статьи нет."
        ),
        grid(rows, [156, 65, 65, 65, 149]),
        p(
            "Все числа - проценты финальной точности. SD рассчитано по трём независимым сидам; в пределах одного сида сравнение парное по начальным весам.",
            "SmallRu",
        ),
        figure(
            "mnist_accuracy",
            "Рис. 3. Средние кривые по трём сидам и разброс между ними. Справа показан одинаковый начальный бюджет до 12 эпох.",
        ),
        bullet(
            "<b>У2 воспроизвелось по направлению.</b> 1cycle лучше длинного baseline во всех трёх парных повторах; средняя разница +0,107 п. п. При этом шагов в 7,11 раза меньше. Такой результат поддерживает полезность режима на этой задаче, но не гарантирует выигрыш для любой инициализации."
        ),
        bullet(
            "<b>Ответ на В1: при равном бюджете 1cycle тоже лучше.</b> Преимущество над коротким baseline составляет 0,500 п. п. Это показывает, что результат не сводится к простому раннему прекращению обычного обучения. Одновременно с LR у 1cycle меняется momentum; их отдельные вклады не измерены."
        ),
        grid(
            [
                [
                    "Финальная точность",
                    "Статья, табл. 2",
                    "Воспроизведение",
                    "Разница средних",
                ],
                [
                    "Baseline",
                    "99,03 ± 0,04%",
                    accuracy(analysis, "mnist_baseline"),
                    "−0,010 п. п.",
                ],
                [
                    "1cycle",
                    "99,25 ± 0,03%",
                    accuracy(analysis, "mnist_cycle"),
                    "−0,123 п. п.",
                ],
            ],
            [111, 140, 140, 109],
        ),
    ]


def cost(analysis: dict) -> list[Flowable]:
    groups = analysis["groups"]
    cifar_base = groups["cifar_baseline"]["training_seconds"]["mean"] / 60
    cifar_cycle = groups["cifar_cycle"]["training_seconds"]["mean"] / 60
    mnist_base = groups["mnist_baseline"]["training_seconds"]["mean"]
    mnist_cycle = groups["mnist_cycle"]["training_seconds"]["mean"]
    return [
        p("5. Сколько это стоило по времени", "HeadingRu"),
        p(
            "<b>Постановка:</b> та же GPU и те же запуски. Обучение и оценка измерены отдельно через perf_counter с синхронизацией CUDA на границах интервалов. Счётчик обучения не включает скачивание и загрузку данных, оценку, запись журналов и checkpoints."
        ),
        figure(
            "training_time",
            "Рис. 4. Время обучения: полные бюджеты, один сид CIFAR-10, среднее трёх сидов MNIST.",
            4 / 10,
        ),
        grid(
            [
                ["Сравнение", "Baseline", "Циклический режим", "Отношение"],
                [
                    "CIFAR-10: обучение",
                    f"{number(cifar_base)} мин",
                    f"{number(cifar_cycle)} мин",
                    "8,00×",
                ],
                ["CIFAR-10: только оценки", "2,67 мин", "0,36 мин", "-"],
                [
                    "CIFAR-10: по UTC*",
                    "103,81 мин",
                    "13,01 мин",
                    "-",
                ],
                [
                    "MNIST: обучение, среднее",
                    f"{number(mnist_base)} с",
                    f"{number(mnist_cycle)} с",
                    "5,92×",
                ],
            ],
            [204, 93, 119, 84],
        ),
        bullet(
            "<b>На CIFAR-10 экономия почти пропорциональна числу шагов.</b> Средний шаг занимает 75,6 мс у baseline и 75,7 мс у CLR. 8,00× - отношение длительностей обучения на 80 000 и 10 000 шагах; сам вычислительный шаг быстрее не стал."
        ),
        bullet(
            "<b>На MNIST очень короткие запуски.</b> Записанные синхронизированные интервалы согласуются с журналами, но это не отдельный аппаратный бенчмарк. Времена не обязаны точно повторять отношение шагов 7,11×."
        ),
        p(
            "* По UTC: от старта запуска до последней оценки, без предварительной загрузки данных и установки окружения.",
            "SmallRu",
        ),
        p("5.1. В2: время до общей точности 90%", "SubheadingRu"),
        grid(
            [
                ["Первое записанное accuracy ≥ 90%", "Baseline", "CLR"],
                ["Шаг; время обучения", "70 900; 89,38 мин", "9600; 12,11 мин"],
            ],
            [278, 111, 111],
        ),
        p(
            "<b>Ответ на В2: да, отношение времён 7,38×.</b> Это первые наблюдаемые пересечения, без интерполяции. Проверки шли каждые 100 шагов; точный момент неизвестен. Более ранние краткие пересечения могли остаться незаписанными."
        ),
    ]


def conclusions() -> list[Flowable]:
    return [
        p("6. Что воспроизвелось, а что нет", "HeadingRu"),
        grid(
            [
                ["№", "Проверка", "Итог", "Основание"],
                [
                    "У1",
                    "ResNet-56: лучшее качество за меньшее число обновлений",
                    "<b>Да, качественно</b>",
                    "91,95% за 10 000 шагов против финальных 90,54% за 80 000; один сид",
                ],
                [
                    "У2",
                    "LeNet: 1cycle лучше длинного baseline",
                    "<b>Да, в трёх повторах</b>",
                    "99,13 ± 0,04% против 99,02 ± 0,05%; шагов в 7,11 раза меньше",
                ],
                [
                    "В1",
                    "MNIST при одинаковых 1407 шагах",
                    "<b>1cycle лучше</b>",
                    "99,13% против 98,63%; +0,500 п. п. по средним",
                ],
                [
                    "В2",
                    "CIFAR: время до первой записанной точности 90%",
                    "<b>CLR быстрее</b>",
                    "12,11 против 89,38 мин обучения; отношение 7,38×",
                ],
            ],
            [25, 176, 122, 177],
        ),
        p(
            "Главный практический вывод для выбранных задач повторился: циклические режимы дали более высокую финальную точность при существенно меньшем числе обновлений."
        ),
        p("6.1. Что можно проверить по приложенным результатам", "SubheadingRu"),
        p(
            "В каталоге results/ сохранены конфигурации, CSV, текстовые журналы и итоговые сводки; исходный код находится в hw2/. Проверка подтвердила полные сетки шагов, соответствие LR и momentum расписаниям, совпадение summary с последней строкой CSV и хешей четырёх файлов обучения с переданными исходниками."
        ),
    ]


def references() -> list[Flowable]:
    return [
        p("Ссылки", "HeadingRu"),
        p(
            "1. Smith L. N., Topin N. <i>Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.</i> arXiv:1708.07120v3, 17 мая 2018. <link href='https://arxiv.org/pdf/1708.07120'>arxiv.org/pdf/1708.07120</link>",
            "SmallRu",
        ),
        p(
            "2. Авторские архитектуры и конфигурации: <link href='https://github.com/lnsmith54/super-convergence'>github.com/lnsmith54/super-convergence</link>. Использованный commit: ecc561594b39442661e7e0a39d2820b0f4147324.",
            "SmallRu",
        ),
        p(
            "3. Caffe: пример MNIST/LeNet, SGD и BatchNorm. <link href='https://github.com/BVLC/caffe'>github.com/BVLC/caffe</link>.",
            "SmallRu",
        ),
        p(
            "4. PyTorch 2.10, ограничения воспроизводимости между платформами. <link href='https://docs.pytorch.org/docs/2.10/notes/randomness.html'>docs.pytorch.org/docs/2.10/notes/randomness.html</link>.",
            "SmallRu",
        ),
        p(
            "5. Запуски от 10.10.2026: results/*/config.json, metrics.csv, training.log и summary.json. Сводные расчёты - results/analysis.json и results/summary.csv.",
            "SmallRu",
        ),
    ]


def page_footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Body", 10)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(letter[0] / 2, 26, str(doc.page))
    canvas.restoreState()


def make_report() -> None:
    analysis = json.loads((ROOT / "results" / "analysis.json").read_text())
    pages = [
        opening(analysis),
        method(),
        protocol(),
        differences(),
        cifar_results(analysis),
        mnist_results(analysis),
        cost(analysis),
        conclusions() + references(),
    ]
    story = []
    for index, page in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.extend(page)
    document = SimpleDocTemplate(
        str(ROOT / "report.pdf"),
        pagesize=letter,
        leftMargin=56,
        rightMargin=56,
        topMargin=50,
        bottomMargin=45,
        title="Отчёт о воспроизведении: «Super-Convergence»",
        pageCompression=1,
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


if __name__ == "__main__":
    make_report()
