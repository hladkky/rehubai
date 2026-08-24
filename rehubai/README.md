# `rehubai` — universal exercise-validation core

Реалізація універсального, конфіг-керованого алгоритму валідації реабілітаційних
вправ (формалізм `F: V×C→R`, `MainAlgorithm.md`). Одна вправа = один JSON-конфіг,
без перенавчання нейромережі під кожну вправу. Валідатор одночасно перевіряє три
класи відхилень: **геометричні** (кути), **послідовнісні** (порядок етапів),
**динамічні** (темп переходів).

## Модулі
| Файл | Призначення |
|---|---|
| `joints.py` | семантичні імена суглобів + мапи схем (`blazepose33`/`kinect25`/`coco17`) |
| `angles.py` | обчислення кутів між трійками точок (2D/3D, confidence-aware) |
| `config.py` | `ExerciseConfig` — парсинг/валідація JSON-конфіга вправи |
| `validator.py` | `ExerciseValidator` — скінченний автомат + 3 класи перевірок + звіт |
| `quality.py` | функціонал якості `Q` |
| `feedback.py` | інтерпретація порушень → пріоритезований фідбек |
| `backends/` | обгортки детекторів (`mediapipe`) зі спільним `.detect(frame)` |
| `eval/` | `run_video.py` (end-to-end на відео), далі — датасет-бенчмарк |

## Швидкий старт
```python
from rehubai import ExerciseConfig, ExerciseValidator
cfg = ExerciseConfig.load("configs/exercises/dead_bug.json")
val = ExerciseValidator(cfg, scheme="blazepose33")   # той самий конфіг → coco17/kinect25
res = val.process_frame(keypoints, confidences, frame_idx=0)
report = val.generate_report()
```

End-to-end на власному відео:
```bash
python -m rehubai.eval.run_video \
  --video sources/videos/dead_bug.MOV \
  --config configs/exercises/dead_bug.json \
  --backend mediapipe --frame-step 2
```

## Тести
```bash
python -m pytest tests/    # 43 юніт-тести логіки валідації
```

## Ключова властивість (новизна)
Конфіг посилається на суглоби **семантично** (`left_elbow`, …), тож один
`dead_bug.json` валідується на BlazePose-відео, на Kinect-скелетах KIMORE і на
COCO-виході RTMPose/YOLO — без зміни коду. Це основа для валідації на відкритих
датасетах (UI-PRMD, KIMORE) у наступних етапах.
