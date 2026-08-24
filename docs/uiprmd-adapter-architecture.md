# Адаптер відкритого датасету UI-PRMD: архітектура, алгоритми та обґрунтування

> Технічний опис модуля `rehubai/datasets/uiprmd.py` для Розділу 3 дисертації
> («Програмна реалізація ІІТ») та для міжнародної публікації.
> Створено 2026-08-24. Мова базового тексту — українська (дисертація);
> §12 містить англомовний abstract і глосарій термінів для статті.
> Пов'язане: [`dissertation-plan-and-progress.md`](dissertation-plan-and-progress.md),
> [`algorithm.md`](algorithm.md) (формалізм валідатора),
> [`3d-pose-rationale.md`](3d-pose-rationale.md) (3D-координати).

---

## 1. Роль адаптера в архітектурі ІІТ

Система розділена на три незалежні конвеєри (вимога `CLAUDE.md`): **детекція
поз → валідація → зворотний зв'язок**. Ядро валідації (`ExerciseValidator`)
свідомо **не залежить** від джерела скелета: воно споживає уніфіковане
представлення пози й описану декларативним конфігом модель вправи
`E = (S, Θ, T, G)`. Адаптер датасету — це компонент **першого** конвеєра, який
приводить дані конкретного відкритого датасету до цього уніфікованого
представлення.

Адаптер UI-PRMD обслуговує два наскрізні тракти доказовості:

```
        RGB-відео  ──►  MediaPipe (world-landmarks)  ─┐
 (власні записи)                                       │   абсолютні 3D-точки
                                                       ├─► семантична схема ─► ExerciseValidator ─► метрики
        UI-PRMD    ──►  адаптер: FK-реконструкція  ───┘        (joints.py)          (F, Q)
      (скелет, без                (uiprmd.py)
        відео)
```

Ключова теза: **обидва входи зводяться до однакового представлення** («абсолютні
3D-точки під семантичною схемою суглобів»), тому **один декларативний конфіг
вправи валідує і власне відео, і скелети UI-PRMD без перенавчання**. Це прямий
емпіричний доказ головної наукової новизни — універсального конфіг-простору.

---

## 2. Джерело даних: структура UI-PRMD

UI-PRMD (Vakanski et al., 2018; ліцензія PDDL 1.0) — це **суто кінематичний**
датасет **без RGB-відео**:

| Параметр | Значення |
|---|---|
| Вправи | 10 (`m01`–`m10`): deep squat, hurdle step, inline lunge, side lunge, sit-to-stand, straight leg raise, shoulder abduction/extension/rotation/scaption |
| Суб'єкти | 10 здорових (`s01`–`s10`) |
| Виконання | **correct** та **incorrect** (навмисна імітація «непацієнтського» руху; суфікс `_inc`) |
| Системи захоплення | **Vicon** (39 маркерів) і **Kinect** (22 суглоби) |
| Виміри | **Positions** (координати) і **Angles** (кути) |
| Сегментація | повні сесії + окремі повтори (епізоди `e01`–`e10`) |

Файлова організація (корінь = тека з підтеками нижче):

```
Movements/                     Incorrect Movements/
Segmented Movements/           Incorrect Segmented Movements/
   └─ {Kinect,Vicon}/{Positions,Angles}/  mXX_sXX[_eXX]_(positions|angles)[_inc].txt
```

Фактичні обсяги (Kinect, файли Positions): 100 повних correct + 100 incorrect;
**1000 сегментованих correct + 1000 incorrect** повторів — саме вони є одиницею
класифікації. Мітки суб'єкта / вправи / коректності закодовані в **імені файлу**.

---

## 3. Проблеми репрезентації, які вирішує адаптер

Це центральна частина: наївне зчитання UI-PRMD дає **неправильні кути** й
несумісні зі схемою скелети. Адаптер закриває сім нетривіальних проблем.

| # | Проблема | Наслідок, якщо проігнорувати | Рішення в адаптері |
|---|---|---|---|
| **P1** | **Kinect-Positions зберігаються в ієрархічних (parent-relative) координатах**, а не абсолютних (абсолютний лише корінь) | Кути між суглобами обчислюються з точок у різних локальних системах → сміття | Пряма кінематика `forward_kinematics()` (§5) |
| **P2** | Схема скелета Kinect-**22** ≠ BlazePose-**33** | Індекси суглобів не збігаються з конфігом | Семантична абстракція + реєстрація схеми `uiprmd_kinect22` (§6) |
| **P3** | Датасет **без RGB-відео** | Неможливо прогнати MediaPipe; здається несумісним із відеотрактом | Подача скелета напряму у валідатор; RGB-нішу закриває власна зйомка (§1) |
| **P4** | **Різні роздільники**: пробіли в `Movements/`, коми в `Segmented Movements/` | Парсер падає на половині архіву | Робастний `load_matrix()` (нормалізує роздільник) |
| **P5** | Метадані **закодовані в іменах файлів** | Немає як згрупувати за суб'єктом/вправою | Регулярний парсер `parse_filename()` → `(вправа, суб'єкт, епізод, correct)` |
| **P6** | Потрібен **cross-subject** протокол (вимога рецензентів) | In-subject оцінка завищена й непереконлива | `loso_splits()` — leave-one-subject-out (§7) |
| **P7** | **Прогалини в даних** (одна Vicon-сесія відсутня: 99/990 замість 100/1000) | Аварія або зсув індексів | М'яка обробка: пропуск відсутніх пар positions/angles |

Додатково знято «пастку провенансу»: **Kinect має 22, а не 25 суглобів**
(66 стовпців = 22×3), тож наявна схема `kinect25` не підходить — потрібна окрема
`uiprmd_kinect22`.

---

## 4. Архітектура модуля

Модуль `rehubai/datasets/uiprmd.py` складається з чотирьох шарів:

```
┌──────────────────────────────────────────────────────────────────┐
│ (D) Індексація та протокол                                         │
│     UIPRMDDataset: samples() · by_subject() · loso_splits()        │
├──────────────────────────────────────────────────────────────────┤
│ (C) Модель зразка                                                  │
│     Sample(movement, subject, episode, correct, ...) · keypoints() │
├──────────────────────────────────────────────────────────────────┤
│ (B) Введення-виведення                                            │
│     load_matrix() (робастний) · parse_filename() (regex)          │
├──────────────────────────────────────────────────────────────────┤
│ (A) Геометричне ядро                                              │
│     forward_kinematics() · euler_to_matrix() · схема uiprmd_kinect22│
└──────────────────────────────────────────────────────────────────┘
```

**Потік даних для одного зразка:**

```
Sample.keypoints()
  ├─ load_matrix(pos_path) ─► offsets (T, 22, 3)   # ієрархічні зміщення
  ├─ load_matrix(ang_path) ─► angles  (T, 22, 3)   # Euler-кути (градуси)
  └─ forward_kinematics(offsets, angles) ─► world (T, 22, 3)   # абсолютні 3D-точки
                                                    ▼
                     ExerciseValidator(cfg, scheme="uiprmd_kinect22")
                     .process_frame(world[t], confidences=None)   # чистий mocap → без гейта видимості
```

---

## 5. Ключовий алгоритм: реконструкція абсолютних координат прямою кінематикою (FK)

### 5.1 Постановка (проблема P1)

У Kinect-файлах зберігається **ієрархічний** скелет: рядок кадру — це 22 трійки,
де перша (Waist) — **абсолютна** позиція кореня, а решта — **зміщення кістки в
локальній системі батьківського суглоба**. Емпіричне підтвердження: у нейтральній
позі Spine = (0, 27.8, 0), L_Forearm = (25.0, 0, 0) — це довжини кісток уздовж
локальних осей, а не світові координати. Отже, безпосередньо обчислити кут
плече–лікоть–зап'ясток **не можна** — точки лежать у різних системах відліку.

### 5.2 Формалізація

Скелет — кореневе дерево з 22 вузлів; корінь — суглоб 0 (Waist), для решти
визначено батька `p(j)`. Для кожного кадру вхід:

- **зміщення** `oⱼ ∈ ℝ³` (з файлу Positions; для кореня `o₀` — абсолютна позиція);
- **Euler-кути** `αⱼ = (αⱼˣ, αⱼʸ, αⱼᶻ)` у градусах (з файлу Angles).

Елементарні обертання `Rₓ, Rᵧ, R_z` — стандартні. Локальне обертання суглоба
(конвенція YXZ, як у референсній реалізації Vakanski):

```
R(α) = R_z(αᶻ) · Rᵧ(αʸ) · Rₓ(αˣ)
```

Абсолютні позиції `gⱼ` обчислюються обходом **п'яти кінематичних ланцюгів**.
Кожен ланцюг задається парою `(s, (c₁,…,c_k))`, де `s` — «затравочний» суглоб,
чий Euler-кут ініціалізує обертання ланцюга:

| Ланцюг | Затравка `s` | Суглоби `(c₁…c_k)` |
|---|---|---|
| Хребет | 0 (Waist) | 1,2,3,4,5 |
| Ліва рука | 2 (Chest) | 6,7,8,9 |
| Права рука | 2 (Chest) | 10,11,12,13 |
| Ліва нога | 0 (Waist) | 14,15,16,17 |
| Права нога | 0 (Waist) | 18,19,20,21 |

Обчислення для ланцюга `(s, (c₁,…,c_k))`, з `g₀ = o₀`:

```
Q ← R(α_s)
g_{c₁} ← Q · o_{c₁} + g_{p(c₁)}
для i = 2..k:
    Q ← Q · R(α_{cᵢ − 1})          # накопичення обертанням попереднього суглоба ланцюга
    g_{cᵢ} ← Q · o_{cᵢ} + g_{p(cᵢ)}
```

### 5.3 Псевдокод

```
FUNCTION forward_kinematics(offsets[T,22,3], angles_deg[T,22,3]) -> world[T,22,3]:
    для кожного кадру t:
        g ← copy(offsets[t])                     # g[0] = абсолютний корінь
        a ← deg2rad(angles_deg[t])
        для кожного (s, chain) у CHAINS:
            Q ← euler_to_matrix(a[s])
            для i, j у enumerate(chain):
                якщо i > 0: Q ← Q · euler_to_matrix(a[j − 1])
                g[j] ← Q · offsets[t][j] + g[PARENT[j]]
        world[t] ← g
    RETURN world
```

### 5.4 Інваріант коректності

Оскільки обертання ортонормовані, `‖gⱼ − g_{p(j)}‖ = ‖oⱼ‖` для кожного кадру —
тобто **довжина кістки стала в часі** (жорсткий скелет). Це необхідна умова
коректності FK і використовується як тест (§8).

---

## 6. Семантична абстракція суглобів (проблема P2 — важіль універсальності)

Конфіги вправ посилаються на суглоби **семантичними іменами** (`left_elbow`, …),
а не індексами. Модуль `rehubai/joints.py` тримає мапи «схема → індекс». Адаптер
**реєструє** схему `uiprmd_kinect22` під час імпорту (`register_scheme`), після
чого той самий конфіг резолвиться і на BlazePose-33, і на UI-PRMD-скелет.

Особливість схеми Kinect: суглоб-**сегмент** несе позицію суглоба на своєму
**проксимальному** кінці (напр. «L_UpperArm» = лівий **плечовий** суглоб,
«L_Forearm» = лівий **лікоть**). Повна мапа 22 суглобів:

| Індекс | Сира назва | Семантичне ім'я | Індекс | Сира назва | Семантичне ім'я |
|---:|---|---|---:|---|---|
| 0 | Waist | `spine_base` | 11 | R_UpperArm | `right_shoulder` |
| 1 | Spine | `spine_mid` | 12 | R_Forearm | `right_elbow` |
| 2 | Chest | — | 13 | R_Hand | `right_wrist` |
| 3 | Neck | `neck` | 14 | L_UpperLeg | `left_hip` |
| 4 | Head | `head` | 15 | L_LowerLeg | `left_knee` |
| 5 | HeadTip | — | 16 | L_Foot | `left_ankle` |
| 6 | L_Collar | — | 17 | L_Toes | `left_foot` |
| 7 | L_UpperArm | `left_shoulder` | 18 | R_UpperLeg | `right_hip` |
| 8 | L_Forearm | `left_elbow` | 19 | R_LowerLeg | `right_knee` |
| 9 | L_Hand | `left_wrist` | 20 | R_Foot | `right_ankle` |
| 10 | R_Collar | — | 21 | R_Toes | `right_foot` |

Зіставлено 18 із 22 суглобів; чотири (Chest, HeadTip, обидва Collar) не мають
відповідника в семантичному словнику й лишаються незіставленими (не потрібні
для кутових перевірок реабілітаційних вправ).

**Вибір Kinect, а не Vicon.** Kinect дає 22 семантичні суглоби, що прямо
лягають на словник; Vicon — 39 **поверхневих маркерів**, для переведення яких у
центри суглобів потрібна біомеханічна модель (Plug-in-Gait) → винесено в майбутню
роботу.

**Вибір Positions + FK, а не готових Angles.** Файли Angles містять Euler-кути
**на суглоб відносно батька** — це не ті «внутрішні» кути (плече–лікоть–зап'ясток),
що перевіряє валідатор. Реконструкція абсолютних точок із Positions забезпечує
**ідентичність тракту** з MediaPipe (обидва → абсолютні точки → `calculate_angle`).

---

## 7. Індексація даних і протокол LOSO (проблеми P4–P7)

**Парсинг імені** (`parse_filename`, regex
`^m(\d+)_s(\d+)(?:_e(\d+))?_positions(_inc)?\.txt$`) повертає `movement`,
`subject`, `episode`, `correct`. Файл кутів отримується заміною
`positions → angles`; відсутні пари пропускаються (P7).

**Вибір теки** за `(correct, segmented)`:

| correct | segmented | тека |
|---|---|---|
| так | так | `Segmented Movements/` |
| так | ні | `Movements/` |
| ні | так | `Incorrect Segmented Movements/` |
| ні | ні | `Incorrect Movements/` |

**`UIPRMDDataset`** параметризується `root`, `system="kinect"`, `segmented`,
`movements`; надає `samples(correct=…)`, `subjects()`, `by_subject()` та
генератор крос-валідації:

```
FUNCTION loso_splits():                      # leave-one-subject-out
    groups ← by_subject()
    для held_out у sorted(groups):
        test  ← groups[held_out]
        train ← усі зразки суб'єктів ≠ held_out
        yield (held_out, train, test)
```

LOSO дає 10 фолдів (по одному на суб'єкта) — коректну оцінку **крос-суб'єктної
генералізації**, якої вимагають рецензенти й яку не забезпечує in-subject поділ.

**Довіра до точок.** UI-PRMD — чистий mocap без per-joint видимості, тому у
валідатор передається `confidences=None`; `confidence_ok` трактує це як «усі
точки видимі» (гейт оклюзій вимкнено). Для власного RGB, навпаки, подається
`visibility` з MediaPipe (див. [`3d-pose-rationale.md`](3d-pose-rationale.md)).

---

## 8. Верифікація коректності адаптера

Адаптер перевірено на трьох рівнях (усе відтворювано; `tests/test_uiprmd.py`,
12 тестів; сукупно 58 у наборі).

**(1) Юніт-рівень (без даних):** ортонормованість `euler_to_matrix`
(`RRᵀ=I`, `det=1`); сталість довжин кісток на синтетичній послідовності (§5.4);
коректність `parse_filename`; наявність усіх ключових семантичних суглобів у схемі.

**(2) Функціональна валідність FK (на даних):** реконструйовані кути анатомічно
осмислені —

| Вправа | Суглоб | Діапазон кута | Очікування |
|---|---|---|---|
| m01 deep squat | коліно | 173° → 10° | глибоке присідання ✔ |
| m05 sit-to-stand | коліно | 176° → 6° | ✔ |
| m06 leg raise | **праве** стегно | 167° → 94° | однонога вправа (ліва статична) ✔ |
| m07 shoulder abduction | **праве** плече | 22° → 152° | повне відведення ✔ |

Побічний висновок, зафіксований у документації адаптера: однокінцівкові вправи
(m06–m09) виконуються **правою** стороною.

**(3) Наскрізна інтеграція:** мінімальний декларативний конфіг присідання
проганяє UI-PRMD-скелет через `ExerciseValidator` за схемою `uiprmd_kinect22` —
детектуються стадії `STANDING↔SQUAT`, зараховується цикл, функціонал якості `Q`
розрізняє correct/incorrect у правильному напрямку. Це підтверджує контракт
«адаптер → валідатор» і головну тезу універсальності.

---

## 9. Відтворюваність

- **Розташування:** архів у `datasets/uiprmd/raw/` (тека з `Movements/`,
  `Segmented Movements/` тощо); `default_root()` резолвить його відносно репозиторію.
- **Оточення:** Python 3.9.13; numpy 1.26.4. FK детермінована, без випадковості.
- **Одиниці/система координат:** міліметри, **+y вгору**. Кути інваріантні до
  масштабу/системи, тож для геометричних перевірок це неважливо; позиційні
  перевірки (`y_above`/`y_below`) слід узгоджувати з цим фреймом окремо.
- **Приклад використання:**

```python
from rehubai.datasets import uiprmd
from rehubai.validator import ExerciseValidator

ds = uiprmd.UIPRMDDataset(segmented=True, movements=["m01"])
for held_out, train, test in ds.loso_splits():
    for s in test:                          # s: Sample
        kps = s.keypoints()                 # (T, 22, 3), абсолютні 3D
        v = ExerciseValidator(cfg, scheme=uiprmd.SCHEME)
        for frame in kps:
            v.process_frame(frame, confidences=None)
        report = v.generate_report()        # Q, цикли, порушення  ↔  s.label
```

---

## 10. Обмеження та майбутня робота

- **Vicon-тракт** (39 маркерів) поки не підтримано (`NotImplementedError`):
  потрібна біомеханічна модель маркери→суглоби.
- **Прогалина даних:** одна Vicon-сесія відсутня (99/990) — обробляється м'яко,
  але зменшує Vicon-вибірку.
- **Специфічність FK:** реконструкція відтворює референсну ланцюгову схему
  Vakanski (зокрема «скидання» обертання на початку кінцівкових ланцюгів);
  валідована функціонально (§8), але не є універсальною біомеханічною FK.
- **Наступні кроки (Місяць 2→3):** конфіги вправ m01/m05/m06/m07; адаптер KIMORE
  (кореляція `Q`↔клінічний бал); харнес метрик `eval/run_benchmark.py`
  (LOSO acc/F1/ROC-AUC + DTW-бейслайн).

---

## 11. Внесок (формулювання для тексту)

Адаптер UI-PRMD — це не просто «завантажувач даних», а компонент, що демонструє
**незалежність ядра валідації від джерела скелета**. Він (а) відновлює
геометрично коректні абсолютні координати з ієрархічного mocap-представлення
прямою кінематикою; (б) зводить чужу схему скелета (Kinect-22) до спільного
семантичного простору, завдяки чому один декларативний конфіг вправи валідує і
клінічний mocap-скелет, і власне RGB-відео (через MediaPipe) **без перенавчання**;
(в) забезпечує крос-суб'єктний протокол оцінювання. Тим самим адаптер
перетворює UI-PRMD на відтворюваний бенчмарк для запропонованого
конфігураційно-керованого методу одночасної валідації геометричних,
послідовнісних і темпоральних відхилень.

---

## 12. English abstract & glossary (for the journal manuscript)

**Abstract (draft).** We describe a dataset-adaptation layer that lets a single,
training-free, configuration-driven exercise validator operate on the UI-PRMD
rehabilitation benchmark. UI-PRMD ships hierarchical (parent-relative) Kinect
skeletons with per-joint YXZ Euler angles and no RGB video. Our adapter (i)
reconstructs absolute 3-D joint coordinates via forward kinematics over the
skeleton hierarchy; (ii) maps the 22-joint Kinect scheme onto a semantic joint
vocabulary shared with MediaPipe BlazePose, so one declarative exercise
configuration validates both clinical mocap skeletons and monocular RGB video
without any per-exercise retraining; and (iii) exposes a leave-one-subject-out
protocol for cross-subject generalisation. We verify the reconstruction
functionally (anatomically consistent joint-angle trajectories; frame-invariant
bone lengths) and end-to-end (the same squat configuration separates correct from
incorrect executions). The layer turns UI-PRMD into a reproducible benchmark for
the proposed universal validation method.

**Glossary (UA ↔ EN).**

| Українською | English |
|---|---|
| пряма кінематика | forward kinematics (FK) |
| ієрархічні / parent-relative координати | hierarchical / parent-relative coordinates |
| зміщення кістки (у локальній системі) | (local) bone offset |
| семантична абстракція суглобів | semantic joint abstraction |
| схема скелета | skeleton scheme |
| крос-суб'єктна генералізація | cross-subject generalisation |
| leave-one-subject-out | leave-one-subject-out (LOSO) |
| функціонал якості | quality functional (Q) |
| конфігураційно-керований метод | configuration-driven method |
| training-free / без перенавчання | training-free |
| відхилення: геометричні / послідовнісні / темпоральні | geometric / sequential / temporal deviations |
```