# План 2026: с нуля до своих нейросетей (24 недели)

**Цель:** инженерная грамотность в ML и DL. К концу плана вы собираете маленький GPT, дообучаете открытую LLM, понимаете diffusion и выкладываете демо.

**Формат:** всё проходится бесплатно (YouTube, Stepik, Kaggle, Hugging Face, audit на Coursera). Платные только сертификаты и, по желанию, GPU.

**Язык:** русский каркас и английские эталоны с субтитрами.

**Нагрузка:** будни (Д1–Д5) по 1–1,5 часа, Д6 (выходной) — 3 часа практики, Д7 — отдых. Всего 10–12 часов в неделю.

Актуализация сентября 2026. Логика сохранена: математика → классический ML → PyTorch и DL → современные архитектуры → проекты.

---

## 0. Что изменилось с плана 2025

- `nanoGPT` остаётся базой, но следующий шаг — [nanochat](#r-nanochat): токенизация → pretrain → SFT → оценка → чат.
- Дообучение открытых моделей (Qwen, Llama, Gemma, Mistral) важнее, чем писать GAN с нуля.
- Генерация изображений: diffusion, flow matching, DiT. На практике — LoRA и ComfyUI, а не только Automatic1111.
- Обязательный навык: локальный инференс (Ollama, llama.cpp, vLLM) и квантизация.
- Рабочий путь «модель с Hub → дообучение → Spaces» даёт [Hugging Face LLM Course](#r-hfllm).
- RAG и оценка качества (evals) стали базовыми навыками AI Engineer. В плане им отведена отдельная неделя.
- TensorFlow Developer Certificate закрыт. Основной фреймворк для учёбы и вакансий — PyTorch.
- Сертификат — фильтр для HR. Он не заменяет GitHub и задеплоенную модель.

За полгода реально собрать маленький GPT, дообучить открытую LLM, понять diffusion и выложить демо. Натренировать конкурента GPT-4 на ноутбуке нереально.

> Если вы уже уверенно пишете на Python и помните матрицы и производные, сожмите фазу 0 до одной недели (только неделя 2 — вероятность и градиент). Освободившееся время отдайте фазе 3.

---

## 1. Ресурсы

В плане по дням название ресурса ведёт сюда, к его полному описанию. Ссылка на тему в том же пункте открывает нужную главу, урок или видео.

**Обозначения в плане:** 📖 книга · 🎬 видео или лекция · 📄 курс, статья, документация · 🧪 квиз или проверка · 🛠 упражнение · 🚀 проект портфолио · 🎯 сертификат · ⏸ отдых

### Математика
- <a id="r-3b1b-la"></a>**🎬 3Blue1Brown — Essence of Linear Algebra** — 16 коротких видео: векторы, матрицы как преобразования, определитель, собственные векторы. Лучшая визуальная интуиция. Бесплатно. Оригинал: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab · русский дубляж Vert Dider: https://www.youtube.com/playlist?list=PL8YZyma552VfeWdaBSz5zj195mPwaDnnk
- <a id="r-3b1b-calc"></a>**🎬 3Blue1Brown — Essence of Calculus** — производная, цепное правило, интегралы. Нужна для понимания градиента и backprop. Бесплатно. https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
- <a id="r-3b1b-nn"></a>**🎬 3Blue1Brown — Neural Networks** — серия о нейросетях: устройство сети, градиентный спуск, backprop, GPT и attention. Бесплатно. Оригинал: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi · русская аудиодорожка: https://www.youtube.com/playlist?list=PLZHQObOWTQDM4E-dwvbnQTiyKDO-y9T2t
- <a id="r-statquest"></a>**🎬 StatQuest (Josh Starmer)** — вероятность, распределения, статистика и ML-алгоритмы простыми словами, по 5–20 минут. Закрывает пробел в вероятности, без которой непонятны функции потерь и diffusion. Бесплатно. https://www.youtube.com/@statquest
- <a id="r-mml"></a>**📖 Mathematics for Machine Learning** — Deisenroth, Faisal, Ong. Справочник по линейной алгебре, матанализу и вероятности именно под ML. Не читать подряд, а открывать по теме. Бесплатный PDF. https://mml-book.github.io/

### Python и данные
- <a id="r-selfedu"></a>**🎬 selfedu — «Добрый, добрый Python»** — Python с нуля по-русски. Бесплатно. https://www.youtube.com/playlist?list=PLA0M1Bcd0w8yWHh2V70bTtbVxJICrnJHd
- <a id="r-selfedu-np"></a>**🎬 selfedu — NumPy** — векторизация, broadcasting, индексация. Бесплатно. https://www.youtube.com/playlist?list=PLA0M1Bcd0w8zmegfAUfFMiACPKfdW4ifD
- <a id="r-kagglelearn"></a>**📄 Kaggle Learn** — короткие курсы с тетрадками прямо в браузере. Бесплатно. [Python](https://www.kaggle.com/learn/python) · [Pandas](https://www.kaggle.com/learn/pandas) · [Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning) · [Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) · [Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- <a id="r-sklearn"></a>**📄 scikit-learn User Guide** — документация с объяснением каждого алгоритма классического ML и примерами кода. Бесплатно. https://scikit-learn.org/stable/user_guide.html

### Классический ML
- <a id="r-ng"></a>**🎬 Andrew Ng — Machine Learning Specialization** — DeepLearning.AI и Coursera, 3 курса. Лучший «словарь» ML: регрессия, bias/variance, регуляризация, деревья, рекомендательные системы. Audit бесплатный, субтитры есть. https://www.coursera.org/specializations/machine-learning-introduction
- <a id="r-dls"></a>**🎬 Deep Learning School (МФТИ)** — русскоязычный курс с домашками. Бесплатно. [Базовый поток](https://www.youtube.com/playlist?list=PL0Ks75aof3Th84kETSlJq_ja-xqLtWov1) · [часть 1: intro ML/DL](https://www.youtube.com/playlist?list=PL0Ks75aof3TiHbkJ95vxNlQefujrj1N2w) · [NLP](https://www.youtube.com/playlist?list=PL0Ks75aof3ThuLLtLIVl_KPUDDQlTDyJI) · актуальные семестры: https://dls.samcs.ru/ · домашки: https://stepik.org/org/dlschool
- <a id="r-kaggle"></a>**🛠 Kaggle Competitions** — учебные соревнования для пайплайна «данные → признаки → модель → валидация → ошибка». Медаль не нужна. [Titanic](https://www.kaggle.com/competitions/titanic) · [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) · [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) · [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

### Глубокое обучение
- <a id="r-zth"></a>**🎬 Andrej Karpathy — Neural Networks: Zero to Hero** — нейросети изнутри, в коде, от backprop до GPT-2. Порядок видео жёсткий, ручной backprop не пропускать. Бесплатно. Плейлист: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ · сайт: https://karpathy.ai/zero-to-hero.html
- <a id="r-fastai"></a>**🎬 fast.ai — Practical Deep Learning for Coders** — подход сверху вниз: с первого урока обучаете модель, теорию разбираете вокруг работающего кода. Бесплатно. https://course.fast.ai/
- <a id="r-pytorch"></a>**📄 PyTorch Tutorials** — официальные «Learn the Basics»: тензоры, Dataset и DataLoader, модели, autograd, цикл обучения, сохранение. Бесплатно. https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- <a id="r-hse"></a>**📄 ФКН ВШЭ — Intro to DL** — материалы 2025–2026: лекции, семинары и домашки. Бесплатно. https://github.com/xiyori/intro-to-dl-hse
- <a id="r-cs231n"></a>**🎬 Stanford CS231N (Spring 2025)** — свёрточные сети и компьютерное зрение. Бесплатно. https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16
- <a id="r-udl"></a>**📖 Understanding Deep Learning** — Simon Prince, MIT Press 2023. Современный учебник: от основ до трансформеров, diffusion и normalizing flows, с блокнотами. Бесплатный PDF. https://udlbook.github.io/udlbook/
- <a id="r-d2l"></a>**📖 Dive into Deep Learning (d2l.ai)** — интерактивный учебник с кодом на PyTorch для каждого раздела. Справочник на всю фазу 2. Бесплатно. https://d2l.ai/
- <a id="r-wandb"></a>**🛠 Трекинг экспериментов** — Weights & Biases (бесплатно для личного использования) или TensorBoard. Без него через месяц не вспомнить, какой запуск был лучшим. https://wandb.ai/site · https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

### Трансформеры и LLM
- <a id="r-illustrated"></a>**📄 The Illustrated Transformer** — Jay Alammar. Трансформер в картинках: self-attention, multi-head, позиционное кодирование. Бесплатно. https://jalammar.github.io/illustrated-transformer/
- <a id="r-nanogpt"></a>**🛠 nanoGPT** — Karpathy. Минимальный код для обучения GPT: от Шекспира по символам до воспроизведения GPT-2. Бесплатно. https://github.com/karpathy/nanoGPT
- <a id="r-nanochat"></a>**🛠 nanochat** — Karpathy. Полный конвейер маленького ChatGPT: токенизатор, pretrain, SFT, оценка, веб-чат. Бесплатно. https://github.com/karpathy/nanochat
- <a id="r-cs336"></a>**🎬 Stanford CS336 — Language Modeling from Scratch** — университетский уровень: токенизация, архитектура, GPU, масштабирование, данные, выравнивание. По желанию. Бесплатно. https://stanford-cs336.github.io/
- <a id="r-hfllm"></a>**📄 Hugging Face LLM Course** — Transformers, Datasets, Tokenizers, дообучение, публикация на Hub. Квизы в главах. Бесплатно. https://huggingface.co/learn/llm-course/chapter1/1
- <a id="r-peft"></a>**📄 PEFT, TRL и Unsloth** — библиотеки для LoRA, QLoRA и SFT. Unsloth даёт готовые Colab- и Kaggle-блокноты для дообучения Qwen, Llama и Gemma на бесплатной GPU. Бесплатно. [PEFT](https://huggingface.co/docs/peft/index) · [TRL](https://huggingface.co/docs/trl/index) · [Unsloth](https://docs.unsloth.ai/)
- <a id="r-local"></a>**🛠 Локальный инференс** — Ollama (запуск модели одной командой), llama.cpp (GGUF-квантизация) и vLLM (сервер с высокой пропускной способностью). Модель 7–8B на потребительской карте — норма. Бесплатно. [Ollama](https://ollama.com/) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [vLLM](https://docs.vllm.ai/)
- <a id="r-rag"></a>**📄 RAG и оценка качества** — Hugging Face Cookbook: RAG от простого к продвинутому и оценка RAG через LLM-as-a-judge. Бесплатно. [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag) · [RAG Evaluation](https://huggingface.co/learn/cookbook/rag_evaluation)
- <a id="r-hfagents"></a>**📄 Hugging Face Agents Course** — агенты, инструменты, фреймворки (smolagents, LangGraph, LlamaIndex). Сертификат выдают за своего агента, прошедшего порог на бенчмарке GAIA. Бесплатно. https://huggingface.co/learn/agents-course

### Diffusion и изображения
- <a id="r-hfdiff"></a>**📄 Hugging Face Diffusion Course** — diffusion с нуля и через библиотеку Diffusers: DDPM, fine-tune, guidance, Stable Diffusion. Бесплатно. https://huggingface.co/learn/diffusion-course/unit0/1
- <a id="r-cme296"></a>**🎬 Stanford CME296 — Diffusion & Large Vision Models (Spring 2026)** — теория: diffusion, score matching, flow matching, U-Net, DiT и MM-DiT, guidance, оценка. Лекции на YouTube. Бесплатно. https://cme296.stanford.edu/
- <a id="r-comfy"></a>**🛠 ComfyUI** — нодовый интерфейс для открытых моделей изображений (SDXL, семейство FLUX и новее). Стандарт 2026 вместо Automatic1111. Бесплатно. https://github.com/comfyanonymous/ComfyUI
- <a id="r-cvweek"></a>**🎬 ШАД и Яндекс — CV Week** — по желанию, по-русски: материалы по diffusion. Бесплатно. Ищите «CV Week ШАД» на YouTube.

### Статьи
- <a id="r-papers"></a>**📄 Ключевые статьи** — по одной на фазу. Сначала посмотрите разбор, потом читайте сам текст: аннотацию, рисунки, метод. Бесплатно. [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) · [RoFormer / RoPE (2021)](https://arxiv.org/abs/2104.09864) · [LoRA (2021)](https://arxiv.org/abs/2106.09685) · [DDPM (2020)](https://arxiv.org/abs/2006.11239) · [Flow Matching (2022)](https://arxiv.org/abs/2210.02747) · [DiT (2022)](https://arxiv.org/abs/2212.09748)

### Где считать и деплоить
- <a id="r-gpu"></a>**🛠 Бесплатные GPU** — Kaggle Notebooks (бесплатная квота GPU-часов в неделю) и Google Colab (бесплатный тир, при необходимости недорогой Pro). Своей карты на 8–16 ГБ хватает для учёбы и LoRA, но не для pretrain миллиардов параметров. https://www.kaggle.com/code · https://colab.research.google.com/
- <a id="r-spaces"></a>**🛠 Hugging Face Spaces и Gradio** — бесплатный хостинг демо; Gradio собирает веб-интерфейс к модели в несколько строк. https://huggingface.co/spaces · https://www.gradio.app/guides/quickstart

### Сертификаты
- <a id="r-cert-ml"></a>**🎯 Machine Learning Specialization (DeepLearning.AI)** — самый узнаваемый сигнал «я знаю ML». Сертификат платный (подписка Coursera, около $49 в месяц, есть financial aid). https://www.coursera.org/specializations/machine-learning-introduction
- <a id="r-cert-dl"></a>**🎯 PyTorch for Deep Learning Professional Certificate или Deep Learning Specialization** — DeepLearning.AI. Первый практичнее (тензоры, CV и NLP, Hugging Face, деплой через ONNX и квантизацию), второй академичнее. Альтернатива — сертификат DLS на Stepik при сданных домашках. https://www.deeplearning.ai/courses/ · https://www.coursera.org/specializations/deep-learning
- <a id="r-cert-hf"></a>**🎯 Сертификаты Hugging Face** — квизы LLM Course и сертификат Agents Course. Бесплатно. На собеседовании весомее, чем в HR-фильтре: на Hub видны ваши модели и Spaces. https://huggingface.co/learn
- <a id="r-cert-cloud"></a>**🎯 Облачный экзамен (один)** — Google Professional ML Engineer (около $200) · AWS Certified ML Engineer – Associate · Azure AI Engineer (AI-102, около $165; в 2026 добавляются AI-103 и AI-300). Сдавать после плана, когда есть проект на этой платформе. См. раздел «Сертификаты» в конце. [Google PMLE](https://cloud.google.com/learn/certification/machine-learning-engineer) · [AWS MLA](https://aws.amazon.com/certification/certified-machine-learning-engineer-associate/) · [Azure AI-102](https://learn.microsoft.com/credentials/certifications/azure-ai-engineer/)

---

## 2. Повторяющийся шаблон практики (каждый Д6)

Каждую неделю — **3 часа кода, который запускается**, а не только просмотр лекций.

1. Запустите чужой код 1-в-1: репозиторий, блокнот или решение из урока (45 мин).
2. Поменяйте одну деталь: данные, гиперпараметр, слой. Запишите гипотезу до запуска (60 мин).
3. Сравните метрики. Если есть трекинг, смотрите в [W&B или TensorBoard](#r-wandb) (30 мин).
4. Закоммитьте код в GitHub и допишите запись в `ml-journal.md`: что сделали, что получилось, что непонятно (45 мин).

Застряли — сначала воспроизведите оригинал точно, потом меняйте по одной детали.

---

## ФАЗА 0. Математика и Python (недели 1–3)

Пропускайте, если уверенно пишете на Python и не пугаетесь матриц и градиента.

### Неделя 1. Линейная алгебра
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [3Blue1Brown: линейная алгебра](#r-3b1b-la), видео 1–3: векторы, линейные комбинации, матрицы как преобразования</li></ul> |
| Д2 | <ul><li>🎬 [3Blue1Brown: линейная алгебра](#r-3b1b-la), видео 4–6: умножение матриц, 3D, определитель</li></ul> |
| Д3 | <ul><li>🎬 [3Blue1Brown: линейная алгебра](#r-3b1b-la), видео 7–9: обратная матрица, ранг, скалярное произведение</li></ul> |
| Д4 | <ul><li>🎬 [3Blue1Brown: линейная алгебра](#r-3b1b-la), видео 10–14: смена базиса, собственные векторы</li><li>📖 [Mathematics for ML](#r-mml), гл. 2 — пролистать как справочник</li></ul> |
| Д5 | <ul><li>🎬 [selfedu: NumPy](#r-selfedu-np), первые уроки: массивы, индексация, broadcasting</li></ul> |
| Д6 | <ul><li>🛠 **Упражнение:** на NumPy без циклов — умножение матриц, поворот точек на плоскости матрицей, нормализация векторов; сверьте с `np.linalg`</li></ul> |

### Неделя 2. Производные, градиент и вероятность
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [3Blue1Brown: Essence of Calculus](#r-3b1b-calc), видео 1–4: производная, степенные функции</li></ul> |
| Д2 | <ul><li>🎬 [3Blue1Brown: Essence of Calculus](#r-3b1b-calc): цепное правило и производная произведения</li><li>🛠 **Упражнение:** на бумаге найдите производную `sigmoid(w·x + b)` по `w`</li></ul> |
| Д3 | <ul><li>🎬 [StatQuest](#r-statquest): вероятность, нормальное распределение, maximum likelihood</li></ul> |
| Д4 | <ul><li>🎬 [StatQuest](#r-statquest): условная вероятность, теорема Байеса, энтропия и cross-entropy</li></ul> |
| Д5 | <ul><li>🎬 [3Blue1Brown: Neural Networks](#r-3b1b-nn), видео 1–2: что такое нейросеть, градиентный спуск</li></ul> |
| Д6 | <ul><li>🛠 **Упражнение:** градиентный спуск на NumPy — подберите прямую `y = w·x + b` по зашумлённым точкам, нарисуйте кривую потерь</li></ul> |

### Неделя 3. Python для ML и интуиция нейросетей
| День | Что делать |
|---|---|
| Д1 | <ul><li>📄 [Kaggle Learn: Python](#r-kagglelearn) — или, если Python новый, [selfedu: Python](#r-selfedu) по ключевым темам</li></ul> |
| Д2 | <ul><li>📄 [Kaggle Learn: Pandas](#r-kagglelearn), уроки 1–3</li></ul> |
| Д3 | <ul><li>📄 [Kaggle Learn: Pandas](#r-kagglelearn), уроки 4–6</li></ul> |
| Д4 | <ul><li>🎬 [3Blue1Brown: Neural Networks](#r-3b1b-nn), видео 3–4: backpropagation и его матанализ</li></ul> |
| Д5 | <ul><li>🎬 [selfedu: NumPy](#r-selfedu-np): векторизация, агрегаты, `reshape`, `axis`</li></ul> |
| Д6 | <ul><li>🛠 **Контрольная фазы:** загрузите CSV в Pandas, почистите, посчитайте статистики, постройте 3 графика, перепишите один цикл Python в векторизованный NumPy</li></ul> |

**Минимум на выход из фазы:** векторы и матрицы, скалярное произведение, производная сложной функции, градиентный спуск, вероятность и cross-entropy, NumPy без циклов по элементам.

---

## ФАЗА 1. Классическое машинное обучение (недели 4–7)

Без этой фазы глубокое обучение превращается в копирование ноутбуков. Английский эталон — [Andrew Ng](#r-ng), русский путь — [DLS](#r-dls). Смотрите один основной курс, второй — по сложным темам.

### Неделя 4. Регрессия и градиентный спуск
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 1, неделя 1: supervised и unsupervised learning, линейная регрессия, функция стоимости</li></ul> |
| Д2 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 1, неделя 1: градиентный спуск, learning rate</li></ul> |
| Д3 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 1, неделя 2: множественная регрессия, масштабирование признаков, полиномиальные признаки</li></ul> |
| Д4 | <ul><li>📄 [Kaggle Learn: Intro to ML](#r-kagglelearn), уроки 1–4</li></ul> |
| Д5 | <ul><li>📄 [Kaggle Learn: Intro to ML](#r-kagglelearn), уроки 5–7: переобучение, random forest</li></ul> |
| Д6 | <ul><li>🛠 [Kaggle: House Prices](#r-kaggle) — первый сабмит: baseline на линейной регрессии и random forest, честная валидация</li></ul> |

### Неделя 5. Классификация и регуляризация
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 1, неделя 3: логистическая регрессия, граница решения, log loss</li></ul> |
| Д2 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 1, неделя 3: переобучение, регуляризация L1 и L2</li></ul> |
| Д3 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 2, неделя 3: bias/variance, learning curves, выбор модели</li></ul> |
| Д4 | <ul><li>🎬 [StatQuest](#r-statquest): precision, recall, ROC и AUC, кросс-валидация</li></ul> |
| Д5 | <ul><li>📄 [scikit-learn User Guide](#r-sklearn): Linear Models и Model selection (cross-validation, метрики)</li></ul> |
| Д6 | <ul><li>🛠 [Kaggle: Titanic](#r-kaggle) — пайплайн: данные → признаки → модель → валидация → разбор ошибок</li></ul> |

### Неделя 6. Деревья, ансамбли, бустинг
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 2, неделя 4: деревья решений, энтропия, information gain</li></ul> |
| Д2 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 2, неделя 4: ансамбли, random forest, XGBoost</li></ul> |
| Д3 | <ul><li>🎬 [DLS](#r-dls), базовый поток: лекция о деревьях и градиентном бустинге</li></ul> |
| Д4 | <ul><li>📄 [Kaggle Learn: Intermediate ML](#r-kagglelearn): пропуски, категориальные признаки, pipelines, XGBoost</li></ul> |
| Д5 | <ul><li>📄 [Kaggle Learn: Intermediate ML](#r-kagglelearn): cross-validation, утечки данных</li></ul> |
| Д6 | <ul><li>🛠 [Kaggle: Spaceship Titanic](#r-kaggle) — бустинг против линейной модели, feature importance, запись выводов в `ml-journal.md`</li></ul> |

### Неделя 7. Без учителя, признаки и контрольная точка
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 3, неделя 1: кластеризация k-means, поиск аномалий</li></ul> |
| Д2 | <ul><li>🎬 [Andrew Ng](#r-ng), курс 3, неделя 2: рекомендательные системы</li></ul> |
| Д3 | <ul><li>🎬 [StatQuest](#r-statquest): PCA</li><li>📄 [Kaggle Learn: Feature Engineering](#r-kagglelearn), уроки 1–3</li></ul> |
| Д4 | <ul><li>🛠 **Упражнение:** логистическая регрессия с нуля на NumPy (градиентный спуск, log loss), сверка с `sklearn`</li></ul> |
| Д5 | <ul><li>🧪 Самопроверка: объясните вслух bias/variance, регуляризацию, утечку данных и выбор метрики</li><li>🎯 По желанию: [Machine Learning Specialization](#r-cert-ml) — сдать задания курса</li></ul> |
| Д6 | <ul><li>🚀 **Итог фазы:** один чистый репозиторий на GitHub с лучшим Kaggle-пайплайном и README (задача, валидация, метрика, что не сработало)</li></ul> |

---

## ФАЗА 2. Глубокое обучение: backprop, PyTorch, CNN, RNN (недели 8–14)

Главный стержень — [Karpathy Zero to Hero](#r-zth), видео про micrograd и makemore. Видео про GPT перенесены в фазу 3, чтобы не проходить их дважды. Русский каркас — [DLS](#r-dls) и [ФКН ВШЭ](#r-hse), практичный top-down — [fast.ai](#r-fastai).

### Неделя 8. micrograd: backprop руками
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Karpathy](#r-zth): [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0), первая треть — производная, граф вычислений</li></ul> |
| Д2 | <ul><li>🎬 [Karpathy](#r-zth): micrograd, вторая треть — ручной backprop, цепное правило в коде</li></ul> |
| Д3 | <ul><li>🎬 [Karpathy](#r-zth): micrograd, финал — нейрон, MLP, цикл обучения, сравнение с PyTorch</li></ul> |
| Д4 | <ul><li>🛠 **Упражнение:** напишите micrograd сами, не подглядывая. Добавьте операции `exp`, `tanh`, `relu`</li></ul> |
| Д5 | <ul><li>📄 [PyTorch Tutorials](#r-pytorch): Tensors, Autograd</li></ul> |
| Д6 | <ul><li>🛠 Обучите свой micrograd-MLP на игрушечном датасете (moons); проверьте градиенты численно</li></ul> |

### Неделя 9. makemore: языковая модель по символам
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Karpathy](#r-zth): [makemore, часть 1](https://www.youtube.com/watch?v=PaCmpygFfXo) — биграммы, первая половина</li></ul> |
| Д2 | <ul><li>🎬 [Karpathy](#r-zth): makemore, часть 1 — нейросетевая биграмма, negative log likelihood</li></ul> |
| Д3 | <ul><li>🎬 [Karpathy](#r-zth): [makemore, часть 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I), первая половина — эмбеддинги</li></ul> |
| Д4 | <ul><li>🎬 [Karpathy](#r-zth): makemore, часть 2 — train/dev/test, подбор learning rate</li></ul> |
| Д5 | <ul><li>📄 [PyTorch Tutorials](#r-pytorch): Datasets & DataLoaders, Build the Neural Network, Optimization Loop</li></ul> |
| Д6 | <ul><li>🛠 makemore на русских именах или городах: добейтесь loss ниже биграммы, сгенерируйте 20 примеров</li></ul> |

### Неделя 10. Обучение глубоких сетей
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Karpathy](#r-zth): [makemore, часть 3](https://www.youtube.com/watch?v=P6sfmUTpUmc) — инициализация, активации, насыщение</li></ul> |
| Д2 | <ul><li>🎬 [Karpathy](#r-zth): makemore, часть 3 — BatchNorm, диагностические графики</li></ul> |
| Д3 | <ul><li>🎬 [Karpathy](#r-zth): [makemore, часть 4: backprop ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI), первая половина</li></ul> |
| Д4 | <ul><li>🎬 [Karpathy](#r-zth): makemore, часть 4 — backprop через cross-entropy и BatchNorm вручную</li></ul> |
| Д5 | <ul><li>🎬 [Karpathy](#r-zth): [makemore, часть 5: WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)</li></ul> |
| Д6 | <ul><li>🛠 Подключите [трекинг экспериментов](#r-wandb): 5 запусков makemore с разной инициализацией и BatchNorm, сравнение кривых</li></ul> |

### Неделя 11. fast.ai: модель в проде с первого урока
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [fast.ai](#r-fastai): урок 1 — классификатор изображений за вечер</li></ul> |
| Д2 | <ul><li>🎬 [fast.ai](#r-fastai): урок 2 — деплой модели</li></ul> |
| Д3 | <ul><li>🎬 [fast.ai](#r-fastai): урок 3 — нейросеть с нуля в таблице и коде</li></ul> |
| Д4 | <ul><li>🎬 [fast.ai](#r-fastai): урок 4 — NLP и Hugging Face</li></ul> |
| Д5 | <ul><li>🎬 [fast.ai](#r-fastai): урок 5 — модель с нуля; сравните с micrograd</li></ul> |
| Д6 | <ul><li>🛠 Свой классификатор изображений (fine-tune предобученной сети) на своих фото; демо в [Gradio](#r-spaces)</li></ul> |

### Неделя 12. Свёрточные сети и компьютерное зрение
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [DLS, часть 1](#r-dls): лекция о свёрточных сетях</li></ul> |
| Д2 | <ul><li>🎬 [CS231N](#r-cs231n): лекция о CNN — свёртка, pooling, receptive field</li></ul> |
| Д3 | <ul><li>🎬 [CS231N](#r-cs231n): лекция об архитектурах — VGG, ResNet, residual connections</li></ul> |
| Д4 | <ul><li>📖 [Understanding Deep Learning](#r-udl), гл. 10–11: свёрточные сети и residual</li></ul> |
| Д5 | <ul><li>📄 [ФКН ВШЭ](#r-hse): семинар по CNN в PyTorch</li></ul> |
| Д6 | <ul><li>🛠 CNN на PyTorch с нуля для CIFAR-10 (без готовых моделей) + аугментации; сравните с fine-tune из недели 11</li></ul> |

### Неделя 13. Последовательности: эмбеддинги, RNN, LSTM
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [DLS, NLP](#r-dls): эмбеддинги слов, word2vec</li></ul> |
| Д2 | <ul><li>🎬 [DLS, NLP](#r-dls): RNN, LSTM, GRU, затухание градиента</li></ul> |
| Д3 | <ul><li>🎬 [DLS, NLP](#r-dls): seq2seq и первая идея attention</li></ul> |
| Д4 | <ul><li>📖 [Dive into Deep Learning](#r-d2l), главы о RNN и механизме внимания — код как справочник</li></ul> |
| Д5 | <ul><li>📄 [ФКН ВШЭ](#r-hse): домашка или семинар по RNN</li></ul> |
| Д6 | <ul><li>🛠 LSTM для генерации текста на русской прозе; сравните с makemore-MLP по loss и качеству текста</li></ul> |

### Неделя 14. Консолидация и контрольная точка
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [DLS, часть 1](#r-dls): оптимизаторы (SGD, momentum, Adam), learning rate schedules</li></ul> |
| Д2 | <ul><li>🎬 [DLS, часть 1](#r-dls): регуляризация в DL — dropout, weight decay, early stopping</li></ul> |
| Д3 | <ul><li>🧪 Самопроверка вслух: backprop на примере, зачем residual и нормализация, почему Adam, что такое переобучение в DL</li></ul> |
| Д4 | <ul><li>🎯 По желанию: [PyTorch Professional Certificate или сертификат DLS](#r-cert-dl) — сверьте программу, запланируйте</li></ul> |
| Д5 | <ul><li>🛠 Приведите в порядок репозитории фазы: README, графики, воспроизводимый запуск</li></ul> |
| Д6 | <ul><li>🚀 **Итог фазы:** классификатор изображений или текстов, обученный вами, с демо на [Hugging Face Spaces](#r-spaces)</li></ul> |

---

## ФАЗА 3. Современные архитектуры: трансформеры, LLM, diffusion (недели 15–24)

Три трека идут по очереди блоками, а проекты портфолио (фаза 4 исходного плана) встроены в Д6. Не нужно закрывать трек целиком, прежде чем писать код.

- **Трек A.** Собрать языковую модель самому: недели 15–17.
- **Трек B.** Экосистема 2026: Hugging Face, LoRA, локальный инференс, RAG, агенты. Недели 18–21.
- **Трек C.** Изображения и генеративные модели: недели 22–23.

### Неделя 15. Attention и трансформер
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [3Blue1Brown: Neural Networks](#r-3b1b-nn): [But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M)</li></ul> |
| Д2 | <ul><li>🎬 [3Blue1Brown: Neural Networks](#r-3b1b-nn): [Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)</li></ul> |
| Д3 | <ul><li>📄 [The Illustrated Transformer](#r-illustrated)</li></ul> |
| Д4 | <ul><li>🎬 [Karpathy](#r-zth): [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), первая половина — от биграммы к self-attention</li></ul> |
| Д5 | <ul><li>🎬 [Karpathy](#r-zth): Let's build GPT, вторая половина — multi-head, residual, LayerNorm</li></ul> |
| Д6 | <ul><li>🛠 Допишите GPT из видео сами; обучите на Шекспире по символам. Прочитайте [Attention Is All You Need](#r-papers) — рисунки и раздел 3</li></ul> |

### Неделя 16. Токенизация и свой маленький GPT
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Karpathy](#r-zth): [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE), первая треть — Unicode, UTF-8</li></ul> |
| Д2 | <ul><li>🎬 [Karpathy](#r-zth): Tokenizer, вторая треть — алгоритм BPE</li></ul> |
| Д3 | <ul><li>🎬 [Karpathy](#r-zth): Tokenizer, финал — tiktoken, sentencepiece, странности токенизации</li></ul> |
| Д4 | <ul><li>🛠 **Упражнение:** свой BPE-токенизатор; обучите на русском тексте и сравните длину в токенах с tiktoken</li></ul> |
| Д5 | <ul><li>🛠 [nanoGPT](#r-nanogpt): прочитайте `model.py` и `train.py`, запустите пример shakespeare_char</li></ul> |
| Д6 | <ul><li>🚀 **Проект 1:** свой крошечный GPT на русской прозе в [nanoGPT](#r-nanogpt) на [бесплатной GPU](#r-gpu); выложите код и примеры генерации</li></ul> |

### Неделя 17. Воспроизвести GPT-2 и современные детали архитектуры
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Karpathy](#r-zth): [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), часть 1 — архитектура, загрузка весов</li></ul> |
| Д2 | <ul><li>🎬 [Karpathy](#r-zth): GPT-2, часть 2 — скорость: mixed precision, flash attention, `torch.compile`</li></ul> |
| Д3 | <ul><li>🎬 [Karpathy](#r-zth): GPT-2, часть 3 — гиперпараметры, распределённое обучение, оценка</li></ul> |
| Д4 | <ul><li>📄 Современные детали: RMSNorm, SwiGLU, RoPE — статья [RoFormer](#r-papers), разделы 1–3</li><li>🎬 По желанию: [Stanford CS336](#r-cs336), лекция об архитектурах</li></ul> |
| Д5 | <ul><li>🛠 [nanochat](#r-nanochat): прочитайте README и схему конвейера (токенизатор → pretrain → SFT → оценка → чат)</li></ul> |
| Д6 | <ul><li>🛠 Добавьте в свой GPT RoPE и RMSNorm вместо позиционных эмбеддингов и LayerNorm; сравните loss. По желанию — прогон nanochat в минимальной конфигурации</li></ul> |

### Неделя 18. Экосистема Hugging Face
| День | Что делать |
|---|---|
| Д1 | <ul><li>📄 [HF LLM Course](#r-hfllm): [глава 1](https://huggingface.co/learn/llm-course/chapter1/1) — трансформеры и pipeline</li></ul> |
| Д2 | <ul><li>📄 [HF LLM Course](#r-hfllm): [глава 2](https://huggingface.co/learn/llm-course/chapter2/1) — модели и токенизаторы изнутри</li></ul> |
| Д3 | <ul><li>📄 [HF LLM Course](#r-hfllm): [глава 3](https://huggingface.co/learn/llm-course/chapter3/1) — fine-tune предобученной модели, Trainer</li></ul> |
| Д4 | <ul><li>📄 [HF LLM Course](#r-hfllm): [глава 4](https://huggingface.co/learn/llm-course/chapter4/1) — публикация модели на Hub</li></ul> |
| Д5 | <ul><li>📄 [HF LLM Course](#r-hfllm): [глава 5](https://huggingface.co/learn/llm-course/chapter5/1) — библиотека Datasets</li><li>🧪 Квизы глав 1–5</li></ul> |
| Д6 | <ul><li>🛠 Fine-tune небольшой модели-энкодера на русской классификации текстов, публикация модели и model card на Hub</li></ul> |

### Неделя 19. LoRA и QLoRA: дообучить открытую LLM
| День | Что делать |
|---|---|
| Д1 | <ul><li>📄 [HF LLM Course](#r-hfllm): [глава 11](https://huggingface.co/learn/llm-course/chapter11/1) — supervised fine-tuning, chat templates</li></ul> |
| Д2 | <ul><li>📄 [HF LLM Course](#r-hfllm), глава 11: LoRA; статья [LoRA](#r-papers) — аннотация, рисунок 1, раздел 4</li></ul> |
| Д3 | <ul><li>📄 [PEFT, TRL и Unsloth](#r-peft): QLoRA, 4-битная загрузка, готовый блокнот Unsloth для Qwen или Llama</li></ul> |
| Д4 | <ul><li>🛠 Соберите датасет из своих текстов (200–1000 примеров в формате чата); проверьте лицензию выбранной модели</li></ul> |
| Д5 | <ul><li>🛠 Первый прогон QLoRA на [бесплатной GPU](#r-gpu), сравнение ответов до и после на 10 одинаковых вопросах</li></ul> |
| Д6 | <ul><li>🚀 **Проект 2:** LoRA-дообучение открытой LLM (Qwen, Llama, Gemma или Mistral) под свою задачу; адаптер на Hub + таблица «до и после»</li></ul> |

### Неделя 20. Локальный инференс, квантизация, деплой
| День | Что делать |
|---|---|
| Д1 | <ul><li>🛠 [Ollama](#r-local): запустите модель 7–8B локально, API-запрос из Python</li></ul> |
| Д2 | <ul><li>🛠 [llama.cpp](#r-local): форматы GGUF и уровни квантизации (Q4_K_M, Q8_0), замер скорости и памяти</li></ul> |
| Д3 | <ul><li>🛠 Сконвертируйте свою LoRA-модель из недели 19 в GGUF и запустите в Ollama</li></ul> |
| Д4 | <ul><li>📄 [vLLM](#r-local): когда нужен сервер вместо Ollama — батчинг, PagedAttention, OpenAI-совместимый API</li></ul> |
| Д5 | <ul><li>📄 [Gradio](#r-spaces): чат-интерфейс; [HF Spaces](#r-spaces): бесплатное железо и его ограничения</li></ul> |
| Д6 | <ul><li>🚀 **Проект 4 (деплой):** чат-демо своей дообученной модели на Hugging Face Space или простом Gradio / FastAPI</li></ul> |

### Неделя 21. RAG, оценка качества, агенты
| День | Что делать |
|---|---|
| Д1 | <ul><li>📄 [RAG](#r-rag): Advanced RAG — чанкинг, эмбеддинги, векторный поиск, reranking</li></ul> |
| Д2 | <ul><li>📄 [RAG](#r-rag): RAG Evaluation — синтетический тестовый набор, LLM-as-a-judge</li></ul> |
| Д3 | <ul><li>📄 [HF Agents Course](#r-hfagents): [Unit 1](https://huggingface.co/learn/agents-course/unit1/introduction) — что такое агент, инструменты, цикл Thought-Action-Observation</li></ul> |
| Д4 | <ul><li>📄 [HF Agents Course](#r-hfagents): [Unit 2](https://huggingface.co/learn/agents-course/unit2/introduction) — фреймворки (smolagents, LangGraph, LlamaIndex)</li></ul> |
| Д5 | <ul><li>🛠 **Упражнение:** 20 вопросов с эталонными ответами по своим документам — это ваш eval-набор</li></ul> |
| Д6 | <ul><li>🛠 RAG по своим документам на локальной модели из недели 20; прогон eval-набора до и после улучшения чанкинга</li></ul> |

### Неделя 22. Diffusion: теория и DDPM с нуля
| День | Что делать |
|---|---|
| Д1 | <ul><li>📄 [HF Diffusion Course](#r-hfdiff): [Unit 1](https://huggingface.co/learn/diffusion-course/unit1/1) — введение в Diffusers</li></ul> |
| Д2 | <ul><li>📄 [HF Diffusion Course](#r-hfdiff), Unit 1: diffusion с нуля — зашумление, U-Net, сэмплирование</li></ul> |
| Д3 | <ul><li>🎬 [Stanford CME296](#r-cme296): лекции о diffusion и score matching</li></ul> |
| Д4 | <ul><li>📖 [Understanding Deep Learning](#r-udl), гл. 18: diffusion models</li><li>📄 Статья [DDPM](#r-papers): раздел 2 и алгоритмы 1–2</li></ul> |
| Д5 | <ul><li>🛠 Реализуйте прямой процесс зашумления и визуализируйте шаги на MNIST</li></ul> |
| Д6 | <ul><li>🚀 **Проект 3, вариант А:** DDPM с нуля на MNIST или CIFAR-10 — формула шума перестаёт быть магией</li></ul> |

### Неделя 23. Flow matching, DiT, ComfyUI
| День | Что делать |
|---|---|
| Д1 | <ul><li>🎬 [Stanford CME296](#r-cme296): [лекция 3 — flow matching](https://www.youtube.com/watch?v=agN3AlfGFrk)</li></ul> |
| Д2 | <ul><li>🎬 [Stanford CME296](#r-cme296): латентная diffusion, VAE, guidance</li><li>📄 Статья [Flow Matching](#r-papers): аннотация и рисунки</li></ul> |
| Д3 | <ul><li>🎬 [Stanford CME296](#r-cme296): архитектуры — U-Net, DiT, MM-DiT</li><li>📄 Статья [DiT](#r-papers): аннотация и рисунок 3</li></ul> |
| Д4 | <ul><li>🛠 [ComfyUI](#r-comfy): установка, базовый граф text-to-image на открытой модели</li></ul> |
| Д5 | <ul><li>📄 [HF Diffusion Course](#r-hfdiff): fine-tune и guidance; как устроена LoRA для изображений</li><li>🎬 По желанию: [CV Week](#r-cvweek)</li></ul> |
| Д6 | <ul><li>🚀 **Проект 3, вариант Б:** LoRA для генерации изображений + понятный пайплайн в ComfyUI (граф в репозитории, примеры до и после)</li></ul> |

### Неделя 24. Портфолио и итоговая проверка
| День | Что делать |
|---|---|
| Д1 | <ul><li>📄 [HF Agents Course](#r-hfagents): [Unit 4](https://huggingface.co/learn/agents-course/unit4/introduction) — финальное задание; начните своего агента под GAIA</li></ul> |
| Д2 | <ul><li>🛠 Агент: доведите до порога сертификата [Hugging Face](#r-cert-hf)</li></ul> |
| Д3 | <ul><li>🛠 README для 4 проектов: задача, данные, метрики, ограничения, как запустить</li></ul> |
| Д4 | <ul><li>🧪 Пройдите «Чек-лист через 6 месяцев» вслух, по пункту на 5 минут; пробелы запишите</li></ul> |
| Д5 | <ul><li>🎯 Выберите [облачный экзамен](#r-cert-cloud) по вакансиям своего рынка и запишитесь на дату через 6–8 недель</li></ul> |
| Д6 | <ul><li>🚀 **Итог плана:** профиль GitHub и Hugging Face с 4 проектами и живым демо; обновлённое резюме</li></ul> |

---

## Проекты портфолио

Четыре проекта важнее сорока курсов. Каждый — публичный репозиторий с README, метриками и честным разделом «что не сработало».

| № | Проект | Неделя | Результат |
|---|---|---|---|
| 1 | Крошечный GPT на Шекспире или русской прозе ([nanoGPT](#r-nanogpt) / [nanochat](#r-nanochat)) | 16–17 | Код, кривые loss, примеры генерации |
| 2 | LoRA-дообучение открытой LLM под свою задачу | 19 | Адаптер на Hub, таблица «до и после» |
| 3 | Генерация изображений: свой DDPM или LoRA + пайплайн в ComfyUI | 22–23 | Код или граф, примеры |
| 4 | Публичный деплой: Hugging Face Space или Gradio / FastAPI | 20 | Живая ссылка на демо |

Бонус: RAG с eval-набором (неделя 21) — самый частый вопрос на собеседованиях AI Engineer в 2026.

## Контрольные точки

| Неделя | Критерий готовности |
|---|---|
| 3 | Градиентный спуск на NumPy без циклов, понятна cross-entropy |
| 7 | Kaggle-пайплайн с честной валидацией в публичном репозитории |
| 10 | micrograd написан без подсказок, backprop через BatchNorm посчитан руками |
| 14 | Своя обученная модель с демо на Spaces |
| 17 | Свой GPT обучен и объяснён по слоям |
| 20 | Дообученная LLM работает локально в GGUF и в публичном демо |
| 23 | DDPM или image LoRA готовы, понятна разница DDPM и flow matching |
| 24 | 4 проекта, сертификат Hugging Face Agents, выбран облачный экзамен |

---

## Чек-лист через 6 месяцев

Вы сможете:

- объяснить attention, residual, LayerNorm и RMSNorm, и зачем нужен RoPE;
- написать и обучить маленький GPT;
- дообучить открытую LLM на своих данных и выложить Space;
- запустить модель локально в квантизации;
- собрать RAG и измерить его качество, а не оценивать «на глаз»;
- обучить простой diffusion и отличить DDPM от flow matching;
- читать релиз модели и отделять маркетинг от архитектуры.

За полгода с ноутбука не получится:

- натренировать конкурента GPT-4 или Gemini;
- изобрести новую архитектуру;
- стать research scientist без математики уровня ШАД или магистратуры.

Это нормально. Цель плана — инженерная грамотность.

## Как не сломаться

- 10–12 часов в неделю в течение 6 месяцев лучше, чем марафон по 8 часов в день одну неделю.
- Каждую неделю — код, который запускается, а не только просмотр лекций.
- Застряли — сначала воспроизведите репозиторий 1-в-1, потом меняйте одну деталь.
- Отстали на неделю — не догоняйте, а сдвиньте план. Пропускать можно Д5, но не Д6.
- Русские курсы (DLS) дают структуру и домашки. Карпаты дают понимание того, из чего всё собрано. Hugging Face даёт то, чем пользуются в проде.

---

## Сертификаты 2026

Сертификат — фильтр для HR и тай-брейкер на собеседовании. Код, метрики и задеплоенная модель важнее стопки бейджей. Оптимально **2–3 штуки**:

1. учебный сигнал (DeepLearning.AI, Hugging Face или DLS);
2. один облачный экзамен под целевые вакансии;
3. по желанию — узкий GenAI или агенты, если цель AI Engineer, а не research.

| Фаза | Что брать | Зачем |
|---|---|---|
| 1. Классический ML | [Machine Learning Specialization](#r-cert-ml) | Самый узнаваемый сигнал «я знаю ML» |
| 2. DL и PyTorch | [PyTorch Professional Certificate или сертификат DLS](#r-cert-dl) | PyTorch — основной язык вакансий |
| 3. LLM и агенты | [Hugging Face LLM Course и Agents Course](#r-cert-hf) | Бесплатно и ближе к работе 2026 года |
| После плана | [Один облачный](#r-cert-cloud): Google PMLE, AWS MLA или Azure AI-102 / AI-103 | Экзамены, которые HR крупных компаний реально ищут |

### Облачные экзамены

Прокторинг, платный экзамен, срок действия 2–3 года. Сдавайте после плана, когда на этой платформе уже есть проект. Готовка — 4–8 недель.

- **Google Professional Machine Learning Engineer** — около $200, примерно 2 часа. Самый инженерный из тройки: Vertex AI, пайплайны, мониторинг, ответственный AI.
- **AWS Certified Machine Learning Engineer – Associate.** Английская версия MLA-C01 принимается до **28 сентября 2026**. С 29 сентября идёт бета **MLA-C02** (код ME1-C02, 85 вопросов, 170 минут, $75); общий релиз запланирован на 14 января 2027. C02 добавляет генеративный AI, Bedrock, RAG и агентов. Если начинаете план сейчас, готовьтесь к C02. Вход для тех, у кого ещё нет облака: AWS Certified AI Practitioner (AIF-C01), $100.
- **Microsoft:** вход — AI-900, рабочий — Azure AI Engineer Associate (AI-102, около $165). В 2026 линейка расширяется: AI-103 (Apps and Agents) и AI-300 (MLOps). Имеет смысл в корпорациях на стеке Microsoft.

**Правило:** сертификат окупается на той платформе, которую используют целевые вакансии. У AWS больше объявлений, GCP даёт более сильный сигнал именно для ML-инженера, Azure популярен в банках и энтерпрайзе. Не собирайте все три.

### Узкие сертификаты (один, не пять)

| Сертификат | Когда нужен |
|---|---|
| NVIDIA-Certified Associate: Generative AI LLMs (около $125) | Локальный инференс, GPU, LLM-инфраструктура |
| Databricks Certified Generative AI Engineer Associate (около $200) | RAG, векторный поиск, озёра данных |
| DeepLearning.AI MLOps Specialization | Цель — пайплайн в прод, а не сама модель |
| IAPP AIGP | Governance и compliance, не research |
| Yandex Cloud | Локальный рынок Яндекса и облаков СНГ |

### Минимальный стек под цель

- **Собирать нейросети и проходить ML/DL-собеседования:** Machine Learning Specialization → PyTorch Professional Certificate или DLS → Hugging Face Agents и публичные модели → Google PMLE или AWS MLA.
- **AI Engineer и LLM-приложения:** Hugging Face LLM и Agents → короткий курс DeepLearning.AI по агентам → AWS AI Practitioner, затем AWS MLA-C02 или Azure AI-103.
- **Локальный рынок РФ и Казахстана, бумага для HR:** DLS, Практикум или вузовское ДПО → один облачный (Yandex Cloud или AWS, смотрите вакансии). Портфолио важнее обоих.

**Не берите:** десяток Udemy Certificate of Completion; «нейросети за 2 недели без кода»; закрытый TensorFlow Developer; второй и третий облачный экзамен, пока нет проекта на первом.

### Стоимость

- Фазы 0–2: 0 ₽, кроме необязательной подписки Coursera на 1–2 месяца ради сертификата.
- Фаза 3: Hugging Face бесплатно, GPU — бесплатные квоты Kaggle и Colab.
- После плана: облачный экзамен $75–300.

В резюме не «10 сертификатов», а так:

> Machine Learning Specialization (DeepLearning.AI) · Hugging Face Agents Course · AWS MLA
> Проекты: GPT с нуля · LoRA на своих данных · RAG с eval-набором · Space с демо

Сертификат открывает дверь. На собеседовании спрашивают код, данные, метрики и то, как модель ломается в проде.

---

## Короткая карта ресурсов

| Тема | Главный ресурс |
|---|---|
| Линейная алгебра и матанализ | [3Blue1Brown](#r-3b1b-la) + Vert Dider |
| Вероятность и статистика | [StatQuest](#r-statquest), [Mathematics for ML](#r-mml) |
| Python и NumPy | [selfedu](#r-selfedu) + [Kaggle Learn](#r-kagglelearn) |
| Классический ML | [Andrew Ng](#r-ng) + [DLS МФТИ](#r-dls) |
| PyTorch, CNN, RNN | [DLS](#r-dls) + [fast.ai](#r-fastai) + [ФКН ВШЭ](#r-hse) |
| Backprop и GPT с нуля | [Karpathy Zero to Hero](#r-zth), [nanoGPT](#r-nanogpt), [nanochat](#r-nanochat) |
| LLM в проде | [Hugging Face LLM Course](#r-hfllm), [PEFT и Unsloth](#r-peft) |
| Локальный запуск | [Ollama, llama.cpp, vLLM](#r-local) |
| RAG и агенты | [HF Cookbook](#r-rag), [HF Agents Course](#r-hfagents) |
| Diffusion | [HF Diffusion Course](#r-hfdiff), [Stanford CME296](#r-cme296), [ComfyUI](#r-comfy) |
| Облачный экзамен | [Google PMLE, AWS MLA, Azure AI-102](#r-cert-cloud) |
