# Neural Networks Roadmap

A 30-week, day-by-day plan for 2026 (in Russian), starting 28 September 2026. It covers math and Python, classical ML, RAG and agents, then the AWS Certified Machine Learning Engineer – Associate exam (MLA-C02) in week 13, while the $75 beta runs, with two spare weeks before general availability on 14 January 2027. After the exam come deep learning with PyTorch, transformers, LLM fine-tuning, local inference, and diffusion, with four portfolio projects.

- **Plan:** [plan.md](plan.md)
- **Web page with progress tracking:** https://alexander-shamray.github.io/neural-networks-plan/

The page renders `plan.md` and adds a checkbox to every day. Each day is a list of resources. A resource name opens its full description from the "Ресурсы" catalog in a popup. Progress is stored in your browser's `localStorage`, so it stays on that device and browser only.

To move progress between devices, use the header buttons:

- **⤓ Export** downloads `nn-roadmap-progress-YYYY-MM-DD.json`.
- **⤒ Import** merges a file into the current progress. Days checked in either place stay checked. Import never unchecks a day, so to uncheck one, do it on every device.

## Run locally

```sh
python -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` straight from disk does not work, because the page fetches `plan.md`.
