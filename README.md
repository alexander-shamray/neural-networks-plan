# Neural Networks Roadmap

A free 24-week, day-by-day plan for 2026 (in Russian): math and Python, classical ML, deep learning with PyTorch, then transformers, LLM fine-tuning, local inference, RAG, agents, and diffusion. It ends with four portfolio projects.

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
