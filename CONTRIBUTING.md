# Contributing Guide 

Thank you for your interest in contributing to **MultimodalEval**!  
This project is designed to advance evaluation standards for multimodal AI models.  
We welcome contributions from developers, researchers, and practitioners.

---

## 📂 Project Structure

- **`multimodal_eval_core/`** — core framework modules  
- **`configs/`** — task and model configuration files  
- **`datasets/`** — dataset definitions (labeled and unlabeled)  
- **`results/`** — evaluation outputs and logs  
- **`app_ui/`** — Streamlit UI components  
- **`docs/`** — documentation files  

---

## 🛠 How to Contribute

### 1. Reporting Issues
- Use the GitHub Issues page.  
- Provide a clear description of the bug, steps to reproduce, and expected behavior.  
- If relevant, attach dataset samples or logs.

### 2. Suggesting Features
- Open a **feature request** issue.  
- Describe the use case and why it benefits the community.  
- We prioritize features that align with the roadmap (robustness, red teaming, new metrics).

### 3. Submitting Code
1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/new-metric
   ```
3. Follow coding standards:
   - Python ≥3.9
   - Use PEP8 conventions
   - Add type hints and docstrings
4. Add tests if applicable.  
5. Run the test suite before pushing:  
   ```bash
   pytest
   ```
6. Submit a pull request with a clear description.

---

## 🎯 Adding New Metrics / Tasks

- **Metrics** → extend `metrics/` module and update `tasks_config.py`.  
- **Tasks** → define schema in `evaluation/schema.py`, add paths in `tasks_config.py`, and update documentation in `docs/tasks.md`.  
- Ensure all new metrics and tasks are covered with at least one test case.

---

## 🧩 Coding Standards

- **Linting**:  
  ```bash
  black .
  flake8 .
  ```
- **Type checking**:  
  ```bash
  mypy multimodal_eval/
  ```
- **Testing**:  
  Use `pytest` and place tests under `tests/`.

---

## 📖 Documentation

- All new features must include corresponding documentation in `docs/`.  
- Examples should be added to `examples/` (Python scripts or Jupyter notebooks).  
- If you update configs, update `configuration.md` accordingly.

---

## 🎯 Contribution Philosophy

We value:
- **Clarity** → code should be easy to understand.  
- **Reproducibility** → results should be deterministic given the same inputs.  
- **Robustness** → handle edge cases gracefully.  
- **Collaboration** → PRs should be constructive and documented.  

---

## 📬 Contact

- GitHub: [rybalena](https://github.com/rybalena) 
- LinkedIn: [Rybina Elena](https://www.linkedin.com/in/elena-rybina-5222bb118/) 

---

Thank you for helping make **MultimodalEval** an industry-grade evaluation framework 🚀
