# Model Installation Guide

This document provides detailed installation instructions for setting up **MultimodalEval** with different model backends:
- [LLaVA via Ollama](#1-llava-via-ollama)
- [Qwen2-VL-2B-Instruct via Hugging Face Transformers](#2-qwen2-vl-2b-instruct-via-hugging-face-transformers)
- [OpenAI GPT-4o API](#3-openai-gpt-4o-api)

---

## 0. Prerequisites

- Python **3.9+**
- Git
- Virtual environment tool (`venv` or `conda` recommended)

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\\Scripts\\activate     # Windows
```

Install project dependencies:

```bash
pip install -e .
```

---

## 1. LLaVA via Ollama

To use **LLaVA** (a multimodal model), you first need to install **Ollama**.

### Install Ollama
Download Ollama from the official website: [Ollama](https://ollama.com/download)

After installation, verify that it works:
```bash
ollama --version
```

### Download LLaVA model
Once Ollama is installed, pull the desired LLaVA model:

- Recommended (7B):
```bash
ollama pull llava:7b
```
- Larger (14B):
```bash
ollama pull llava:14b
```

### Choosing model size (7B vs 14B)
- **LLaVA-7B** → Recommended for comparison with **Qwen2-VL-2B-Instruct**.  
- **LLaVA-14B** → Stronger baseline for benchmarking against **GPT-4o**, but requires much more VRAM.  

### Keep the model running
When you start the model:
```bash
ollama run llava:7b
```
(or `llava:14b` if using 14B)

**⚠️ Do not close this process** — Ollama must remain running in order for **MultimodalEval** to connect to the LLaVA model.  
Keep the terminal open while you run evaluations.

---

## 2. Qwen2-VL-2B-Instruct via Hugging Face Transformers

[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) is a lightweight multimodal vision-language model.

### Install dependencies
```bash
pip install transformers accelerate
```

### Example usage
```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Example with image + text
image = Image.open("example.jpg")
inputs = processor(text="Describe this image", images=image, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```

⚠️ **Note:**  
- This is a small model (2B parameters). It can run even on CPU or small GPUs.  
- Accuracy is limited compared to **LLaVA-7B** and **GPT-4o**, but it is excellent for lightweight testing.  

---

## 3. OpenAI GPT-4o API

To use GPT-4o, you need an API key.

### Install the library
```bash
pip install openai
```

### Set your API key
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

### Example usage
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello from GPT-4o!"}]
)

print(response.choices[0].message["content"])
```

---

## 4. Verifying Installation

After installing one or more backends, run:

```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled
```

This will execute a sample evaluation with the configured models.

---

## 5. Troubleshooting

- **Ollama not found** → Make sure Ollama is installed and available in `PATH`.  
- **CUDA out of memory** → Try smaller models (Qwen2-VL-2B-Instruct or LLaVA-7B) or run on CPU.  
- **OpenAI API errors** → Check if your API key is set correctly in `.env`.  

---

## Next Steps

- See [Usage Guide](./usage.md) for running evaluations.  
- See [Configuration](./configuration.md) for customizing tasks and datasets.
