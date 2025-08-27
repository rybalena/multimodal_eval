from __future__ import annotations

import os
import base64
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# 🔐 Load environment variables from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def image_to_base64(image_path: str) -> str:
    """
    Converts an image to a base64-encoded string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_llm_judge(task: str, image_path: str, prompt: str, system_prompt: str) -> Dict[str, Any]:
    """
    Sends a prompt and image to the OpenAI LLM for judging, returns parsed JSON response.
    """
    base64_img = image_to_base64(image_path)

    # OpenAI's vision input format requires a list of message parts for multimodal
    user_prompt = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
    ]

    # Send multimodal request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # Try extracting JSON from markdown-style response
            if "```json" in content:
                json_block = content.split("```json")[-1].split("```")[0].strip()
                return json.loads(json_block)
            elif "```" in content:
                json_block = content.split("```")[-1].split("```")[0].strip()
                return json.loads(json_block)
            else:
                raise
        except Exception as e:
            print(f"❌ JSON parsing failed from OpenAI response:\n{content}")
            raise e
