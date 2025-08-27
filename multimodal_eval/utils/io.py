import os
from io import BytesIO
from urllib.parse import urlparse, unquote
from PIL import Image

def load_image_any(src: str) -> Image.Image:
    """
    Opens an image from http(s)://, file://, or a regular local path.
    Always returns RGB.
    """
    if not src:
        raise ValueError("Empty image source")

    if os.path.exists(src):
        return Image.open(src).convert("RGB")

    p = urlparse(str(src))
    scheme = (p.scheme or "").lower()

    if scheme in ("http", "https"):
        import requests
        r = requests.get(src, timeout=15)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")

    if scheme == "file":
        # file:///Users/me/a.png → /Users/me/a.png
        path = unquote(p.path)
        return Image.open(path).convert("RGB")

    # fallback
    return Image.open(os.path.abspath(src)).convert("RGB")
