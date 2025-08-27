import pytest
from pathlib import Path
from multimodal_eval.utils.downloader import download_image_to_tmp

TEST_IMAGE_URLS = [
    "https://huggingface.co/datasets/rybalena/aiqa_images/resolve/main/captioning/boy_playing_football.jpg",
    "https://huggingface.co/datasets/rybalena/aiqa_images/resolve/main/captioning/mountains_and_lake.jpg"
]

@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
def test_download_hf_images(image_url):

    path = download_image_to_tmp(image_url)
    path = Path(path)

    assert path.exists(), f"Downloaded file not found: {path}"
    assert path.suffix.lower() in [".jpg", ".jpeg", ".png"], f"Unexpected file extension: {path.suffix}"
    assert path.stat().st_size > 0, "Downloaded file is empty"

@pytest.mark.parametrize("bad_url", [
    "https://huggingface.co/datasets/rybalena/aiqa_images/resolve/main/nonexistent_image.jpg",
    "https://this-domain-does-not-exist.aiqa/fake.jpg"
])
def test_download_invalid_urls(bad_url):

    with pytest.raises(Exception):
        download_image_to_tmp(bad_url)
