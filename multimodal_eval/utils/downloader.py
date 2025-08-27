import requests
import tempfile

def download_image_to_tmp(url: str) -> str:
    """
    Downloads an image from the given URL into a temporary .jpg file and returns its path.

    :param url: Image URL
    :return: Path to the temporary image file
    :raises ValueError: if the image could not be downloaded
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from {url}")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name
