import sys
import requests
from pathlib import Path

from bs4 import BeautifulSoup


def main():
    emojipedia_url = "https://emojipedia.org/smiling-face-with-smiling-eyes/"

    req = requests.get(emojipedia_url)
    soup_response = BeautifulSoup(req.text, "html.parser")
    out_dir = Path("datasets") / "emojipedia"

    try:
        out_dir.mkdir(exist_ok=True)

    except OSError:
        pass  # already exists

    for img in soup_response.find_all("img"):
        src_attr = img.get("src")

        if "/" in src_attr:
            filename = src_attr.rsplit("/", 1)[1].split("?", 1)[0]

        with open(out_dir / filename, "wb") as file:
            print("Downloading %s" % filename)
            response = requests.get(emojipedia_url + src_attr, stream=True)
            file_size = response.headers.get("content-length")

            if file_size is None:  # no content length header
                file.write(response.content)
            else:
                dl = 0
                file_size = int(file_size)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    file.write(data)
                    done = int(50 * dl / file_size)
                    sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                    sys.stdout.flush()


if __name__ == "__main__":
    main()
