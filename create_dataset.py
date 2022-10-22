import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def get_files(path, exts):
    files = []
    for ext in exts:
        files.extend(path.glob(f"**/*{ext}"))
    return files


def convert_to_hf_format(img_paths, tgt_dir):
    """Puts images into dataset format required by HuggingFace in target
    directory. Also creates annotation file with image name as metadata.
    """

    # Create annotation list.
    annotations = []

    img_names = defaultdict(lambda: False)

    # Move images.
    for img_path in tqdm(img_paths, total=len(img_paths)):
        if img_path.is_dir():
            continue

        save_name = img_path.stem
        if img_names[save_name]:
            save_name += str(random.randint(1, 10))
        img_names[save_name] = True

        if img_path.suffix in [".png"]:
            shutil.move(img_path, tgt_dir / img_path.name)
        else:
            # Convert to .png with PIL.
            try:
                img = Image.open(img_path).convert("RGB")
                img.save(tgt_dir / f"{img_path.stem}.png")
            except:
                print(f"\nFailed to open image {str(img_path)}")

        annotations.append(
            {"file_name": img_path.stem + ".png", "additional_feature": img_path.stem}
        )

    # Save annotation file.
    with open(tgt_dir / "metadata.jsonl", "w") as outfile:
        for entry in annotations:
            json.dump(entry, outfile)
            outfile.write("\n")


# Hard-coded dataset path for now.
datasets_dir = Path("datasets/slackmojis")

img_dir = datasets_dir / "temp"
tgt_dir = datasets_dir / "train"
tgt_dir.mkdir(exist_ok=True)

# exts = [".png", "gif"]
# img_paths = get_files(img_dir, exts)
img_paths = list(img_dir.glob("**/*"))

convert_to_hf_format(img_paths, tgt_dir)
