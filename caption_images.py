import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ext.BLIP.models.blip import blip_decoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)

    return parser.parse_args()


def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def caption_image_dir(input_dir, output_dir):
    image_size = 384

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    output_dir.mkdir(exist_ok=True)

    captions = []

    with torch.no_grad():
        img_paths = list(input_dir.glob("**/*.png")) # Assume PNG for now.

        for img_path in tqdm(img_paths, total=len(img_paths)):
            try:
                img = load_image(img_path, image_size, device)
            except:
                continue

            # beam search
            caption = model.generate(img, sample=False, num_beams=3, max_length=20, min_length=5) 
            # nucleus sampling
            # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
            # print('\ncaption: '+caption[0])

            captions.append(
                {"file_name": img_path.stem + ".png", "additional_feature": caption[0]}
            )

    # Save annotation file.
    with open(output_dir / "captions.jsonl", "w") as outfile:
        for entry in captions:
            json.dump(entry, outfile)
            outfile.write("\n")

        


def main(args):
    caption_image_dir(args.input_dir, args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)