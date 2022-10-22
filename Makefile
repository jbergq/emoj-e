download-slackmojis:
	docker build -t slackmojis . -f containers/slackmojis/Dockerfile && docker run -v $$(pwd):/workspace slackmojis
caption-images:
	docker build -t blip_image . -f containers/blip/Dockerfile && docker run -v $$(pwd):/workspace --gpus 1 blip_image
build-train:
	docker build -t stable_confusion_image . -f containers/stable_diffusion/Dockerfile
run-train:
	docker run -itd -v $$(pwd):/workspace --gpus all --name stable_confusion_container stable_confusion_image
