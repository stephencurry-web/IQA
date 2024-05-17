import os

import torchvision.transforms as T
from PIL import Image

SPAQ = "/data/qgy/IQA-Dataset/SPAQ/SPAQ zip/TestImage"
preprocess = T.Compose([T.Resize((512, 384))])
inspaqfolder = os.listdir(SPAQ)
for _, item in enumerate(inspaqfolder):
    img = Image.open(imgpath := os.path.join(SPAQ, item))

    resized_img = preprocess(img)
    resized_img.save(resized_path := (imgpath.replace("TestImage", "512x384")))
    print(resized_path)
