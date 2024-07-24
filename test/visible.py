import os, sys, pdb
import argparse
import torch
import torchvision
from models import get_model
import numpy as np
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Training for M-GCN')
parser.add_argument('--resume', default='./checkpoint/checkpoint_07.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--model_name', type=str, default='M_GCN')
parser.add_argument('--num_class', type=int, default=20)
parser.add_argument('--y', default=0.5, type=float)
parser.add_argument('--image-size', '-i', default=448, type=int)

def pred2name(predlist):
    # voclabel = ["areoplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    voclabel = ['baking powder', 'basmati rice', 'bitter gourd', 'black lentils', 'black rice', 'bread', 'bread crumbs', 'butter', 'capsicum', 'cardamom', 'carrots', 'cauliflower', 'chicken', 'chickpea flour', 'chickpeas', 'coconut', 'cornmeal', 'cottage cheese', 'cream', 'fenugreek leaves', 'fish', 'flour', 'ghee', 'gram flour', 'green peas', 'jaggery', 'khoya', 'kidney beans', 'lentils', 'milk', 'mustard greens', 'mustard oil', 'nuts', 'oil', 'okra', 'onions', 'paneer', 'peanuts', 'peas', 'poppy seeds', 'potatoes', 'rice', 'rice flour', 'roasted gram flour', 'saffron', 'salt', 'sesame seeds', 'spices', 'sugar', 'tomatoes', 'urad dal', 'vegetables', 'water', 'wheat flour', 'whole wheat flour', 'yellow lentils', 'yogurt']
    final_class = []
    for i in range(len(predlist)):
        if predlist[i]>0:
            final_class.append(voclabel[i])
    if len(final_class)==0:
        print("No Class Detected !")
    print(final_class)


def main(args):
    resize = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    num_classes = args.num_class
    model = get_model(num_classes, args)
    print("* Loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model_dict = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
        else:
            print('\tMismatched layers: {}'.format(k))
    model.load_state_dict(model_dict)

    print("Image filename Like image/1.jpg ")

    while True:
        img = input('Input image filename:')
        # img = "image/aloo_matar-18.jpg"
        try:
            image = Image.open(img).convert('RGB')
        except:
            print('Open Error! Try again!')
            continue
        else:
            # image.show()
            with torch.no_grad():
                model.eval()
                inputs = resize(image)
                inputs = torch.unsqueeze(inputs,0)
                outputs1, outputs2 = model(inputs)
                y = args.y
                outputs = (1-y)*outputs1+y*outputs2
            print(outputs)
            pred = np.array(outputs)
            predlist = pred.squeeze(0)
            plt.imshow(image)
            pred2name(predlist)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


