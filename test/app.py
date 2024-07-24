import streamlit as st 
import tensorflow as tf from tensorflow.keras.applications.imagenet_utils
import decode_predictions 
import cv2 from PIL
import Image, ImageOps
import numpy as np
import torchvision
import torch
from .LTP import M_GCN
import torchvision.transforms as transforms
@st.cache(allow_output_mutation=True)

def pred2name(predlist):
    # voclabel = ["areoplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    voclabel = ['baking powder', 'basmati rice', 'bitter gourd', 'black lentils', 'black rice', 'bread', 'bread crumbs', 'butter', 'capsicum', 'cardamom', 'carrots', 'cauliflower', 'chicken', 'chickpea flour', 'chickpeas', 'coconut', 'cornmeal', 'cottage cheese', 'cream', 'fenugreek leaves', 'fish', 'flour', 'ghee', 'gram flour', 'green peas', 'jaggery', 'khoya', 'kidney beans', 'lentils', 'milk', 'mustard greens', 'mustard oil', 'nuts', 'oil', 'okra', 'onions', 'paneer', 'peanuts', 'peas', 'poppy seeds', 'potatoes', 'rice', 'rice flour', 'roasted gram flour', 'saffron', 'salt', 'sesame seeds', 'spices', 'sugar', 'tomatoes', 'urad dal', 'vegetables', 'water', 'wheat flour', 'whole wheat flour', 'yellow lentils', 'yogurt']
    final_class = []
    for i in range(len(predlist)):
        if predlist[i]>0:
            final_class.append(voclabel[i])
    if len(final_class)==0:
        print("No Class Detected !")
    return final_class

def get_model(num_classes, args):
    model_dict = {'M_GCN': M_GCN}
    res101 = torchvision.models.resnet101(pretrained=True)
    model = model_dict[args.model_name](res101, num_classes)
    return model

class Args:
    def __init__(self, model_name= "M_GCN", num_class = 57, resume = './checkpoint/checkpoint_best.pth', y = 0.5, image_size = 224): 
        # Instance Variable     
        self.model_name = model_name
        self.num_class = num_class
        self.y = y
        self.resume = resume
        self.image_size = image_size

args = Args()


with st.spinner('Model is being loaded..'): 
    model=get_model(args.num_class,args)

st.write(""" # Image Classification """ ) 
file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"]) 
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(upload_image, model, args):
    print("* Loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model_dict = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
        else:
            print('\tMismatched layers: {}'.format(k))
    model.load_state_dict(model_dict)

    image = Image.open(upload_image).convert('RGB')
    resize = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
    pred_class=pred2name(predlist)
    return pred_class 

if file is None: 
    st.text("Please upload an Indian Dish") 
else: 
    image = Image.open(file) 
    st.image(image, use_column_width=True) 
    predictions = upload_predict(image,model,args) 
    image_class = ",".join(predictions)
    # score=np.round(predictions[0][0][2]) 
    st.write("This dish has these ingredients:: ",image_class) 
    # st.write("The similarity score is approximately",score)
    # print("The image is classified as ",image_class, "with a similarity score of",score)
