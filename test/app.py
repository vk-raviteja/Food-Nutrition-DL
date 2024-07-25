import streamlit as st 
import tensorflow as tf
from PIL import Image
import numpy as np
import torchvision
import torch
from models import get_model
import torchvision.transforms as transforms
import google.generativeai as genai
import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBDznwGIj6ohCqsCiogRyG5yY7dNoFJ9s8'
# Set up the Gemini API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


# Function to get nutritional information for a list of ingredients
def get_nutritional_info(ingredients):
    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(m.name)
    #         print(m.description)
    #         print("--"*25)

    model = genai.GenerativeModel('gemini-1.0-pro-latest')

    prompt = f"""
    Let's have a conversation about the nutritional values of some ingredients.
    Here are the ingredients I'm interested in: {', '.join(ingredients)}.

    Could you please tell me the nutritional information of an Indian dish made from these ingredient, including calories, protein, carbohydrates, and fat content ?
    Please respond as a summary paragraph.
    """

    response = model.generate_content(prompt)
    # response = model.generate_content("The opposite of hot is")
    print("Response text :",type(response),  vars(response))
    return response
    # return response.text

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

class Args:
    def __init__(self, model_name= "M_GCN", num_class = 57, resume = 'https://github.com/vk-raviteja/Food-Nutrition-DL/blob/main/test/checkpoint/checkpoint_best.pth', y = 0.5, image_size = 224): 
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
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True) 
    predictions = upload_predict(image,model,args) 
    image_class = ", ".join(predictions)
    # score=np.round(predictions[0][0][2]) 
    st.write("This dish has these ingredients:: ",image_class)
    # Example usage
    nutritional_info = get_nutritional_info(predictions)
    print("CAlorie::", type(nutritional_info))
    st.write("Approximate nutritional profile of this dish:: \n",nutritional_info)
    # st.write("The similarity score is approximately",score)
    # print("The image is classified as ",image_class, "with a similarity score of",score)
