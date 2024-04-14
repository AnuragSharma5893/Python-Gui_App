from taipy.gui import Gui
#from tensorflow.python import tf2
#from keras import models
#from tensorflow import models
#from keras import models
from tensorflow.keras import models
from PIL import Image
import numpy as np

class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

model = models.load_model("baseline_mariya.keras")

def predict_image(model, path_to_img):
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data = np.asarray(img)
    #print("before", data[0][0])
    data = data / 255 ## normalize our images
    #print("after", data[0][0])
    probs = model.predict(np.array([data]) [:1])
    #print(probs)
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    #print(model.summary())
    #print(path_to_img)

    return top_prob, top_pred

content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

## Html webpage   ## "# Hello from python!"  ## markdown syntax use "# then space" and write
## image control component
index = """ 
<|text-center|
<|{"logo.png"}|image|width= 25vw|>

<|{content}|file_selector|extensions = .png|>
Select an image from your browser

<|{pred}|>

<|{img_path}|image|>

<|
{prob}|indicator|value={prob}|min=0|max=100|width= 25vw|> 

>
"""        

## updating components on stage change
def on_change(state, var_name,var_val):
    if var_name == "content":
        top_prob, top_pred = predict_image(model,var_val)
        state.prob = round(top_prob * 100)
        state.pred = "this is a " + top_pred
        state.img_path = var_val
    #print(var_name, var_val) 

app = Gui(page=index)

if __name__ == "__main__": 
    app.run(use_reloader=True) ## taipy GUI this will run on the browser 

