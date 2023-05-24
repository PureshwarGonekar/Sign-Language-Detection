import streamlit as st
import numpy as np
import random as ran
import pandas as pd
import keras
from PIL import Image
from skimage.transform import rescale
path =''
test = pd.read_csv(path+'test.csv')
test_X = test.drop(['label'],axis = 1)
test_X = test_X.astype('float32')/255
test_X = test_X.values.reshape((7172, 28, 28, 1))

test_Y = test['label']
test_Y = keras.utils.to_categorical(test_Y,26)

model = keras.models.load_model('live_cnn_model.h5')
r=1

def run():
    r= ran.randint(0,500)
    rimg = test_X[r]
    showimage = rimg.reshape((28,28))
    showimage = np.asarray(showimage)
    st.subheader('Testing image(data) : ')
    st.image(showimage)     
    test_img = rimg.reshape((1,28,28,1))
    predict_x=model.predict(test_img) 
    img_class=np.argmax(predict_x,axis=1)
    classname = img_class[0]
    return classname
        
def show_predict_page():
    st.title("Sign Language Detection Model")

    st.write("""## Jabalpur Engineering College, Jabalpur​""")
    st.write("""### Computer Science Engineering """)
    st.write("""### 6th Semester""")
    st.write("""### Team- G6""")
    st.markdown("Team Members : :green[Pureshwar Gonekar] | :green[Ishani Malviya] | :green[Granendra Pratap Singh] | :green[Gaurav Kumar]")
    st.markdown("Project Guide : **:blue[Prof. M Gokhale]** ")
  
    imageshowbutton = st.button("Generate the Random Image")
    if imageshowbutton:
        classname=run()
        # if st.button("Predict the output"):
        st.write('Generating Result....')
        signlabels='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        st.write('## Output : :green[{}] Alphabet'.format(signlabels[classname]))
        accuracy = model.evaluate(x=test_X,y=test_Y,batch_size=32)
        st.subheader(f"Accuracy : {round(accuracy[1]*100, 2)}%")

def show_explore_page():
    st.title("Sign Language Detection Model")
    st.write("### American Sign Language Alphabets (except J and Z)!!")
    docimg = np.asarray(Image.open(path+'amer_sign2.png'))
    st.image(docimg)

page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()

