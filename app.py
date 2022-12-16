import os
import uuid
import urllib
import numpy as np
import pandas as pd
import cv2
from skimage import io
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
# from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , jsonify
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model = load_model(os.path.join(BASE_DIR , 'my_model.hdf5'))
model =  hub.KerasLayer('aiy_vision_classifier_food_V1_1')
UPLOAD_FOLDER= os.path.join(os.getcwd() , 'static/uploalds')


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

classes =['pizza','idly','dosa','burger','briyani']


def predict(filename , model):
    img = load_img(filename , target_size = (150 , 150))
    img = img_to_array(img)
    img = img.reshape(1 , 150 ,150 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(5):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result

def tf_predict(img_path):
    labelmap_url = "aiy_food_V1_labelmap.csv"
    input_shape = (224, 224)

    image = np.asarray(io.imread(img_path), dtype="float")
    image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
    # Scale values to [0, 1].
    image = image / image.max()
    # The model expects an input of (?, 224, 224, 3).
    images = np.expand_dims(image, 0)
    # This assumes you're using TF2.
    output = model(images)
    # predicted_index = output.numpy().argmax()
    top = tf.nn.top_k(input=output.numpy(),k=4,name=None)
    print("Prediction: {} ".format(top.indices[0][1]))
    print("Prediction: {} ".format((top.values[0][1].numpy()*100).round(2)))
    classes = list(pd.read_csv(labelmap_url)["name"])
    predictions ={
      classes[top.indices[0][0].numpy()]:(top.values[0][0].numpy()*100).round(2),
      classes[top.indices[0][1].numpy()]:(top.values[0][1].numpy()*100).round(2),
      classes[top.indices[0][2].numpy()]:(top.values[0][2].numpy()*100).round(2),
      classes[top.indices[0][3].numpy()]:(top.values[0][3].numpy()*100).round(2)
    }

    return predictions




@app.route('/upload',methods = [ 'POST'])
def upload_file():
    if 'recipe' not in request.files:
        resp = jsonify({'message':'No file selected found.'})
        resp.status_code = 400
        return resp
    file = request.files.getlist('recipe')[0]
    if file and allowed_file(file.filename) :
        unique_filename = str(uuid.uuid4())
        filename = unique_filename+".jpg"
        img_path = os.path.join(UPLOAD_FOLDER , filename)
        file.save(img_path)
        # class_result , prob_result = predict(img_path , model)
        result = tf_predict(img_path )
        os.remove(img_path)
        # predictions = {
        #               prob_result[0]:class_result[0],
        #                 prob_result[1]:class_result[1],
        #                 prob_result[2]:class_result[2]
        #         }
        print(result)
        resp = jsonify(result)
        resp.status_code = 200
        return resp
    else:
        resp = jsonify({'error':file.filename+'File type is not allowed.'})
        resp.status_code = 300
        return resp





def home():
        return render_template("index.html")
@app.route('/',)

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)


