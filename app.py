import flask
from flask import request, redirect, render_template
import os
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

# initialize our Flask application and the Keras model
app.config["IMAGE_UPLOADS"] = "./uploads"

@app.route("/upload-image", methods=["GET", "POST"])
# def hello_world():
#     return "Hello world!"
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            print("Image saved")
            print(image.filename)
            print(type(image))
            # for model
            predictions = predict(image)
            print('predict done')
            return render_template("home_upgrade.html", 
                                   label='Image classified as: {}'.format(predictions['predictions'][0]['label']),
                                   prob = 'with probability of: {0:.1f}%'.format(predictions['predictions'][0]['probability']*100))
            #return redirect(request.url)
    return render_template("home_upgrade.html", label='No predictions yet.')

#Specific for model

def load_model():
 	# load the pre-trained Keras model
 	global model
 	model = ResNet50(weights="imagenet")
 	global graph
 	graph = tf.get_default_graph()

def prepare_image(image, target):
 	# if the image mode is not RGB, convert it
 	if image.mode != "RGB":
         image = image.convert("RGB")

 	# resize the input image and preprocess it
 	image = image.resize(target)
 	image = img_to_array(image)
 	image = np.expand_dims(image, axis=0)
 	image = imagenet_utils.preprocess_input(image)

 	# return the processed image
 	return image
 
def predict(image):
    data = {"success": False}
    byteImgIO = io.BytesIO()
    byteImg = Image.open("./uploads/" + image.filename)
    byteImg.save(byteImgIO, byteImg.format)
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()
    image = Image.open(io.BytesIO(byteImg))
    
   	# preprocess the image and prepare it for classification
    image = prepare_image(image, target=(224, 224))

    # classify the input image and then initialize the list
    # of predictions to return to the client
    with graph.as_default():
    
        preds = model.predict(image)
    
        results = imagenet_utils.decode_predictions(preds)
    
        data["predictions"] = []
    
     			 	# loop over the results and add them to the list of
     			 	# returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)
    
        # indicate that the request was a success
        data["success"] = True
        print(data)
        
    # return the data dictionary as a JSON response
    return data


if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(app.run(host= '0.0.0.0'), debug=False)
    