import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

app = Flask(__name__)



# model.make_predict_function()


def predict_label(img_path):
    model = load_model('models\chest_xray_resnet.h5')
    image = load_img(img_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Expand dimentions only for vgg
    # image = np.expand_dims(image, axis=0)
    # prepare the image for the VGG model only
    # image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    result = int(yhat[0][1])
    return result


# Flask
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)
        if p == 0:
            pr = "Person is Affected By PNEUMONIA"
        else:
            pr = "Person is Normal"

    return render_template("index.html", prediction=pr, img_path=img_path)


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)