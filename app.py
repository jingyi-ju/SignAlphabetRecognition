import numpy as np
from flask import Flask, render_template, request
import pickle
from keras.preprocessing import image


with open('model_pkl', 'rb') as f:
    cnn_model = pickle.load(f)

app = Flask(__name__)


@app.route("/",methods=['GET','POST'])
def main():
    return render_template("index.html")


@app.route('/submit', methods=['GET','POST'])
def get_image():
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y']
    if request.method == 'POST':
        img = request.files['image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = cnn_model.predict(np.reshape(np.asarray(image.load_img(img_path, color_mode="grayscale")), (1, 28, 28, 1)))
        p = np.argmax(p, axis=1)
        index = p[0]
    return render_template("index.html", prediction=classes[index], img_path=img_path)

if __name__ =='__main__':
	app.run()