from flask import Flask, render_template, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load MNIST dataset globally
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_mnist_image', methods=['GET'])
def get_mnist_image():
    # Select a random image from the training portion of the MNIST dataset
    random_index = np.random.randint(0, x_train.shape[0])
    image_data = x_train[random_index]

    # Convert the image data to a nested list
    image_data_list = image_data.tolist()

    # Return the image data as a JSON response
    return jsonify({'image': image_data_list})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
