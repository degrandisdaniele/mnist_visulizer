from flask import Flask, render_template, jsonify, send_from_directory
import os
import random
from PIL import Image # Import Pillow

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_mnist_image', methods=['GET'])
def get_mnist_image():
    image_filenames = [f for f in os.listdir('public') if f.endswith('.png')]
    if not image_filenames:
        return jsonify({'error': 'No images found in public directory'}), 404

    random_image_filename = random.choice(image_filenames)
    image_path = os.path.join('public', random_image_filename)

    try:
        img = Image.open(image_path).convert('L') # Open image and convert to grayscale
        img_array = list(img.getdata()) # Get pixel data
        # MNIST images are 28x28, so reshape the flat list into a 2D array
        mnist_image_data = [img_array[i:i+28] for i in range(0, len(img_array), 28)]

        return jsonify({'image': mnist_image_data})
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)