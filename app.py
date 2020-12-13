"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes ffwd_to_img function from evaluate.py with this image
    - Returns the output file generated at /output

Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: checkpoint
    - It is loaded from /input
"""
import os
from flask import Flask, send_file, request, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from flask_cors import CORS

import random

import base64

import ast

from generate import wrapped_generate, wrapped_generate2, wrapped_generate2p

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
app = Flask(__name__)

CORS(app)


counter = 0


@app.route('/', methods=["GET"])
def random_get2():
    """
    return random image in output
    """
    # check if the post request has the file part

    output_images = os.listdir('output/')
    if len(output_images) < 0:
        return BadRequest("No file would be returned")

    output_filepath = os.path.join('output/', output_images[random.randrange(0, len(output_images))])
    return send_file(output_filepath, mimetype='image/jpg')


@app.route('/generate', methods=["POST"])
def random_generate():
    """
    Take the input image and style transfer it
    """
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(filename):
        return BadRequest("Invalid file type")

    input_filepath = os.path.join('./images/', filename)
    output_filepath = os.path.join('/output/', filename)
    input_file.save(input_filepath)

    # Get checkpoint filename from la_muse
    # checkpoint = request.form.get("checkpoint", "la_muse.ckpt")
    # generate(input_filepath, output_filepath, '/input/' + checkpoint, '/gpu:0')
    return send_file(output_filepath, mimetype='image/jpg')


@app.route('/inverse', methods=["POST"])
def inverse_generate():
    """
    Take the input image and style transfer it
    """
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(filename):
        return BadRequest("Invalid file type")

    input_filepath = os.path.join('./images/', filename)
    output_filepath = os.path.join('/output/', filename)
    input_file.save(input_filepath)

    # Get checkpoint filename from la_muse
    # checkpoint = request.form.get("checkpoint", "la_muse.ckpt")
    # generate(input_filepath, output_filepath, '/input/' + checkpoint, '/gpu:0')
    return send_file(output_filepath, mimetype='image/jpg')


@app.route('/random_get', methods=["POST"])
def random_get():
    global counter
    """
    return random image in output
    """
    # check if the post request has the file part

    posts = ast.literal_eval(request.get_data().decode('utf-8'))
    if (not 'num' in posts) or int(posts['num']) < 1:
        return BadRequest("Impliced Number")

    rets = {'count' : counter,
            'nums' : posts['num'],
            'img' : []}

    output_images = os.listdir('output/')
    if len(output_images) < 0:
        return BadRequest("No file would be returned")
    
    for _ in range(int(posts['num'])):
        output_filepath = os.path.join('output/', output_images[random.randrange(0, len(output_images))])

        with open(output_filepath, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        rets['img'].append(img_base64)

    counter += 1

    print(rets)

    return jsonify(rets)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
