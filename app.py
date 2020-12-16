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
import os.path as osp
import torch
from model import Generator
from tqdm import tqdm

from PIL import Image
import numpy as np

import time

import hashlib

from queue import Queue

import random

import functools

import os

from flask import Flask, send_file, request, jsonify
from flask_cors import CORS

import base64

import ast
import subprocess
import json

import gc


from generate import generate, generate1, generate2p

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
app = Flask(__name__)

CORS(app)
# run_with_ngrok(app)  # Start ngrok when app is run

# 多重実行用のキュー
multipleQueue = Queue(maxsize=2)
singleQueue = Queue(maxsize=3)

# ★ポイント２
# 多重制御の関数デコレータ
def multiple_control(q):
    def _multiple_control(func):
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            q.put(time.time())
            print("/// [start] critial zone")
            result = func(*args,**kwargs)
            print("/// [end] critial zone")
            q.get()
            q.task_done()
            return result

        return wrapper
    return _multiple_control

# ★ポイント３
# you should limit execute this function
def heavy_process(data):
    print("<business> : " + data)
    time.sleep(10)
    return "ok : " + data

counter = 0

device = 'cuda'

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

g_ema = Generator(
    128, 512, 8, channel_multiplier=2
    ).to(device)
checkpoint = torch.load(osp.join('models', '050000.pt'))
g_ema.load_state_dict(checkpoint["g_ema"])
mean_latent = None


@app.route("/wakeup_test")
def hello():
    return "world"


@app.route("/nvidia-smi")
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    rets = [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
    
    return str(rets).replace(', ', ', <br/>')


@app.route('/generate', methods=["GET"])
def random_generate():
    if request.args.get('hash') is not None:
        hash_ = request.args.get('hash')
    else:
        hash_ = str(hashlib.sha1(time.time()).hexdigest())

    filename = f"{hash_}.png"

    output_filepath = os.path.join('output/', filename)
    generate1(g_ema, device, mean_latent, outputname=output_filepath)

    gc.collect()
    torch.cuda.empty_cache

    return send_file(output_filepath, mimetype='image/jpg')


@app.route('/generate_gif', methods=["GET"])
@multiple_control(singleQueue)
def random_generate_gif():
    if request.args.get('hash') is not None:
        hash_ = request.args.get('hash')
    else:
        hash_ = str(hashlib.sha1(time.time()).hexdigest())

    filename = f"{hash_}.gif"

    output_filepath = os.path.join('output/', filename)
    generate2p(g_ema, device, mean_latent, torch.randn(512, device=device), 0.25, 32, filename=output_filepath)

    gc.collect()
    torch.cuda.empty_cache

    return send_file(output_filepath, mimetype='image/jpg')

@app.route('/generate_gif2', methods=["GET"])
@multiple_control(singleQueue)
def random_generate_gif2():
    if request.args.get('hash') is not None:
        hash_ = request.args.get('hash')
    else:
        hash_ = str(hashlib.sha1(time.time()).hexdigest())

    filename = f"{hash_}.gif"

    rets = {'latent' : None,
            'img' : None}

    output_filepath = os.path.join('output/', filename)
    center_latent = torch.randn(512, device=device)
    generate2p(g_ema, device, mean_latent, center_latent, 0.25, 32, filename=output_filepath)

    rets['latent'] = center_latent.cpu().numpy().tolist()

    with open(output_filepath, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    rets['img'] = img_base64

    return jsonify(rets)

@app.route('/generate_gif3', methods=["POST"])
@multiple_control(singleQueue)
def random_generate_gif3():
    if request.args.get('hash') is not None:
        hash_ = request.args.get('hash')
    else:
        hash_ = str(hashlib.sha1(time.time()).hexdigest())

    if request.form.get('latent') is not None:
        latent = request.form.get('latent').split(',')
        if len(latent) < 512:
            latent = [random.random() for _ in range(512)]
        latent = [float(l) for l in latent]
    else:
        latent = [random.random() for _ in range(512)]

    if request.form.get('r1') is not None:
        r1 = float(request.form.get('r1'))
    else:
        r1 = 0.125

    if request.form.get('r2') is not None:
        r2 = float(request.form.get('r2'))
    else:
        r2 = 0.25

    filename = f"{hash_}.gif"

    rets = {'latent' : None,
            'img' : None}

    latent2 = torch.FloatTensor(latent).to(device)

    output_filepath = os.path.join('output/', filename)
    center_latent = latent2 + r1*torch.randn(512, device=device)
    generate2p(g_ema, device, mean_latent, center_latent, r2, 32, filename=output_filepath)

    rets['latent'] = center_latent.cpu().numpy().tolist()

    with open(output_filepath, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    rets['img'] = img_base64

    return jsonify(rets)


@app.route('/generate', methods=["POST"])
def random_get_num():
    global counter
    """
    return random image in output
    """
    # check if the post request has the file part

    posts = request.get_data().decode('utf-8')
    key_value = posts.split('=')
    posts = {key_value[0] : key_value[1]}
    app.logger.debug(posts)
    if (not 'num' in posts) or int(posts['num']) < 1 or int(posts['num']) > 50:
        return BadRequest("Impliced Number")

    app.logger.debug(f"{int(posts['num'])}")

    rets = {'count' : counter,
            'nums' : posts['num'],
            'img' : []}
    
    for _ in range(int(posts['num'])):
        filename = f"{counter}.png"
        output_filepath = os.path.join('output', filename)
        generate1(g_ema, device, mean_latent, outputname=output_filepath)

        with open(output_filepath, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        rets['img'].append(img_base64)

        counter = (counter + 1) % 100

    return jsonify(rets)

@app.route('/generate_gif', methods=["POST"])
def random_get_num_gif():
    global counter
    """
    return random image in output
    """
    # check if the post request has the file part

    posts = request.get_data().decode('utf-8')
    if (not 'num' in posts) or int(posts['num']) < 1 or int(posts['num']) > 50:
        return BadRequest("Impliced Number")

    rets = {'count' : counter,
            'nums' : posts['num'],
            'img' : []}
    
    for _ in range(int(posts['num'])):
        filename = f"{counter}.png"
        output_filepath = os.path.join('output', filename)
        generate2p(g_ema, device, mean_latent, torch.randn(512, device=device), 0.25, 16, filename=output_filepath)

        with open(output_filepath, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        rets['img'].append(img_base64)

        counter = (counter + 1) % 100

    return jsonify(rets)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
