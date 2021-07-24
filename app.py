from __future__ import absolute_import
from __future__ import print_function

import gc
import json
import os
import sqlite3
from datetime import datetime
from datetime import timezone
from os import path, makedirs
from shutil import disk_usage

import boto3
import requests
import torch
import werkzeug
from flask import Flask, redirect, request, url_for
from flask import render_template
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from oauthlib.oauth2 import WebApplicationClient
from werkzeug.utils import secure_filename

from S3Handler import download_from_s3
from db import init_db_command
from inference import ALLOWED_EXTENSIONS, predict_melanoma, enet_type, enetv2, device, out_dim
from inference import CLASS_NAMES
from inference import kernel_type
from user import User

# GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", None)
# GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", None)

GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""

GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration"
)

'''Not Changing variables'''
s3_region = 'ap-south-1'
dynamodb_region = 'ap-south-1'
dynamodb_melanoma_tablename = 'oncology-melanoma'
model_bucket = 'oncology-melanoma-ap1'
model_dir = '/home/model/melanoma'

model_list = []

data_bucket = "oncology-melanoma-data-from-radiology-ap1"
data_dir = '/home/endpoint/data/melanoma'

port = 8080
debug = False

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 20 * 4096 * 4096

login_manager = LoginManager()
login_manager.init_app(app)

# Naive database setup
try:
    init_db_command()
except sqlite3.OperationalError:
    pass

client = WebApplicationClient(GOOGLE_CLIENT_ID)


@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    return 'bad request!', 400


@login_manager.unauthorized_handler
def unauthorized():
    return "You must be logged in to access this content.", 403


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()


@app.route('/')
def index():
    # [[0.96211, 'unknown'], [0.02061, 'nevus'], [0.01696, 'BKL'], [0.00011, 'melanoma'], [7e-05, 'AK'], [7e-05, 'DF'],
    #  [5e-05, 'VASC'], [1e-05, 'BCC'], [1e-05, 'SCC']]

    preds_html = [['https://endpoint-app-ap1.s3.ap-south-1.amazonaws.com/webapp/static/img/ISIC_0015719.jpg',
                   'ISIC_0015719.jpg',
                   [[0.96211, 'unknown'], [0.02061, 'nevus'], [0.01696, 'BKL'], [0.00011, 'melanoma'],
                    [7e-05, 'AK'], [7e-05, 'DF'], [5e-05, 'VASC'], [1e-05, 'BCC'], [1e-05, 'SCC']]
                   ]]
    if current_user.is_authenticated:
        return render_template("index.html", user_authenticated=True,
                               preds_html=preds_html, current_user=current_user)
    else:
        return render_template("index.html", user_authenticated=False,
                               preds_html=preds_html, current_user=current_user)


@app.route('/ping', methods=['GET'])
def ping():
    print(f'Found a {request.method} request for prediction. form ping()')
    return redirect(url_for("index"))


@app.route("/login")
def login():
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)


@app.route("/login/callback")
def callback():
    # Get authorization code Google sent back to you
    code = request.args.get("code")

    # Find out what URL to hit to get tokens that allow you to ask for
    # things on behalf of a user
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Prepare and send request to get tokens! Yay tokens!
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code,
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    # Parse the tokens!
    client.parse_request_body_response(json.dumps(token_response.json()))

    # Now that we have tokens (yay) let's find and hit URL
    # from Google that gives you user's profile information,
    # including their Google Profile Image and Email
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    # We want to make sure their email is verified.
    # The user authenticated with Google, authorized our
    # app, and now we've verified their email through Google!
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json()["picture"]
        users_name = userinfo_response.json()["given_name"]
    else:
        return "User email not available or not verified by Google.", 400

    # Create a user in our db with the information provided
    # by Google
    user = User(
        id_=unique_id, name=users_name, email=users_email, profile_pic=picture
    )

    # Doesn't exist? Add to database
    if not User.get(unique_id):
        User.create(unique_id, users_name, users_email, picture)

    # Begin user session by logging the user in
    login_user(user)

    # Send user back to homepage
    return redirect(url_for("index"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


class ClassificationService(object):
    dynamodb_cli = None
    s3_res_bucket = None

    @classmethod
    def IsVerifiedUser(cls, request):
        if request.content_type == 'application/json':
            return True
        else:
            return False

    @classmethod
    def cleanDirectory(cls):
        space_left = disk_usage('/').free / 1e9
        if space_left < 1:
            print(f"{space_left} GB of space left so cleaning {data_dir} dir")
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    os.unlink(os.path.join(root, f))

    @classmethod
    def DynamoDBPutItem(cls, item):
        if cls.dynamodb_cli is None:
            cls.dynamodb_cli = boto3.client('dynamodb', region_name=dynamodb_region)
        res = cls.dynamodb_cli.put_item(TableName=dynamodb_melanoma_tablename, Item=item)

    @classmethod
    def upload_to_s3_(cls, bucket, channel, filepath):  # public=true, if not file won't be visible after prediction
        if cls.s3_res_bucket is None:
            cls.s3_res_bucket = boto3.resource('s3', region_name=s3_region).Bucket(bucket)
        data = open(filepath, "rb")
        key = channel + '/' + str(filepath).split('/')[-1]
        cls.s3_res_bucket.put_object(Key=key, Body=data, ACL='public-read')


def convert_dicom(image_id):
    print('Converting dicom image to png...')

    image_id = image_id.replace('dcm', 'png')
    return image_id


def image_with_name_in_dir(image_id: str):
    im_ex = image_id.rsplit('.', 1)[1].lower()
    for ext in ALLOWED_EXTENSIONS:
        if ext == im_ex:
            break
    if im_ex == 'dcm':
        image_id = convert_dicom(image_id)
    try:
        if path.isfile(image_id):
            return image_id
    except FileNotFoundError(image_id) as e:
        print(e)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def Loading_model_in_memory(model_dir=''):
    model = enetv2(enet_type, n_meta_features=0, out_dim=out_dim)
    for fold in range(5):
        print(f"Loading Model {kernel_type}_best_fold{fold}.pth")
        model_file = path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
        state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        model_list.append(model)
    return model_list


@app.route('/', methods=['POST'])
def transformation():
    gc.collect()  # try to free some memory.
    ClassificationService.cleanDirectory()

    print(f'Found a {request.method} request for prediction...')
    if request.method == "POST":
        image_files = request.files.getlist("files[]")
        if image_files:
            image_locs = []
            print(f'Saving image file')
            for image in image_files:
                if image and allowed_file(image.filename):
                    img_path = os.path.join(data_dir, secure_filename(image.filename))
                    image_locs.append(img_path)
                    image.save(img_path)

            if model_list is None:
                Loading_model_in_memory(model_dir=model_dir)

            dfs_split, PROBS, LOGITS = predict_melanoma(image_locs, model_list=model_list)

            # [[2.55993285e-07 3.70259755e-07 1.58978344e-06 5.99362806e-07
            #   2.24804509e-07 7.07100980e-07 7.22086334e-05 9.99855161e-01 6.89909793e-05]]
            #   [[-26.85477 - 24.823887 - 17.240356 - 21.16139 - 27.319925 - 21.92301      0.72298104  50.634743 - 14.968703]]

            # single_df, LOGITS = ensemble(dfs_split, LOGITS, len=len(image_locs))

            preds_html = []
            invocation_time = datetime.now(tz=timezone.utc).strftime('%y-%m-%d %H:%M:%S')

            for i, prob in enumerate(PROBS):
                probs = []
                for j, prob_ in enumerate(prob):
                    probs.append([round(prob_, 5), CLASS_NAMES[j]])

                # sorting the probs in descending
                for j in range(0, len(CLASS_NAMES)):
                    for j_ in range(0, len(CLASS_NAMES) - j - 1):
                        if probs[j_][0] < probs[j_ + 1][0]:
                            probs[j_], probs[j_ + 1] = probs[j_ + 1], probs[j_]

                print(probs)
                image_id = dfs_split['filepath'][i].rsplit('/', 1)[1]
                img_url = f"https://{data_bucket}.s3.amazonaws.com/image/{image_id}"
                preds_html.append([img_url, image_id, probs])
                item = {
                    'invocation_time': {'S': str(invocation_time)},
                    'image_id': {'S': image_id},
                    # 'user_id': {'S': str(current_user.id)},
                    # 'name': {'S': str(current_user.name)},
                    # 'email': {'S': str(current_user.email)},
                    'img_url': {'S': img_url},
                    'probs': {'S': str(probs)},
                }
                ClassificationService.DynamoDBPutItem(item=item)
                ClassificationService.upload_to_s3_(bucket=data_bucket, channel="image", filepath=image_locs[i])
                return render_template("index.html", user_authenticated=False,
                                       preds_html=preds_html, current_user=current_user)
            gc.collect()
            # return render_template("index.html", user_authenticated=False,
            #                        preds_html=preds_html, current_user=current_user)
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("Initialising app, checking directories and model files...")
    if not path.exists(data_dir):
        makedirs(data_dir, exist_ok=True)
    if not path.exists(model_dir):
        makedirs(model_dir, exist_ok=True)

    if not path.isfile(path.join(model_dir, f'{kernel_type}_best_fold{0}.pth')):
        for fold in range(5):
            checkpoint_fname = f'{kernel_type}_best_fold{fold}.pth'
            download_from_s3(region=s3_region, bucket=model_bucket,
                             s3_filename='deployment/' + checkpoint_fname,
                             local_path=path.join(model_dir, checkpoint_fname))

    model_list = Loading_model_in_memory(model_dir=model_dir)

    print(f'Initialising app on {requests.get("http://ip.42.pl/raw").text}:{port} with dubug={debug}')
    app.run(host="0.0.0.0", port=port, debug=debug)  # for running on instances
    # app.run(host="0.0.0.0", port=port, debug=debug, ssl_context="adhoc")  # for running on instances
