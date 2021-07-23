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
from inference import ALLOWED_EXTENSIONS, predict_melanoma, ensemble, loading_model_in_memory
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
    preds_html = [[f'https://{data_bucket}.s3.amazonaws.com/image/ISIC_0015719.jpg',
                   'ISIC_0015719.jpg', '0.000021', [[9.94348, 'unknown'],
                                                    [4.95314, 'nevus'],
                                                    [-1.11696, 'BKL'],
                                                    [-4.33019, 'melanoma'],
                                                    [-4.36317, 'DF'],
                                                    [-4.48071, 'VASC'],
                                                    [-7.16753, 'SCC'],
                                                    [-7.73477, 'AK'],
                                                    [-8.36905, 'BCC']]
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

            dfs_split, LOGITS = predict_melanoma(image_locs, model_dir=model_dir)
            single_df, LOGITS = ensemble(dfs_split, LOGITS, len=len(image_locs))

            print("rendering index.html with predictions and image file,")
            preds_html = []
            invocation_time = datetime.now(tz=timezone.utc).strftime('%y-%m-%d %H:%M:%S')

            for i, log in enumerate(LOGITS):
                logits = []
                for j, log_ in enumerate(log):
                    logits.append([round(log_, 5), CLASS_NAMES[j]])

                # sorting the logits in descending
                for j in range(0, len(CLASS_NAMES)):
                    for j_ in range(0, len(CLASS_NAMES) - j - 1):
                        if logits[j_][0] < logits[j_ + 1][0]:
                            logits[j_], logits[j_ + 1] = logits[j_ + 1], logits[j_]

                image_id = single_df['filepath'][i].rsplit('/', 1)[1]
                img_url = f"https://{data_bucket}.s3.amazonaws.com/image/{image_id}"
                diagnosis = format(single_df['pred'][i], '.5f')
                preds_html.append([img_url, image_id, diagnosis, logits])
                item = {
                    'invocation_time': {'S': str(invocation_time)},
                    'image_id': {'S': image_id},
                    # 'user_id': {'S': str(current_user.id)},
                    # 'name': {'S': str(current_user.name)},
                    # 'email': {'S': str(current_user.email)},
                    'img_url': {'S': img_url},
                    'logits': {'S': str(logits)},
                    'diagnosis': {'S': str(diagnosis)},
                }
                ClassificationService.DynamoDBPutItem(item=item)
                ClassificationService.upload_to_s3_(bucket=data_bucket, channel="image", filepath=image_locs[i])

            gc.collect()
            return render_template("index.html", user_authenticated=False,
                                   preds_html=preds_html, current_user=current_user)
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

    if loading_model_in_memory.model_list is not None:
        loading_model_in_memory.load(model_dir=model_dir)

    print(f'Initialising app on {requests.get("http://ip.42.pl/raw").text}:{port} with dubug={debug}')
    app.run(host="0.0.0.0", port=port, debug=debug)  # for running on instances
    # app.run(host="0.0.0.0", port=port, debug=debug, ssl_context="adhoc")  # for running on instances
