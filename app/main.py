from flask import Flask, request, jsonify

from dotenv import load_dotenv
from pathlib import Path
from boto3.dynamodb.conditions import Key, Attr
import boto3
import os

from soda.utils import BipartiteGraph
from soda.crowd import SimpleMajorityClassifier


env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
REGION_NAME = os.getenv('REGION_NAME')

session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION_NAME
)


def get_task_ids(dataset_id):
    dynamodb = session.resource('dynamodb')
    task_table = dynamodb.Table(f"dataset_{dataset_id}")
    return {x['taskId'] for x in task_table.scan()['Items']}


def get_job_triplets(task_ids):
    # task ids, labeler ids, results
    U, V, E = [], [], []
    dynamodb = session.resource('dynamodb')
    task_table = dynamodb.Table("job")
    scan = task_table.scan()
    for job in scan["Items"]:
        if job["taskId"] in task_ids:
            U.append(job['taskId'])
            V.append(job['labelerId'])
            E.append(job['class'])
    return U, V, E


def encode_classes(E):
    if not E:
        return [], None
    ret = []
    classes = list(E[0].keys())
    for e in E:
        for i, c in enumerate(classes):
            if e[c]:
                ret.append(i)
                break
    return ret, classes


def get_sparse_input(dataset_id):
    U, V, E = get_job_triplets(get_task_ids(dataset_id))
    E, classes = encode_classes(E)
    return U, V, E, classes


# pip install -r git+git://github.com/Lambda-AI-Dev/soda.git


app = Flask(__name__)


@app.route("/")
def index():
    return "Test Route"

# Test Dataset ID: 4898691044887699
@app.route("/simple-majority/<dataset_id>/")
def get_simple_majority(dataset_id):
    U, V, E, classes = get_sparse_input(dataset_id)
    content = request.json
    n_classes = len(classes)
    params = {}
    # weight function specification to be continued
    if content and "weight_func" in content:
        params["weight_func"] = content["weight_func"]
    smc = SimpleMajorityClassifier(n_classes=n_classes, **params)
    bg = BipartiteGraph().add_edges_t(U, V, E)
    ret = {u: classes[i] for u, i in zip(U, smc.predict_sparse(bg))}
    return jsonify(ret)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
