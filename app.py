from flask import Flask
from all_code import picture_show
from all_code import retrieval_similar_cases
from all_code import compute_similar
from all_code import real_layout
from flask import request
from py_trace import identification

from sf_active import data_analysis
from sf_active import period_mining
from sf_active import activity_semantic_recognition

from zb_model import predict

import os
os.chdir(r'E:/building')
data_path = './data/1.shp'
cases_path = './cases/residential_mo.shp'

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/cal', methods = ["POST"])
def cal():
    filePath = request.form.get("filePath")
    print(r'显示图片')
    return picture_show(filePath)

@app.route('/search', methods = ["POST"])
def search():
    filePath = request.form.get("filePath")
    baseDataPath = request.form.get("basePath")
    threshold = request.form.get('threshold')
    print(threshold)
    print(r'搜索相似案例')
    return retrieval_similar_cases(filePath, baseDataPath, threshold)

@app.route('/deep', methods = ["POST"])
def deep():
    filePath = request.form.get("filePath")
    inputData = request.form.get("inputData")
    infoData = str(request.form.get('infoData'))
    print(r'深度解析（雷达图）')
    return compute_similar(filePath, inputData, infoData);

@app.route('/real', methods = ["POST"])
def real():
    filePath = request.form.get("filePath")
    casePath = request.form.get("casePath")
    infoData = str(request.form.get('infoData'))
    print(r'真实布局')
    return real_layout(casePath, filePath, infoData)

@app.route('/path', methods = ["POST"])
def path():
    filePath = request.form.get("filePath")
    print(r'轨迹识别')
    return identification(filePath)

@app.route('/analysis', methods= ["POST"])
def analysis():
    filePath = request.form.get("filePath")
    distPath = request.form.get("distPath")
    a = data_analysis(filePath, distPath);
    return str(a)

@app.route('/period', methods= ["POST"])
def period():
    distPath1 = request.form.get("distPath1")
    distPath2 = request.form.get("distPath2")
    period_mining(distPath1, distPath2)
    return 'OK'

@app.route('/active', methods= ["POST"])
def activity():
    distPath2 = request.form.get("distPath2")
    distPath3 = request.form.get("distPath3")
    userId = request.form.get("userId")
    activity_semantic_recognition(distPath2, distPath3, int(userId))
    return 'OK'

@app.route('/zbactive', methods= ["POST"])
def zbactive():
    filePath = request.form.get("filePath")
    resultPath = request.form.get("resultPath")
    count = request.form.get("count")
    userId = request.form.get("userId")
    predict(filePath, resultPath, int(userId), int(count))
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000')
