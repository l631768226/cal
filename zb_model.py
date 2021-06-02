import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import  os

# 读取数据
def predict(path, filepath, user_id=1, num=3):
    data = pd.read_csv(path)
    label_encoder = LabelEncoder()
    label = data[['label']].values
    label = label_encoder.fit(label)

    pre_data = data[data['user'] == user_id]
    x = pre_data[['user', 'month', 'season', 'day', 'week', 'hour2', 'workday',
                  'hour1', 'weight', 'cluster', 'lat', 'lng']].values
    y = pre_data['label'].values
    place_name = pre_data['name'].values
    if num > len(pre_data):
        num = len(pre_data)
    place = place_name[0:num]
    x_test = x[0:num]
    # 调用模型
    tar = xgb.Booster(model_file='xgb.model')
    x_test = xgb.DMatrix(x_test)
    pre = tar.predict(x_test)

    # 模型预测结果为概率值
    pre_label = []
    for i in range(len(pre)):
        a = np.argmax(pre[i])
        pre_label.append(a)

    Res_true = y[0:num]
    Res_pre = label_encoder.inverse_transform(pre_label)
    vectorPath = os.path.join(filepath, 'Semantic_vector.csv')
    vector = pd.read_csv(vectorPath)
    name = vector['place_name'].values
    vec = vector['Semantic vector'].values
    Sem_Vec = []
    for i in range(len(place)):
        for j in range(len(name)):
            if place[i] == name[j]:
                Sem_Vec.append(vec[j])

    temp = {'user': pre_data['user'].values[0:num],
            'lat': pre_data['lat'].values[0:num],
            'lng': pre_data['lng'].values[0:num],
            'hot_value': pre_data['weight'].values[0:num]*1000,
            'place_name': place,
            'true_label': Res_true,
            'predict_label': Res_pre,
            'Semantic_vector': Sem_Vec,
            }
    df = pd.DataFrame(temp, columns=['user', 'lat', 'lng', 'hot_value', 'place_name',
                                     'true_label', 'predict_label', 'Semantic_vector'])
    dist_path = os.path.join(filepath, 'result.csv')
    df.to_csv(dist_path, index=False)


if __name__ == "__main__":
    label_encoder = LabelEncoder()
    # 以下3个变量需要由前端传递给函数作为输入
    # Path 为数据集的路径指定即可
    # User_id 为待预测用户id
    # num_value 为待预测数据的数据量
    Path = r'E:\building\NYC_C_W.csv'
    User_id = int(input("此处输入待预测用户id（最大为1083）：\n"))
    num_value = int(input("此处输入待预测数据量：\n"))
    predict(Path, r'E:\building\zb', User_id, num_value)


