import math
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.signal import savgol_filter
from keras.models import load_model
import json

def identification(path):
    info = []
    f = pd.read_csv(path)
    users_id = list(set(f.iloc[:, 0]))
    for u in users_id:
        slice_id = list(set(f.loc[f['user_id'] == u].iloc[:,1]))
        for s in slice_id:

            Data = f.loc[(f['user_id'] == u)&(f['slice_id']==s)].values.tolist()
            # Data = pd.read_csv(path,header=None).values.tolist()

            min_trip_time = 10 * 60
            threshold = 200
            i = 0
            slice_ID = []

            while i < (len(Data) - 1):
                Counter = 0
                index = []
                while i < (len(Data) - 1) and Counter < threshold:
                    tmp_time = (Data[i + 1][4] - Data[i][4]) * 24 * 3600
                    if tmp_time <= min_trip_time:
                        Counter += 1
                        index.append(i)
                        i += 1
                    else:
                        Counter += 1
                        index.append(i)
                        i += 1
                        break
                if Counter >= 10:
                    slice_ID.append(index)

            Speed = [[] for i in range(len(slice_ID))]
            BearingRate = [[] for i in range(len(slice_ID))]
            RelativeDistance = [[] for i in range(len(slice_ID))]
            Acceleration = [[] for i in range(len(slice_ID))]
            Jerk =  [[] for i in range(len(slice_ID))]


            for k in range(len(slice_ID)):

                Delta_Time = [2]
                Bearing = []
                for i in slice_ID[k][:-1]:
                    A = (Data[i][2], Data[i][3])
                    B = (Data[i + 1][2], Data[i + 1][3])
                    Temp_RD = geodesic(A, B).meters
                    Temp_Time = (Data[i + 1][4] - Data[i][4]) * 24. * 3600 + 0.1
                    Temp_S = Temp_RD / Temp_Time
                    Speed[k].append(Temp_S)
                    RelativeDistance[k].append(Temp_RD)
                    Delta_Time.append(Temp_Time)

                    y = math.sin(math.radians(Data[i + 1][3]) - math.radians(Data[i][3])) * math.radians(
                        math.cos(Data[i + 1][2]))
                    x = math.radians(math.cos(Data[i][2])) * math.radians(math.sin(Data[i + 1][2])) - \
                        math.radians(math.sin(Data[i][2])) * math.radians(math.cos(Data[i + 1][2])) \
                        * math.radians(math.cos(Data[i + 1][3]) - math.radians(Data[i][3]))
                    b = (math.atan2(y, x) * 180. / math.pi + 360) % 360
                    Bearing.append(b)
                if len(Speed[k]) == 0:
                    continue
                Speed[k].append(Speed[k][-1])
                RelativeDistance[k].append(RelativeDistance[k][-1])
                Delta_Time.append(Delta_Time[-1])
                Bearing.append(Bearing[-1])


                for i in range(len(Speed[k]) - 1):
                    DeltaSpeed = Speed[k][i + 1] - Speed[k][i]
                    Temp_ACC = DeltaSpeed / Delta_Time[i]

                    Acceleration[k].append(Temp_ACC)

                Acceleration[k].append(Acceleration[k][-1])

                for i in range(len(Acceleration[k]) - 1):
                    Diff = Acceleration[k][i + 1] - Acceleration[k][i]
                    J = Diff / Delta_Time[i]
                    Jerk[k].append(J)

                Jerk[k].append(Jerk[k][-1])

                for i in range(len(Bearing) - 1):
                    Diff = abs(Bearing[i + 1] - Bearing[i])
                    BearingRate[k].append(Diff)
                BearingRate[k].append(BearingRate[k][-1])


                if len(slice_ID[k]) >= 10:
                    Speed[k] = savgol_filter(np.array(Speed[k]), 9, 3)
                    Acceleration[k] = savgol_filter(np.array(Acceleration[k]), 9, 3)
                    Jerk[k] = savgol_filter(np.array(Jerk[k]), 9, 3)
                    BearingRate[k] = savgol_filter(np.array(BearingRate[k]), 9, 3)

                # print(len(slice_ID[k]),len(Speed[k]),len(Acceleration[k]),len(Jerk[k]),len(BearingRate[k]))

            TotalInput = np.zeros((len(slice_ID), 1, threshold, 4), dtype=float)
            V = Speed
            A = Acceleration
            J = Jerk
            B = BearingRate
            counter = 0
            for i in range(len(slice_ID)):
                end = len(slice_ID[i])
                if end == 0 and len(V[i]) == 0:
                    continue
                TotalInput[counter, 0, 0:end, 0] = V[i]
                TotalInput[counter, 0, 0:end, 1] = A[i]
                TotalInput[counter, 0, 0:end, 2] = J[i]
                TotalInput[counter, 0, 0:end, 3] = B[i]
                counter += 1
            TotalInput = TotalInput[:counter, :, :, :]

            model_path = "my_model.h5"
            model = load_model(model_path, compile=False)
            predict = model.predict(TotalInput)
            result = list(predict.argmax(axis=1))
            label = max(result,key=result.count)

            if label == 0:
                mode = 'walk'
            if label == 1:
                mode = 'bike'
            if label == 2:
                mode = 'bus'
            if label == 3:
                mode = 'car'
            if label == 4:
                mode = 'train'

            speed = []
            acc = []
            jerk = []
            bearingrate = []
            distance = []

            for k in range(len(slice_ID)):
                speed += list(Speed[k])
                acc += list(Acceleration[k])
                jerk += list(Jerk[k])
                bearingrate += list(BearingRate[k])
                distance += list(RelativeDistance[k])
            duration_time = abs(Data[-1][4] - Data[0][4]) * 24 * 3600
            m, sa = divmod(duration_time, 60)
            h, m = divmod(m, 60)

            tmp_result = {'用户ID': u, '轨迹编号': s,'出行方式': mode, '平均速度': round(np.average(speed), 4),'平均加速度': round(np.average(acc), 4), '平均加加速度': round(np.average(jerk), 4),'平均方位变化角':round(np.average(bearingrate), 4), '移动总距离':round(sum(distance), 4),'起始时间':Data[0][5] + '/' + Data[0][6], '结束时间':Data[-1][5] + '/' + Data[-1][6], '持续时间': "%02d:%02d:%02d" % (h, m, sa)}
            info.append(tmp_result)

    result = json.dumps(info,ensure_ascii=False)
    return result

            # result = json.dumps({'出行方式': mode, '平均速度': round(np.average(speed),4),'平均加速度':round(np.average(acc),4),'平均加加速度':round(np.average(jerk),4),'平均方位变化角':round(np.average(bearingrate),4),'移动总距离':round(sum(distance),4),'起始时间':Data[0][5]+'/'+Data[0][6], '结束时间':Data[-1][5]+'/'+Data[-1][6], '持续时间':"%02d:%02d:%02d" % (h, m, s)}, sort_keys=True, indent=4, separators=(',', ':'),ensure_ascii=False)
            # f_2 = open('/Users/yueyue/desktop/Geolife/data/result.csv', 'a+', encoding='utf-8')
            # csv_writer_2 = csv.writer(f_2)
            # csv_writer_2.writerow(
            #     [u, s, mode, round(np.average(speed), 4), round(np.average(acc), 4), round(np.average(jerk), 4),
            #      round(np.average(bearingrate), 4), round(sum(distance), 4), Data[0][5] + '/' + Data[0][6],
            #      Data[-1][5] + '/' + Data[-1][6], "%02d:%02d:%02d" % (h, m, sa)])
            # f_2.close()
        # return result


if __name__ == '__main__':
    path = 'E:/building/trajectory.csv'
    result = identification(path)
    print(result)
