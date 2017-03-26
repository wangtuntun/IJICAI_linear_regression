#encoding=utf-8
import pandas as pd
import numpy as np
import time
from sklearn import cross_validation, linear_model
from sklearn.metrics import make_scorer
DATA_DIR="/home/wangtuntun/IJCAI/Data/"
shop_info_column_names=["shop_id","city_name","location_id","per_pay","score","comment_cnt","shop_level","cate_1","cate_2","cate_3"]
# user_pay_colimn_names=["user_id","shop_id","time_stamp"]#用python实现groupby方法不好实现，利用spark的sparkcontext.sql()实现，然后存取文件
shop_info=pd.read_csv(DATA_DIR+"shop_info.txt",names=shop_info_column_names)
flow_path="/home/wangtuntun/IJCAI/Data/ml_flow_raw_data_file.txt/part-00000"#这个文件是用sparkContext.sql()实现的，在本代码中不做代码展示。
merge_data_path="/home/wangtuntun/shop_info_flow.csv" #将合并后的特征存入该文件
feature_save_path="/home/wangtuntun/feature_data.csv"#将最终生成的特征存入该文件

def get_features_target(data):
    data_array=pd.np.array(data)#传入dataframe，为了遍历，先转为array
    features_list=[]
    target_list=[]
    for line in data_array:
        temp_list=[]
        for i in range(0,384):#一共有384个特征
            if i == 360 :#index=360对应的特征是flow
                target_temp=int(line[i])
            else:
                temp_list.append(int(line[i]))
        features_list.append(temp_list)
        target_list.append(target_temp)
    return features_list, target_list

#该评价指标用来评价模型好坏
def rmspe(zip_list):
    # w = ToWeight(y)
    # rmspe = np.sqrt(np.mean((y - yhat) ** 2))
    sum_value=0.0
    count=len(zip_list)
    for real,predict in zip_list:
        v1=(real-predict)**2
        sum_value += v1
    v2=sum_value / count
    v3=np.sqrt(v2)
    return v3

def rmspe_model(ground_truth, predictions):
    import math
    # return math.sqrt(np.sum(np.square((ground_truth - predictions)/ground_truth))/float(len(predictions)))
    sums = np.square((ground_truth - predictions) / ground_truth)
    sum = 0
    for value in sums:
        if value != float('inf'):
            sum += value
    sum = sum / float(len(predictions))
    return math.sqrt(sum)

def get_shop_number_dict():
    data = pd.read_csv(feature_save_path)
    data_array=pd.np.array(data)
    max_dict = {}
    min_dict = {}
    ave_dict = {}
    sum_dict = {}
    count_dict = {}
    all_shop_id_list = []
    for line in data_array:
        all_shop_id_list.append(line[0])
    all_shop_id_set = set(all_shop_id_list)
    for shop in all_shop_id_set:
        max_dict[shop] = 0
        min_dict[shop] = 10000
        ave_dict[shop] = 0
        sum_dict[shop] = 0
        count_dict[shop] = 0
    for line in data_array:
        flow = line[360]
        shop = line[0]
        sum_dict[shop] += flow
        count_dict[shop] += 1
        if max_dict[shop] < flow:
            max_dict[shop] = flow
        if min_dict[shop] > flow:
            min_dict[shop] = flow
    for shop in all_shop_id_set:
        ave_dict[shop] = sum_dict[shop] / count_dict[shop]
    return max_dict,min_dict,ave_dict

def predict_with_linear_regression():
    #获取训练集测试集验证集的 feature 和 target
    data=pd.read_csv(feature_save_path)
    data_other,data=cross_validation.train_test_split(data,test_size=0.001)#为了减少代码运行时间，方便测试
    train_and_valid,test=cross_validation.train_test_split(data,test_size=0.2)
    train,valid=cross_validation.train_test_split(train_and_valid,test_size=0.01)
    train_feature,train_target=get_features_target(train)
    test_feature,test_target=get_features_target(test)
    #开始训练
    clf = linear_model.LinearRegression()  # 调用新建模型
    clf.fit(train_feature, train_target)  # 代入训练集训练模型
    loss = make_scorer(rmspe_model, greater_is_better=False)  # rmspe方法不用传入参数吗？
    print loss(clf, test_feature, test_target)
    #开始预测
    max_dict, min_dict, ave_dict=get_shop_number_dict()
    predicted_list=[]
    real_list=[]
    for i in range(0, len(test_target)):
        shop=int(test_feature[i][0])
        prediction = clf.predict(test_feature[i])
        prediction=prediction[0]
        if prediction > max_dict[shop]:
            prediction = ave_dict[shop]
        if prediction < min_dict[shop]:
            prediction = ave_dict[shop]
        predicted_list.append(prediction)
        real_list.append(test_target[i])

        # print("shop: " + str(test_feature[i][0]) +" Prediction: " + str(prediction) + " Actual: " + str(test_target[i]))
    list_zip=zip(real_list,predicted_list)
    error=rmspe(list_zip)
    print error


if __name__ == '__main__':
    predict_with_linear_regression()