#encoding=utf-8
import os
import csv
from datetime import timedelta
import datetime
import pickle


import numpy as np
from sklearn import cross_validation, linear_model
from sklearn.metrics import make_scorer

# Constants
DATA_DIR = "/home/wangtuntun/IJCAI/Rossman/Data/"
# DATA_DIR = "rossmann_data/"
CLEANED_DATA_DIR = "/home/wangtuntun/IJCAI/Rossman/cleaned_data/"
OUTPUT_DIR = "/home/wangtuntun/IJCAI/Rossman/Prediction/"

SIX_WEEKS = timedelta(days=42)


# Turns the day of the week digit into a feature list
def get_day_of_week_list(day):#1--->1,0,0,0,0,0,0        2->0,1,0,0,0,0,0
    day_list = []

    for i in range(1, 8):
        if day == str(i):
            day_list.append(1)
        else:
            day_list.append(0)

    return day_list


# Turns an abc into a feature list
def get_abc_list(abc_char):#a---> 1,0,0      b--->0,1,0
    abc_list = []

    for i in ("a", "b", "c"):
        if i == abc_char:
            abc_list.append(1)
        else:
            abc_list.append(0)

    return abc_list


"""
Gets a dictionary containing the store feature set
返回一个词典表示每个商店的店铺信息，index为shopid，内容是格式化简单清晰后的特征，没有增加和减少特征数量。
"""


def get_store_dict():
    store_dict = {}

    today = datetime.date.today()#今天的年-月-日

    with open(DATA_DIR + 'store.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for row in reader:

            if row[5] == '':#如果日期缺失，填充为今天
                comp_open = today
            else:
                comp_open = datetime.date(int(row[5]), int(row[4]), today.day)#row[5]第6列表示年，row[4]第5列表示月，用今天来表示日。不明白为什么这么处理？？-----------------------------------------

            feature_list = get_abc_list(row[1]) + get_abc_list(row[2])#两个list的内容连接复制后feature_list
            if row[3] == '':
                feature_list.append(50000)
            else:
                feature_list.append(int(row[3]))
            feature_list.append((today - comp_open).days)

            store_dict[row[0]] = feature_list#将shop_id作为feature_dict的index

    return store_dict


"""
The training data is laid out like so:
Store ID, Day of week (1-Monday, 7-Sunday), Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday

The store data is laid out like so:
Store ID, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth, CompOpenSinceYear,
    Promo2, Promo2SinceWeek, PromoSinceYear, PromoInterval
"""


def clean_training_data():#得到训练集
    store_dict = get_store_dict()#store_dict存储store的固有信息，对应store.csv
    store_history = {}#store_history存储store的历史信息(用于训练模型)，对应train.csv。另外还有统计出来的average_sales，average_customers，items
    #store_history[store_id]={'average_sales': 0.0, 'average_customers': 0.0, 'items': 0.0}
    #store_history[store_id]['average_sales']='0.0'
    #store_history[store_id][datetime]=整个记录
    x = []#所有的训练特征
    y = []#所有的目标值

    # First, get all of the training data out the file
    with open(DATA_DIR + 'train.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for row in reader:
            #如果训练数据集在中出现了store没有出现在store.csv文件，则将其平均值置为0
            if row[0] not in store_history.keys():
                store_history[row[0]] = {'average_sales': 0.0, 'average_customers': 0.0, 'items': 0.0}

            # is store open
            if row[5] != '1':#row[5]=0表示店已经关了，这个训练集中的记录无效，不做处理
                continue

            # store_history[row[0]][datetime.datetime.strptime(row[2], '%m-%d-%Y')] = row
            #第二层的key有三个加一类
            store_history[row[0]][datetime.datetime.strptime(row[2], '%Y-%m-%d')] = row  #该store在该天的信息
            store_history[row[0]]['average_sales'] += float(row[3]) # 该store所有历史天的sales累加
            store_history[row[0]]['average_customers'] += float(row[4])  #该store所有历史天的scustomers累加
            store_history[row[0]]['items'] += 1.0    #统计累加次数

    # Create our training data
    for hash_key in store_history.keys():#二维dict的key是什么？第一维是id，第二维是average_sales，average_customers，items
                                        #遍历这个二维dict需要双重for循环。外围是遍历id，每个store

        average_sales = store_history[hash_key]['average_sales'] / store_history[hash_key]['items']  #对于每个store ： ave=sum/count
        average_customers = store_history[hash_key]['average_customers'] / store_history[hash_key]['items']
                                        # 遍历这个二维dict需要双重for循环。内层是每个store的三个key：average_sales，average_customers，items
        for store_date in store_history[hash_key].keys():

            if isinstance(store_date, basestring):#如果store-date是字符串，表示这个key是average_sales，average_customers，items。不管它，继续下一个内部循环，也即是下一个第二层的key。
                continue

            predict_data = store_history[hash_key][store_date]#该store在该天的记录

            # create feature set, start with the store features
            feature_list = list(store_dict[hash_key])#用于训练的特征首先要有store在store.csv中的所有特征

            # add the day of the week
            feature_list += get_day_of_week_list(predict_data[1])#训练特征再加上该天是该周的第几天

            # add date information
            feature_list.append(store_date.day)#年月日直接分割就可以了
            feature_list.append(store_date.month)
            feature_list.append(store_date.year)

            # is the store open
            # feature_list.append(int(predict_data[5]))
            # is it a promo
            feature_list.append(int(predict_data[6]))#是否有促销
            # is it a state holiday this has letters in it, need to figure that out
            # feature_list.append(int(predict_data[7]))
            # is it a school holiday
            feature_list.append(int(predict_data[8]))#是否学校假期

            feature_list.append(average_sales)#平均销售量
            feature_list.append(average_customers)#平均客户

            # sales data from 6 weeks ago
            # feature_list.append(store_history[hash_key][store_date - SIX_WEEKS][3])

            x.append(feature_list)#训练特征是一系列特征

            # append sales 6 weeks out
            y.append(int(predict_data[3]))#目标值是该store在改天的sale

    #为什么要存储为pkl格式的？--------------------------------------------------------------------------------------------
    pickle.dump(x, open(CLEANED_DATA_DIR + "x.pkl", 'wb'))#看来list是完全按照添加顺序来的，所以可以直接存入文件，不担心特征和目标不匹配
    pickle.dump(y, open(CLEANED_DATA_DIR + "y.pkl", 'wb'))
    pickle.dump(store_history, open(CLEANED_DATA_DIR + "store_history.pkl", 'wb'))
    pickle.dump(store_dict, open(CLEANED_DATA_DIR + "store_dict.pkl", 'wb'))


def clean_test_data():#得到测试集
    x_test = []        #存储的是训练的特征值
    include_ids = []#存储的是已经经过处理的记录的id
    skipped_ids = []#存储的是跳过的记录的id

    store_history = pickle.load(open(CLEANED_DATA_DIR + "store_history.pkl", 'rb'))
    store_dict = pickle.load(open(CLEANED_DATA_DIR + "store_dict.pkl", 'rb'))

    with open(DATA_DIR + 'test.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for row in reader:
            if row[4] == '0':#如果这个store已经关了，就添加到跳过列表，继续下次循环
                skipped_ids.append(row[0])
                continue

            include_ids.append(row[0])

            # create feature set, start with the store features
            feature_list = list(store_dict[row[1]])

            # add the day of the week
            feature_list += get_day_of_week_list(row[2])

            # create the date
            store_date = datetime.datetime.strptime(row[3], '%Y-%m-%d')

            # add date information
            feature_list.append(store_date.day)
            feature_list.append(store_date.month)
            feature_list.append(store_date.year)

            # add promo
            feature_list.append(int(row[5]))

            # is it a school holiday
            feature_list.append(int(row[7]))

            average_sales = store_history[row[1]]['average_sales'] / store_history[row[1]]['items']
            average_customers = store_history[row[1]]['average_customers'] / store_history[row[1]]['items']

            # add average sales and customers
            feature_list.append(average_sales)
            feature_list.append(average_customers)

            x_test.append(feature_list)

    return x_test, include_ids, skipped_ids


def rmspe(ground_truth, predictions):
    import math
    # return math.sqrt(np.sum(np.square((ground_truth - predictions)/ground_truth))/float(len(predictions)))
    sums = np.square((ground_truth - predictions) / ground_truth)
    sum = 0
    for value in sums:
        if value != float('inf'):
            sum += value
    sum = sum / float(len(predictions))
    return math.sqrt(sum)


def predict_linear_regression():#通过训练集和验证集的数据看下模型效果如何
    x = pickle.load(open(CLEANED_DATA_DIR + "x.pkl", 'rb'))
    y = pickle.load(open(CLEANED_DATA_DIR + "y.pkl", 'rb'))

    # make_scorer方法是什么自带的方法
    loss = make_scorer(rmspe,greater_is_better=False)  # rmspe方法不用传入参数吗？

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=0)#得到训练集和验证集的特征和目标值
    clf = linear_model.LinearRegression()#调用新建模型
    clf.fit(x_train, y_train)#代入训练集训练模型
    print(loss(clf, x_test, y_test))#计算误差

    for i in range(0, len(y_test)):
        prediction = clf.predict(x_test[i])
        print("Prediction: " + str(prediction) + " Actual: " + str(y_test[i]))

    # scores = cross_validation.cross_val_score(clf, x, y, cv=5, scoring="mean_squared_error")
    # print(scores)


def predict_actual_test():#得到测试集中的预测结果
    x = pickle.load(open(CLEANED_DATA_DIR + "x.pkl", 'rb'))
    y = pickle.load(open(CLEANED_DATA_DIR + "y.pkl", 'rb'))

    clf = linear_model.LinearRegression()
    clf.fit(x, y)

    x_test_items, include_ids, exclude_ids = clean_test_data()

    with open(OUTPUT_DIR + 'submission.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Sales'])
        for i in range(0, len(include_ids)):
            writer.writerow([include_ids[i], clf.predict(x_test_items[i])[0]])

        for i in range(0, len(exclude_ids)):
            writer.writerow([exclude_ids[i], 0])


if __name__ == '__main__':

    clean_training_data()#清晰训练集
    predict_linear_regression()#测试模型性能
    # predict_actual_test()#预测未来提交数据
