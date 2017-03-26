#encoding=utf-8
from datetime import timedelta
import datetime



start_time_str="2015-07-01"
end_time_str="2016-10-30"
start_time=datetime.datetime.strptime(start_time_str,'%Y-%m-%d')
end_time=datetime.datetime.strptime(end_time_str,'%Y-%m-%d')


flow_dict={}
for shop_id in range(1,2001):
    for day in range(0,487):
        next_day=start_time + timedelta(days=day)
        # print next_day.date()
        flow_dict[(shop_id,next_day)]=0

f=open("/home/wangtuntun/IJCAI/Data/id_date_flow","r+")
raw_data=f.readlines()
f.close()
for ele in raw_data:
    ele=ele.split(",")
    shop_id=int(ele[0])
    date_str=ele[1]
    date=datetime.datetime.strptime(date_str,'%Y-%m-%d')
    flow=int(ele[2])
    flow_dict[(shop_id,date)]=flow
for ele in flow_dict:
    # print ele,flow_dict[ele]
    shop_id=ele[0]
    date=ele[1]
    # print flow_dict[(shop_id,date)]
    date1 = date + timedelta(days=1)
    date2 = date + timedelta(days=2)
    date3 = date + timedelta(days=3)

    date4 = date - timedelta(days=1)
    date5 = date - timedelta(days=2)
    date6 = date - timedelta(days=3)

    if flow_dict[ele] == 0:
        #如果该shop改天的flow为0,则用未来三天的
        if end_time - date > timedelta(days=3):            #如果不是截止日期的最后三天
            if flow_dict[(shop_id,date1)] != 0:
                flow_dict[ele]=flow_dict[(shop_id,date1)]
                continue
            elif flow_dict[(shop_id,date2)] != 0 :
                flow_dict[ele]=flow_dict[(shop_id, date2)]
                continue
            elif flow_dict[(shop_id,date3)] != 0:
                flow_dict[ele]=flow_dict[(shop_id,date3)]
                continue
        else:
            if flow_dict[(shop_id, date4)] != 0:
                flow_dict[ele] = flow_dict[(shop_id, date4)]
                continue
            elif flow_dict[(shop_id, date5)] != 0:
                flow_dict[ele] = flow_dict[(shop_id, date5)]
                continue
            elif flow_dict[(shop_id, date6)] != 0:
                flow_dict[ele] = flow_dict[(shop_id, date6)]
                continue
        #如果未来连续三天都没有，则就下一个商家未来三天的信息
        if shop_id != 2000:#如果shop_id=2000，则+1就超出范围。
            if flow_dict[(shop_id+1,date1)] != 0:
                flow_dict[ele]=flow_dict[(shop_id+1,date1)]
                continue
            elif flow_dict[(shop_id+1,date2)] != 0:
                flow_dict[ele]=flow_dict[shop_id+1,date2]
                continue
            elif flow_dict[(shop_id+1,date3)] != 0:
                flow_dict[ele]=flow_dict[(shop_id+1,date3)]
                continue
        else:
            if flow_dict[(shop_id-1,date1)] !=0:
                flow_dict[ele]=flow_dict[(shop_id-1,date1)]
                continue
            elif flow_dict[(shop_id-1,date2)] !=0:
                flow_dict[ele]=flow_dict[(shop_id-1,date2)]
                continue
            elif flow_dict[(shop_id-1,date3)] !=0:
                flow_dict[ele]=flow_dict[(shop_id-1,date3)]
                continue

for ele in flow_dict:
    print flow_dict[ele]