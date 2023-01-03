import json
import requests
import time
from datetime import timedelta

def wx_push(message):
    # touser = 'iiicebear'      # 多个接收者用 | 分隔
    touser = 'LingSen'      # 多个接收者用 | 分隔
    # touser = 'HangGuDao|LingSen'      # 多个接收者用 | 分隔
    agentid = '1000003'
    # secret = 'j9na9ZEHLcIWaHWB2WXDcAS0XKdLpFSYsDhCeHbaab8'
    secret = '3RUBThNo05XPQSw6tbt4LNEoczlD65T4bevacMPpv0M'
    corpid = 'ww8c13e4b54ed1192e'

    json_dict = {
        "touser": touser,
        "msgtype": "text",
        "agentid": agentid,
        "text": {
            "content": message
        },
        "safe": 0,
        "enable_id_trans": 0,
        "enable_duplicate_check": 0,
        "duplicate_check_interval": 1800
    }

    response = requests.get(
        f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={corpid}&corpsecret={secret}")
    data = json.loads(response.text)
    access_token = data['access_token']

    json_str = json.dumps(json_dict)
    response_send = requests.post(
        f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}", data=json_str)
    return json.loads(response_send.text)['errmsg'] == 'ok'
if __name__=='__main__':
    # 发送的消息
    message = 'Only you!\n{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    wx_push(message)
    
    # # ! TransReID
    # start_time = round(time.monotonic()) # * 加上round就不会一长串小数了
    # mAP=0.81112312 
    # Rank1 = 0.9123
    # Rank5 = 0.94324
    # config_file = 'configs/Market/vit_transreid_stride.yml'

    # time.sleep(2)
    # end_time = round(time.monotonic()) # * 加上round就不会一长串小数了
    # message1 = '{}: mAP={:.1%}, R1={:.1%}, R5={:.1%} \nTotal running time: {}'.format(config_file[8:], mAP, Rank1, Rank5, timedelta(seconds=end_time - start_time))
    # print('message1={}'.format(message1))
    # wx_push(message1)

#! 调用方法
# from wechat_push import wx_push
# import time
# acc=0.5
# mAP=22.4
# rank1=66.5
# dataset='duke'
# message = '{} {} {}'.format(mAP,rank1,dataset) + '\n{}\n\nRunning Time: ?? \n\n==========\n\nArgs: \n\n=========='.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# wx_push(message)