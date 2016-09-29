# coding=utf-8
import json
import urllib
import urllib2


fit_url = 'http://127.0.0.1:8000/fit/'
fit_trend_url = 'http://127.0.0.1:8000/fit/trend/'
better_hp_trend_url = 'http://127.0.0.1:8000/fit/trend/'
hp2trend_url = 'http://127.0.0.1:8000/hp2trend/'
half_trend_url = 'http://127.0.0.1:8000/half_trend/'
fit2_url = 'http://127.0.0.1:8000/fit2/'
better_hp_url = 'http://127.0.0.1:8000/hyper/'
predict_loss_url = 'http://127.0.0.1:8000/predict/'


def fit_loss(reset, hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss, 'reset': reset})
    req = urllib2.Request(fit_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    return res


def fit_more(reset, hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss, 'reset': reset})
    req = urllib2.Request(fit2_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    return res


def better_hyper(hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss})
    req = urllib2.Request(better_hp_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    better_hypers = res['msg']
    # print(res)
    return better_hypers


def predict_future(hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss})
    req = urllib2.Request(predict_loss_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    more_index = res['msg']
    # print(res)
    return more_index


def fit_trend(hypers, loss_es):
    sample_loss = list()
    for i in range(len(loss_es)):
        if i % 10 == 0:
            sample_loss.append(loss_es[i])
    data = urllib.urlencode({'hyper': hypers, 'loss': sample_loss, 'reset': 0})
    req = urllib2.Request(fit_trend_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    return res['ret']


def better_trend_hyper(hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss})
    req = urllib2.Request(hp2trend_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    better_hypers = res['msg']
    # print(res)
    return better_hypers


def half_trend_hyper(hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss})
    req = urllib2.Request(half_trend_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    better_hypers = res['msg']
    # print(res)
    return better_hypers
