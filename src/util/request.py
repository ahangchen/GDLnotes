# coding=utf-8
import json
import urllib
import urllib2


fit_url = 'http://127.0.0.1:8000/fit/'
better_hp_url = 'http://127.0.0.1:8000/hyper/'


def fit_loss(hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss})
    req = urllib2.Request(fit_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    return res


def better_hyper(hypers, loss):
    data = urllib.urlencode({'hyper': hypers, 'loss': loss})
    req = urllib2.Request(better_hp_url, data)
    response = urllib2.urlopen(req)
    res = json.loads(response.read())
    print(res)
    return res
