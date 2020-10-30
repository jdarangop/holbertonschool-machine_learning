#!/usr/bin/env python3
""" User Location """
import requests
import sys
import time


if __name__ == '__main__':
    r = requests.get(sys.argv[1])
    if r.status_code == 200:
        r_json = r.json()
        print(r_json['location'])
    elif r.status_code == 403:
        get_time = r.headers['X-Ratelimit-Reset']
        reset = int((int(get_time) - int(time.time())) / 60)
        print('Reset in {} min'.format(reset))
    else:
        print('Not found')
