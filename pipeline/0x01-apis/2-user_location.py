#!/usr/bin/env python3
""" User Location """
import requests
import sys


if __name__ == '__main__':
    r = requests.get(sys.argv[1])
    if r.status_code == 200:
        r_json = r.json()
        print(r_json['location'])
    elif r.status_code == 403:
        time = r.headers['X-Ratelimit-Reset']
        print('Reset in {} min'.format(time))
    else:
        print('Not found')
