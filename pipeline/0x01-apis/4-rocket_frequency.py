#!/usr/bin/env python3
""" Rocket Frecuency """
import requests


if __name__ == '__main__':
    r = requests.get('https://api.spacexdata.com/v4/launches')
    r_json = r.json()
    rockets = {}
    for i in r_json:
        rocket_id = i['rocket']
        rocket = requests.get('https://api.spacexdata.com/v4/rockets/' +
                              rocket_id).json()
        rocket_name = rocket['name']
        if rocket_name not in rockets.keys():
            rockets[rocket_name] = 1
        else:
            rockets[rocket_name] += 1

    for name, launches in sorted(rockets.items(),
                                 key=lambda item: item[1],
                                 reverse=True):
        print("{} : {}".format(name, launches))
