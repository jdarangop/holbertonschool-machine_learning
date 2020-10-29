#!/usr/bin/env python3
""" SpaceX upcoming launches """
import requests


if __name__ == '__main__':

    r = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
    r_json = r.json()

    date = None
    launch = None

    for i in r_json:
        if date is None or i['date_unix'] < date:
            date = i['date_unix']
            launch = i
    launch_name = launch['name']
    local_date = launch['date_local']
    rocket_id = launch['rocket']
    get_rocket = requests.get('https://api.spacexdata.com/v4/rockets/' +
                              str(rocket_id)).json()
    rocket_name = get_rocket['name']
    launchpad_id = launch['launchpad']
    launchpad = requests.get('https://api.spacexdata.com/v4/launchpads/' +
                             str(launchpad_id)).json()
    launchpad_name = launchpad['name']
    launchpad_loc = launchpad['locality']
    print('{} ({}) {} - {} ({})'.format(launch_name,
                                        local_date,
                                        rocket_name,
                                        launchpad_name,
                                        launchpad_loc))
