#!/usr/bin/env python3
import requests


def availableShips(passengerCount):
    """ create a method that returns the list of
        ships that can hold a given number
        of passengers.
        Args:
            passengerCount: (int) number of passengers.
        Returns:
            (list) containing the names of the starships.
    """

    link = 'https://swapi-api.hbtn.io/api/starships/'

    results = []

    while link is not None:
        r = requests.get(link)
        r_json = r.json()
        link = r_json['next']
        for i in r_json['results']:
            # print(i['passengers'])
            passengers = i['passengers'].replace(',', '')
            if passengers.isnumeric():
                if int(passengers) >= passengerCount:
                    results.append(i['name'])

    return results
