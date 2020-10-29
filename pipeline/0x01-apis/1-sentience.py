#!/usr/bin/env python3
""" Planets """
import requests


def sentientPlanets():
    """ create a method that returns the list of names
        of the home planets of all sentient species.
        Args:
            None.
        Returns:
            (list) containing the names of the sentient planets.
    """

    link = 'https://swapi-api.hbtn.io/api/species/'

    results = []

    while link is not None:
        r = requests.get(link)
        r_json = r.json()
        link = r_json['next']
        for i in r_json['results']:
            if i['designation'] == 'sentient':
                if i['homeworld'] is not None:
                    name = requests.get(i['homeworld'])
                    name = name.json()['name']
                    results.append(name)

    return results
