#!/usr/bin/env python3
""" List all documents """


def list_all(mongo_collection):
    """ lists all documents in a collection.
        Args:
            mongo_collection: (pymongo collection object)
        Returns:
            (list) all documents in the collection.
    """
    results = list(mongo_collection.find({}))
    return results
