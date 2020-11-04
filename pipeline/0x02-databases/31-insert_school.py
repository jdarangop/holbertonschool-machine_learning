#!/usr/bin/env python3
""" Insert a document """


def insert_school(mongo_collection, **kwargs):
    """ inserts a new document in a collection based on kwargs.
        Args:
            mongo_collection: (pymongo collection object)
        Returns:
            the new _id.
    """
    result = mongo_collection.insert(kwargs)
    return result
