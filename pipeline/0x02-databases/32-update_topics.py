#!/usr/bin/env python3
""" Change school topics """


def update_topics(mongo_collection, name, topics):
    """ changes all topics of a school document based on the name.
        Args:
            mongo_collection: (pymongo collection object)
            name: (str) the school name to update.
            topics: (list of strings) the list of topics
                    approached in the school.
        Returns:
            None.
    """
    query = {"name": name}
    newvalues = {"$set": {"topics": topics}}

    mongo_collection.update_many(query, newvalues)
