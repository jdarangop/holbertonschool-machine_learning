#!/usr/bin/env python3
""" Schools by Topic """


def schools_by_topic(mongo_collection, topic):
    """ returns the list of school having a specific topic.
        Args:
            mongo_collection: (pymongo collection object)
            topic: topic searched.
        Returns:
            (list) list of school having a specific topic.
    """
    result = list(mongo_collection.find({"topics": {"$all": [topic]}}))
    return result
