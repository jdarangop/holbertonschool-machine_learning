#!/usr/bin/env python3
""" Log stats """
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_coll = client.logs.nginx
    print("{} logs".format(logs_coll.count_documents({})))
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for i in methods:
        count = logs_coll.count_documents({"method": i})
        print("\tmethod {}: {}".format(i, count))
    query = {"method": "GET", "path": "/status"}
    print("{} status check".format(logs_coll.count_documents(query)))
