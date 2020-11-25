#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search
# semantic_search = __import__('3-prueba').semantic_search

print(semantic_search('ZendeskArticles', 'When are PLDs?'))
