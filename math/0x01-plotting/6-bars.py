#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

width = 0.5

apples = plt.bar(range(3), fruit[0, :], width, color='r')
bananas = plt.bar(range(3), fruit[1, :], width, color='y', bottom=fruit[0, :])
oranges = plt.bar(range(3), fruit[2, :], width, color='#ff8000',
                  bottom=fruit[0, :] + fruit[1, :])
peaches = plt.bar(range(3), fruit[3, :], width, color='#ffe5b4',
                  bottom=fruit[0, :] + fruit[1, :] + fruit[2, :])

plt.xticks(range(3), ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 81, 10))
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend((apples[0], bananas[0], oranges[0], peaches[0]),
           ('apples', 'bananas', 'oranges', 'peaches'))

plt.show()
