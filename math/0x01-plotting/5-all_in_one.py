#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure()


# 0 graph
plt.subplot('321')
plt.plot(y0, 'r-')
plt.xlim(0, 10)

# 1 graph
plt.subplot('322')
plt.scatter(x1, y1, s=7, c='m')
plt.xlabel('Height (in)', fontsize=7)
plt.ylabel('Weight (lbs)', fontsize=7)
plt.title('Men\'s Height vs Weight', fontsize=7)

# 2 graph
plt.subplot('323')
plt.plot(x2, y2)
plt.xlabel('Time (years)', fontsize=7)
plt.ylabel('Fraction Remaining', fontsize=7)
plt.title('Exponential Decay of C-14', fontsize=7)
plt.yscale('log')
plt.xlim(0, 28650)

# 3 graph
plt.subplot('324')
line1 = plt.plot(x3, y31, 'r--', label='C-14')
line2 = plt.plot(x3, y32, 'g-', label='Ra-226')
plt.xlabel('Time (years)', fontsize=7)
plt.ylabel('Fraction Remaining', fontsize=7)
plt.title('Exponential Decay of Radioactive Elements', fontsize=7)
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.legend(prop={'size': 6})

# 4 graph
plt.subplot2grid(shape=(3, 2), loc=(2, 0), colspan=2)
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
plt.xlabel('Grades', fontsize=7)
plt.ylabel('Number of Students', fontsize=7)
plt.title('Project A', fontsize=7)
plt.xlim(0, 100)
plt.ylim(0, 30)

plt.suptitle('All in One')
plt.subplots_adjust(wspace=0.5, hspace=0.9)

plt.show()
