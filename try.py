import matplotlib.pyplot as plt
from numpy import pi

n = 16
# create a n x n square with a marker at each point as dummy data
x_data = []
y_data = []
for x in range(n):
    for y in range(n):
        x_data.append(x)
        y_data.append(y)

# open figure
fig,ax = plt.subplots(1, 1, figsize=[20,20], dpi=100)
# set limits BEFORE plotting
ax.set_xlim((0,n-1))
ax.set_ylim((0,n-1))
# radius in data coordinates:
r = 0.5 # units
# radius in display coordinates:
r_ = ax.transData.transform([r,0])[0] - ax.transData.transform([0,0])[0] # points
# marker size as the area of a circle
marker_size = 2 * r_**2
# plot
ax.scatter(x_data, y_data, s=marker_size, edgecolors='black')

plt.show()