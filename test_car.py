import torch
import os
import networkx as nx
import torch_geometric
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import cv2
from macbf_gnn.env.simple_car import SimpleCar


# mpl.use('macosx')
device = torch.device('cpu')
env = SimpleCar(5, device)

data = env.reset()

out = cv2.VideoWriter(
    './test.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    25,
    (1000, 1000)
)

traj = []
for i in range(100):
    data, reward, done, _ = env.step(torch.zeros(5, 4))
    unsafe_mask = env.unsafe_mask(data)
    traj.append(data)
    # out.write(env.render()[0])
traj = tuple(traj)

gif = env.render(traj)
# plt.imshow(gif[0])
# plt.show()
# a = 0
for fig in gif:
    out.write(fig)
out.release()

