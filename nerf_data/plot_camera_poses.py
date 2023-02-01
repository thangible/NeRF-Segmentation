import matplotlib.pyplot as plt
import json
import numpy as np

FILE = 'transforms_train.json'

def plot_coordinate_system(ax, point_3d, scale=1):
    ## Coordinate system axis
    ax.quiver(point_3d[0],point_3d[1],point_3d[2], scale, 0, 0, color="r")
    ax.quiver(point_3d[0],point_3d[1],point_3d[2], 0, scale, 0, color="g")
    ax.quiver(point_3d[0],point_3d[1],point_3d[2], 0, 0, scale, color="b")

    # axis label placement
    ax.text(point_3d[0]+scale+0.1, point_3d[1], point_3d[2], r'$x$')
    ax.text(point_3d[0], point_3d[1]+scale+0.1, point_3d[2], r'$y$')
    ax.text(point_3d[0], point_3d[1], point_3d[2]+scale+0.1, r'$z$')

def plot_coordinate_system_from_matrix(ax, matrix, scale=1):

    R = matrix[0:3,0:3]
    t = matrix[0:3,3]

    # get rotated axes
    v1 = np.matmul(R,np.array([-scale,0,0]).transpose())
    v2 = np.matmul(R,np.array([0,-scale,0]).transpose())
    v3 = np.matmul(R,np.array([0,0,-scale]).transpose())

    ## Coordinate system axis
    ax.quiver(t[0],t[1],t[2], v1[0], v1[1], v1[2], color="r")
    ax.quiver(t[0],t[1],t[2], v2[0], v2[1], v2[2], color="g")
    ax.quiver(t[0],t[1],t[2], v3[0], v3[1], v3[2], color="b")


def create_3d_axes():
    ax=plt.figure().add_subplot(projection='3d')

    # if i dont set these, the plot is all zoomed in
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(30, 45)
    return ax


all_poses = []
with open(FILE, 'r') as f:
    file = json.load(f)
    for frame in file['frames']:
        all_poses.append(np.array(frame['transform_matrix']))


ax = create_3d_axes()
for pose in all_poses:
    plot_coordinate_system_from_matrix(ax, pose, 0.3)




plt.ioff()
plt.show()
