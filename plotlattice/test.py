# Create the data.
from mayavi import mlab

mlab.plot3d([1.,2.],[1.,2.],[1.,2.],tube_radius=0.05,tube_sides=20)
mlab.points3d([1.,2.],[1.,2.],[1.,2.],resolution=20)
mlab.quiver3d([1.,2.],[1.,2.],[1.,2.],[1.,1.],[0.,0.],[0.,0.],mode='arrow',scale_factor=1.,resolution=20)
mlab.show()
