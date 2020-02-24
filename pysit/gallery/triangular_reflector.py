import numpy as np

from pysit.gallery.gallery_base import GeneratedGalleryModel

# from pysit import * #PML, Domain

__all__ = ['TriangularReflectorModel', 'triangular_reflector']

class TriangularReflectorModel(GeneratedGalleryModel):
    """ Gallery model for constant background plus simple horizontal reflectors. 
    """
    model_name = "Triangular_Reflector"
    valid_dimensions = (2)

    @property
    def dimension(self):
        return self.domain.dim

    supported_physics = ('acoustic',)

    def __init__(self, mesh,
                       reflector_position=[(0.35, 0.42), (0.65, 0.42)], # as percentage of domain size
                       reflector_amplitude=[1.0, 1.0],
                       reflector_width=[0.05, 0.05],
                       background_velocity=1.0,
                       C0_model=None,
                       ):
        """ Constructor for a triangular reflectors
        Parameters
        ----------
        mesh : pysit mesh
            Computational mesh on which to construct the model
        reflector_position : list
            Positions of the reflectors, in global coordinates.
        reflector_amplitude : list
            Scale of the reflectors
        reflector_radius : list
            Radius of the reflectors as FWHM of Gaussian
        background_velocity : float
        Notes
        -----
        * custom C0_model() takes priority over background_velocity
        * assumes C0_model() is compliant with dimensions and scaling
            of the computational mesh. does not do error checking.
        """
        GeneratedGalleryModel.__init__(self)

        self._mesh = mesh
        self._domain = mesh.domain

        self.reflector_position = reflector_position
        self.reflector_width = reflector_width
        self.reflector_amplitude = reflector_amplitude

        self.background_velocity = background_velocity

        if C0_model is None:
            C0 = self.background_velocity*np.ones(self._mesh.shape())
        else:
            sh = self._mesh.shape(as_grid=True)
            grid = self._mesh.mesh_coords() # retrieve meshgrid

            if self.domain.dim==1:
                raise NotImplementedError()
                C0 = self.C0_model(grid[0])
            elif self.domain.dim==2:
                C0 = self.C0_model(grid[0], grid[1]).reshape(sh)
            elif self.domain.dim==3:
                raise NotImplementedError()
                C0 = self.C0_model(grid[0], grid[1], grid[2]).reshape(sh)

        dC = self._build_reflectors()

        self._initial_model = C0
        self._true_model = C0 + dC

    def _build_reflectors(self):

        mesh = self.mesh
        domain = self.domain

        grid = mesh.mesh_coords()
        XX = grid[0]
        ZZ = grid[-1]

        dC = np.zeros(mesh.shape())
        for kk in range(len(self.reflector_position)):
            # first reflector only
            XX_kk = XX - self.reflector_position[kk][0]
            ZZ_kk = ZZ - self.reflector_position[kk][1]

            dC_kk = (XX_kk <= 0.5*self.reflector_width[kk])
            dC_kk *= (ZZ_kk <= XX_kk + 0.5*self.reflector_width[kk]/np.sqrt(3))
            dC_kk *= (ZZ_kk >= -(XX_kk + 0.5*self.reflector_width[kk]/np.sqrt(3)))
            dC_kk = self.reflector_amplitude[kk]*dC_kk.astype(np.float32)

            dC += dC_kk

        return dC

def triangular_reflector(mesh, **kwargs):
    """ Friendly wrapper for instantiating the triangular reflector model. """

    # Setup the defaults
    model_config = dict()
    model_config.update(kwargs)

    return TriangularReflectorModel(mesh, **model_config).get_setup()

if __name__ == '__main__':

    from pysit import *

    #       Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 90, 70)

    #       Generate true wave speed
    C, C0, m, d = triangular_reflector(m)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.add_subplot(2,1,1)
    vis.plot(C, m)
    fig.add_subplot(2,1,2)
    vis.plot(C0, m)
    plt.show()