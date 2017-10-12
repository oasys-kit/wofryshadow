import numpy, itertools
import scipy.signal as signal
from scipy.interpolate import griddata, bisplev, bisplrep
import Shadow

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.decorators import WavefrontDecorator
from wofry.propagator.wavefront import WavefrontDimension

class SHADOW3Wavefront(Shadow.Beam, WavefrontDecorator):

    def __init__(self, N=250000):
        Shadow.Beam.__init__(self,N=N)

    @classmethod
    def initialize_from_shadow3_beam(cls,shadow3_beam):
        wf3 = SHADOW3Wavefront(N=shadow3_beam.nrays())
        wf3.rays = shadow3_beam.rays.copy()

        return wf3


    def get_mean_wavelength(self, nolost=True): # meters
        wavelength_in_angstroms = self.getshcol(19, nolost=nolost)

        return 1e-10*wavelength_in_angstroms.mean()

    def toGenericWavefront(self, pixels_h=None, pixels_v=None, range_h=None, range_v=None, shadow_to_meters=1e-2):

        # guess number of pixels (if not defined)
        if pixels_h == None or pixels_v == None:
            pixels_estimated = int(numpy.sqrt(self.nrays()))

            if pixels_h == None:
                pixels_h = pixels_estimated

            if pixels_v == None:
                pixels_v = pixels_estimated

        # guess definition limits (if not defined)
        if range_h==None or range_v==None:
            intensity_histogram = self.histo2(1, 3,
                                              nbins_h=pixels_h,
                                              nbins_v=pixels_v,
                                              nolost=1,
                                              calculate_widths=1)

            if range_h==None:
                try:
                    range_h = 3 * intensity_histogram['fwhm_h']
                except:
                    shadow_x = intensity_histogram['bin_h_center']
                    range_h = numpy.abs(shadow_x[-1] - shadow_x[0])

            if range_v == None:
                try:
                    range_v = 3 * intensity_histogram['fwhm_v']
                except:
                    shadow_y = intensity_histogram['bin_v_center']
                    range_v = numpy.abs(shadow_y[-1] - shadow_y[0])

        intensity_histogram = self.histo2(1, 3,
                                          nbins_h=pixels_h,
                                          nbins_v=pixels_v,
                                          ref=23,
                                          xrange=[-0.5*range_h, 0.5*range_h],
                                          yrange=[-0.5*range_v, 0.5*range_v],
                                          nolost=1,
                                          calculate_widths=1)

        wavelength = self.get_mean_wavelength() # meters

        x = intensity_histogram['bin_h_center'] * shadow_to_meters # in meters
        z = intensity_histogram['bin_v_center'] * shadow_to_meters # in meters

        #
        # AMPLITUDE (NORMALIZATION AND SMOOTHING)
        #

        amplitude = numpy.sqrt(intensity_histogram['histogram'] / intensity_histogram['histogram'].max())
        amplitude = SHADOW3Wavefront.smooth_amplitude(amplitude=amplitude,
                                                      pixels_h=pixels_h,
                                                      pixels_v=pixels_v)

        #amplitude, new_x, new_z = SHADOW3Wavefront.spline_2D(x=x, y=z, z=amplitude)

        wavefront = GenericWavefront2D.initialize_wavefront_from_range(x[0],
                                                                       x[-1],
                                                                       z[0],
                                                                       z[-1],
                                                                       number_of_points=amplitude.shape,
                                                                       wavelength=wavelength)

        #
        # PHASE (consider Kx and Kz as the partial derivate of the phase)
        #

        '''
        xp_histogram = self.histo2(1, 3, nbins_h=pixels_h, nbins_v=pixels_v, ref=4,
                                   xrange=[-0.5*range_h, 0.5*range_h], yrange=[-0.5*range_v, 0.5*range_v],
                                   nolost=1)

        zp_histogram = self.histo2(1, 3, nbins_h=pixels_h, nbins_v=pixels_v, ref=6,
                                   xrange=[-0.5*range_h, 0.5*range_h], yrange=[-0.5*range_v, 0.5*range_v],
                                   nolost=1)
        '''

        optical_path = self.histo2(1, 3, nbins_h=pixels_h, nbins_v=pixels_v, ref=13,
                                   xrange=[-0.5*range_h, 0.5*range_h], yrange=[-0.5*range_v, 0.5*range_v],
                                   nolost=1)

        k_modulus = 2*numpy.pi/wavelength # meters

        '''
        xp = xp_histogram['histogram']
        zp = zp_histogram['histogram']

        kx = xp * k_modulus
        kz = zp * k_modulus
        '''

        phase = (k_modulus*(optical_path['histogram']*shadow_to_meters)) % 2*numpy.pi

        '''
        for i in range(0, pixels_h):
            for j in range(0, pixels_v):
                #phase[i, j] = numpy.trapz(y=kx[:i, 0], x=x[:i]) + numpy.trapz(y=kz[i, :j] , x=z[:j])
                phase[i, j] = kx[i, j]*x[i] + kz[i, j]*z[j]
        '''

        complex_amplitude = amplitude * numpy.exp(1j*phase)

        wavefront.set_complex_amplitude(complex_amplitude)

        return wavefront

    @classmethod
    def fromGenericWavefront(cls, wavefront, shadow_to_meters = 1e-2):

        meters_to_shadow = 1/shadow_to_meters

        w_intensity = wavefront.get_intensity().flatten()
        w_x = wavefront.get_mesh_x().flatten()
        w_y = wavefront.get_mesh_y().flatten()
        w_phase = wavefront.get_phase()
        w_wavelength = wavefront.get_wavelength() # meters
        k_modulus =  2 * numpy.pi / w_wavelength # m-1
        nrays = w_intensity.size

        wf3 = SHADOW3Wavefront(N=nrays)

        # positions
        wf3.rays[:, 0] = w_x * meters_to_shadow # cm
        wf3.rays[:, 2] = w_y * meters_to_shadow # cm

        # Lost ray flag
        wf3.rays[:, 9] = 1.0
        # energy
        wf3.rays[:, 10] = k_modulus / meters_to_shadow # cm-1
        # Ray index
        wf3.rays[:, 11] = numpy.arange(1, nrays+1, 1)

        normalization = nrays/numpy.sum(w_intensity) # Shadow-like intensity

        # intensity
        # TODO: now we suppose fully polarized beam
        wf3.rays[:, 6] = numpy.sqrt(w_intensity*normalization)

        dx, dy  = wavefront.delta()

        # The k direction is obtained from the gradient of the phase
        kx, kz = numpy.gradient(w_phase, dx, dy, edge_order=2)

        nx = kx / k_modulus
        nz = kz / k_modulus
        ny = numpy.sqrt(1.0 - nx**2 - nz**2)

        wf3.rays[:, 3] = nx.flatten()
        wf3.rays[:, 4] = ny.flatten()
        wf3.rays[:, 5] = nz.flatten()

        return wf3

    @classmethod
    def decorateSHADOW3WF(self, shadow3_beam):
        return SHADOW3Wavefront.initialize_from_shadow3_beam(shadow3_beam)

    def get_dimension(self):
        return WavefrontDimension.TWO

    # ------------------------------------------------------
    #
    # TOOLS
    #
    # ------------------------------------------------------

    @classmethod
    def smooth_amplitude(cls, amplitude, pixels_h, pixels_v):
        kern_hanning = signal.hanning(max(5, int(pixels_h/10)))[:, None]
        kern_hanning /= kern_hanning.sum()

        kern_hanning_2 = signal.hanning(max(5, int(pixels_v/10)))[None, :]
        kern_hanning_2 /= kern_hanning.sum()

        return cls.rebin(array=signal.convolve(signal.convolve(amplitude,
                                                               kern_hanning),
                                               kern_hanning_2),
                         new_shape=(pixels_h, pixels_v))

    @classmethod
    def rebin(cls, array=numpy.zeros((100, 100)), new_shape=(100, 100)):
        assert len(array.shape) == len(new_shape)

        slices = [slice(0, old, float(old) / new) for old, new in zip(array.shape, new_shape)]
        coordinates = numpy.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index

        return array[tuple(indices)]


    @classmethod
    def polyfit2d(cls, x, y, z, order=3):
        ncols = (order + 1)**2
        G = numpy.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))

        for k, (i,j) in enumerate(ij):
            G[:,k] = x**i * y**j
        m, _, _, _ = numpy.linalg.lstsq(G, z)

        return m

    @classmethod
    def polyval2d(cls, x, y, m):
        order = int(numpy.sqrt(len(m))) - 1
        ij = itertools.product(range(order+1), range(order+1))
        z = numpy.zeros_like(x)

        for a, (i,j) in zip(m, ij):
            z += a * x**i * y**j

        return z

    @classmethod
    def spline_2D(cls, x=numpy.zeros(100), y=numpy.zeros(100), z=numpy.zeros((100, 100)), method='cubic'):
        n_h, n_v = z.shape

        xx, yy = numpy.meshgrid(x, y)

        points = numpy.zeros((n_h*n_h, 2))
        points [:, 0] = xx.flatten()
        points [:, 1] = yy.flatten()

        grid_x, grid_y = numpy.mgrid[x[0]:x[-1]:complex(n_h*5,0), y[0]:y[-1]:complex(n_v*5,0)]

        z_spline =  griddata(points=points,
                             values=z.flatten(),
                             xi=(grid_x, grid_y),
                             method=method,
                             fill_value=0.0)

        z_spline /= z_spline.max()

        return z_spline, grid_x[:, 0], grid_y[0, :]



if __name__=="__main__":

    def func(x, y):
        return x*(1-x)*numpy.cos(4*numpy.pi*x) * numpy.sin(4*numpy.pi*y**2)**2


    grid_x, grid_y = numpy.mgrid[0:1:100j, 0:1:100j]

    points = numpy.random.rand(1000, 2)

    values = func(points[:,0], points[:,1])

    print(points.shape, values.shape)

    grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=0.0)

    import matplotlib.pyplot as plt
    plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')

    plt.show()