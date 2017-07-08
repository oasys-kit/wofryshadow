import numpy
import scipy.signal as signal

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
            intensity_histogram = self.histo2(1, 3, nbins_h=pixels_h, nbins_v=pixels_v,
                                              nolost=1, calculate_widths=1)
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

        intensity_histogram = self.histo2(1, 3, nbins_h=pixels_h, nbins_v=pixels_v, ref=23,
                                          xrange=[-0.5*range_h, 0.5*range_h], yrange=[-0.5*range_v, 0.5*range_v],
                                          nolost=1, calculate_widths=1)

        wavelength = self.get_mean_wavelength() # meters

        x = intensity_histogram['bin_h_center']*shadow_to_meters # in meters
        y = intensity_histogram['bin_v_center']*shadow_to_meters # in meters

        wavefront = GenericWavefront2D.initialize_wavefront_from_range(x[0],
                                                                       x[-1],
                                                                       y[0],
                                                                       y[-1],
                                                                       number_of_points=(x.size, y.size),
                                                                       wavelength=wavelength)

        # TODO: check normalization and add smoothing
        complex_amplitude_modulus = self.smooth_amplitude(amplitude=numpy.sqrt(intensity_histogram['histogram'] / intensity_histogram['histogram'].max()),
                                                          pixels_h=pixels_h,
                                                          pixels_v=pixels_v)


        # TODO: phase must be calculate from directions !!!!!

        xp_histogram = self.histo2(1, 3, nbins_h=pixels_h, nbins_v=pixels_v, ref=4,
                                   xrange=[-0.5*range_h, 0.5*range_h], yrange=[-0.5*range_v, 0.5*range_v],
                                   nolost=1)
        yp_histogram = self.histo2(1, 3, nbins_h=pixels_h, nbins_v=pixels_v, ref=6,
                                   xrange=[-0.5*range_h, 0.5*range_h], yrange=[-0.5*range_v, 0.5*range_v],
                                   nolost=1)

        k_modulus = 2*numpy.pi/wavelength # meters

        kx = xp_histogram['histogram'] * k_modulus
        ky = yp_histogram['histogram'] * k_modulus

        complex_amplitude_phase = numpy.zeros(shape=(pixels_h, pixels_v))

        for i in range(0, pixels_h):
            for j in range(0, pixels_v):
                #complex_amplitude_phase[i, j] = numpy.trapz(kx[:, j], x) + numpy.trapz(ky[i, :], y)
                complex_amplitude_phase[i, j] = kx[i, j]*x[i] + ky[i, j]*y[j]

        complex_amplitude = complex_amplitude_modulus * numpy.exp(1j*complex_amplitude_phase)

        wavefront.set_complex_amplitude(complex_amplitude)

        return wavefront

    def smooth_amplitude(self, amplitude, pixels_h, pixels_v):
        kern_hanning = signal.hanning(max(5, int(pixels_h/10)))[:, None]
        kern_hanning /= kern_hanning.sum()

        kern_hanning_2 = signal.hanning(max(5, int(pixels_v/10)))[None, :]
        kern_hanning_2 /= kern_hanning.sum()

        return self.rebin(array=signal.convolve(signal.convolve(amplitude,
                                                                kern_hanning),
                                                kern_hanning_2),
                          new_shape=(pixels_h, pixels_v))

    def rebin(self, array=numpy.zeros((100, 100)), new_shape=(100, 100)):
        assert len(array.shape) == len(new_shape)

        slices = [slice(0, old, float(old) / new) for old, new in zip(array.shape, new_shape)]
        coordinates = numpy.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index

        return array[tuple(indices)]


    @classmethod
    def fromGenericWavefront(cls, wavefront, shadow_to_meters = 1e-2):

        meters_to_shadow = 1/shadow_to_meters

        w_intensity = wavefront.get_intensity()
        w_x = wavefront.get_mesh_x()
        w_y = wavefront.get_mesh_y()
        w_phase = wavefront.get_phase()
        w_wavelength = wavefront.get_wavelength() # meters
        k_modulus =  2 * numpy.pi / w_wavelength # m-1
        nrays = w_intensity.size

        wf3 = SHADOW3Wavefront(N=nrays)

        # positions
        wf3.rays[:, 0] = w_x.flatten() * meters_to_shadow # cm
        wf3.rays[:, 2] = w_y.flatten() * meters_to_shadow # cm

        # Lost ray flag
        wf3.rays[:, 9] = 1.0
        # energy
        wf3.rays[:, 10] = k_modulus / meters_to_shadow # cm-1
        # Ray index
        wf3.rays[:, 11] = numpy.arange(1, nrays+1, 1)

        normalization = nrays/numpy.sum(w_intensity.flatten()) # Shadow-like intensity

        # intensity
        # TODO: now we suppose fully polarized beam
        wf3.rays[:, 6] = numpy.sqrt(w_intensity.flatten()*normalization)

        dx = numpy.abs(w_x[1, 0] - w_x[0, 0])
        dy = numpy.abs(w_y[0, 1] - w_y[0, 0])

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






