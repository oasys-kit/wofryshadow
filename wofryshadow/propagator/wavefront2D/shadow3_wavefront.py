import numpy

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


    def get_mean_wavelength(self,nolost=True):
        wavelength_in_angstroms = self.getshcol(19,nolost=nolost)
        return 1e-10*wavelength_in_angstroms.mean()

    def toGenericWavefront(self,pixels_h=None,pixels_v=None,range_h=None,range_v=None):

        # guess number of pixels (if not defined)
        if pixels_h == None or pixels_v == None:
            pixels_esimated = int(numpy.sqrt(self.nrays()))

            if pixels_h == None:
                pixels_h = pixels_esimated

            if pixels_v == None:
                pixels_v = pixels_esimated

        # guess definition limits (if not defined)
        if range_h==None or range_v==None:
            intensity_histogram = self.histo2(1,3,nbins_h=pixels_h,nbins_v=pixels_v,
                                        nolost=1,calculate_widths=1)

            if range_h==None:
                range_h = 3 * intensity_histogram['fwhm_h']

            if range_v == None:
                range_v = 3 * intensity_histogram['fwhm_v']



        intensity_histogram = self.histo2(1,3,nbins_h=pixels_h,nbins_v=pixels_v,ref=23,
                                    xrange=[-0.5*range_h,0.5*range_h],yrange=[-0.5*range_v,0.5*range_v],
                                    nolost=1,calculate_widths=1)

        for key in intensity_histogram.keys():
            print(key)

        x = intensity_histogram['bin_h_center']
        y = intensity_histogram['bin_v_center']
        wavefront = GenericWavefront2D.initialize_wavefront_from_range(x[0],
                                                                       x[-1],
                                                                       y[0],
                                                                       y[-1],
                                                                       number_of_points=(x.size, y.size),
                                                                       wavelength=self.get_mean_wavelength())

        # TODO: check normalization and add smoothing
        complex_aplitude_modulus = numpy.sqrt(
            intensity_histogram['histogram'] / intensity_histogram['histogram'].max() )

        # TODO: phase must be calculate from directions !!!!!

        xp_histogram = self.histo2(1,3,nbins_h=pixels_h,nbins_v=pixels_v,ref=4,
                                    xrange=[-0.5*range_h,0.5*range_h],yrange=[-0.5*range_v,0.5*range_v],
                                    nolost=1,calculate_widths=1)
        yp_histogram = self.histo2(1,3,nbins_h=pixels_h,nbins_v=pixels_v,ref=6,
                                    xrange=[-0.5*range_h,0.5*range_h],yrange=[-0.5*range_v,0.5*range_v],
                                    nolost=1,calculate_widths=1)
        normalization_histogram = self.histo2(1,3,nbins_h=pixels_h,nbins_v=pixels_v,
                                    xrange=[-0.5*range_h,0.5*range_h],yrange=[-0.5*range_v,0.5*range_v],
                                    nolost=1,calculate_widths=1)

        nh = normalization_histogram['histogram']
        nh[numpy.where(nh < 1.0)] = 1.0

        complex_aplitude_phase = (xp_histogram['histogram'] + yp_histogram['histogram']) / nh

        complex_amplitude = complex_aplitude_modulus * numpy.exp(1j*complex_aplitude_phase)

        wavefront.set_complex_amplitude(complex_amplitude)

        return wavefront

    @classmethod
    def fromGenericWavefront(cls, wavefront):

        w_intensity = wavefront.get_intensity()
        w_x = wavefront.get_mesh_x()
        w_y = wavefront.get_mesh_y()
        w_phase = wavefront.get_phase()
        w_wavelength = wavefront.get_wavelength()

        nrays = w_intensity.size
        wf3 = SHADOW3Wavefront(N=nrays)

        # positions
        wf3.rays[:,0] = w_x.flatten()
        wf3.rays[:,2] = w_y.flatten()

        # Lost ray flag
        wf3.rays[:,9] = 1.0
        # energy
        wf3.rays[:,10] = 2 * numpy.pi / (w_wavelength * 1e2) # cm^-1
        # Ray index
        wf3.rays[:,11] = numpy.arange(1,nrays+1,1)

        # intensity
        # TODO: now we suppose fully polarized beam
        wf3.rays[:,6] = numpy.sqrt(w_intensity.flatten())

        # The k direction is obtained from the gradient of the phase
        kx, kz = numpy.gradient(w_phase)
        nx = kx / (2 * numpy.pi / w_wavelength)
        nz = kz / (2 * numpy.pi / w_wavelength)
        ny = numpy.sqrt(1.0 - nx**2 - nz**2)

        wf3.rays[:,3] = nx.flatten()
        wf3.rays[:,4] = ny.flatten()
        wf3.rays[:,5] = nz.flatten()

        return wf3

    @classmethod
    def decorateSHADOW3WF(self, shadow3_beam):
        wavefront = SHADOW3Wavefront.initialize_from_shadow3_beam(shadow3_beam)
        return wavefront

    def get_dimension(self):
        return WavefrontDimension.TWO






