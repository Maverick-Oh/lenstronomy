import copy

import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Plots.model_band_plot import ModelBandPlot


class ModelPlot(object):
    """
    class that manages the summary plots of a lens model
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_params, arrow_size=0.02, cmap_string="gist_heat", likelihood_mask_list=None,
                 bands_compute=None, multi_band_type='multi-linear', band_index=0, source_marg=False, linear_prior=None):
        """

        :param kwargs_options:
        :param kwargs_data:
        :param arrow_size:
        :param cmap_string:
        """
        if bands_compute is None:
            bands_compute = [True] * len(multi_band_list)
        if multi_band_type == 'single-band':
            multi_band_type = 'multi-linear'  # this makes sure that the linear inversion outputs are coming in a list
        self._imageModel = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model,
                                                       bands_compute=bands_compute,
                                                       likelihood_mask_list=likelihood_mask_list,
                                                       band_index=band_index)

        model, error_map, cov_param, param = self._imageModel.image_linear_solve(inv_bool=True, **kwargs_params)
        logL = self._imageModel.likelihood_data_given_model(source_marg=source_marg, linear_prior=linear_prior, **kwargs_params)

        n_data = self._imageModel.num_data_evaluate
        if n_data > 0:
            print(logL * 2 / n_data, 'reduced X^2 of all evaluated imaging data combined.')

        self._band_plot_list = []
        self._index_list = []
        index = 0
        for i in range(len(multi_band_list)):
            if bands_compute[i] is True:
                if multi_band_type == 'joint-linear':
                    param_i = param
                    cov_param_i = cov_param
                else:
                    param_i = param[index]
                    cov_param_i = cov_param[index]

                bandplot = ModelBandPlot(multi_band_list, kwargs_model, model[index], error_map[index], cov_param_i,
                                         param_i, copy.deepcopy(kwargs_params),
                                         likelihood_mask_list=likelihood_mask_list, band_index=i, arrow_size=arrow_size,
                                         cmap_string=cmap_string)
                self._band_plot_list.append(bandplot)
                self._index_list.append(index)
                index += 1
            else:
                self._index_list.append(-1)

    def _select_band(self, band_index):
        """

        :param band_index: index of imaging band to be plotted
        :return: bandplot() instance of selected band, raises when band is not computed
        """
        i = self._index_list[band_index]
        if i == -1:
            raise ValueError("band %s is not computed or out of range." % band_index)
        i = int(i)
        return self._band_plot_list[i]

    def data_plot(self, band_index=0, **kwargs):
        """
        illustrates data

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.data_plot(**kwargs)

    def model_plot(self, band_index=0, **kwargs):
        """
        illustrates model

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.model_plot(**kwargs)

    def convergence_plot(self, band_index=0, **kwargs):
        """
        illustrates lensing convergence in data frame

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.convergence_plot(**kwargs)

    def normalized_residual_plot(self, band_index=0, **kwargs):
        """
        illustrates normalized residuals between data and model fit

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.normalized_residual_plot(**kwargs)

    def absolute_residual_plot(self, band_index=0, **kwargs):
        """
        illustrates absolute residuals between data and model fit

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.absolute_residual_plot(**kwargs)

    def source_plot(self, band_index=0, **kwargs):
        """
        illustrates reconstructed source (de-lensed de-convolved)

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.source_plot(**kwargs)

    def error_map_source_plot(self, band_index=0, **kwargs):
        """
        illustrates surface brightness variance in the reconstruction in the source plane

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.error_map_source_plot(**kwargs)

    def magnification_plot(self, band_index=0, **kwargs):
        """
        illustrates lensing magnification in the field of view of the data frame

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.magnification_plot(**kwargs)

    def deflection_plot(self, band_index=0, **kwargs):
        """
        illustrates lensing deflections on the field of view of the data frame

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.deflection_plot(**kwargs)

    def decomposition_plot(self, band_index=0, **kwargs):
        """
        illustrates decomposition of model components

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.decomposition_plot(**kwargs)

    def subtract_from_data_plot(self, band_index=0, **kwargs):
        """
        subtracts individual model components from the data

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.subtract_from_data_plot(**kwargs)

    def plot_main(self, band_index=0, **kwargs):
        """
        plot a set of 'main' modelling diagnostics

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_main(**kwargs)

    def plot_separate(self, band_index=0):
        """
        plot a set of 'main' modelling diagnostics

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_separate()

    def plot_subtract_from_data_all(self, band_index=0):
        """
        plot a set of 'main' modelling diagnostics

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_subtract_from_data_all()

    def plot_extinction_map(self, band_index=0, **kwargs):
        """

        :param band_index: index of band
        :param kwargs: arguments of plotting
        :return: plot instance of differential extinction map
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_extinction_map(**kwargs)

    def source(self, band_index=0, **kwargs):
        """

        :param numPix: number of grid points per axis
        :param deltaPix: width of grid points
        :param band_index: index of band
        :return: 2d array of source surface brightness
        """
        plot_band = self._select_band(band_index)
        return plot_band.source(**kwargs)
