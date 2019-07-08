import numpy as np
from lenstronomy.Cosmo.background import Background
from lenstronomy.LensModel.single_plane import SinglePlane
import lenstronomy.Util.constants as const


class MultiPlaneBase(object):

    """
    Multi-plane lensing class

    The lens model deflection angles are in units of reduced deflections from the specified redshift of the lens to the
    sourde redshift of the class instance.
    """

    def __init__(self, lens_model_list, lens_redshift_list, z_source_convention, cosmo=None, numerical_alpha_class=None):
        """

        :param lens_model_list: list of lens model strings
        :param lens_redshift_list: list of floats with redshifts of the lens models indicated in lens_model_list
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
        models. If None, 'z_source' is used.
        :param cosmo: instance of astropy.cosmology
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
        (see documentation in Profiles/numerical_alpha)

        """
        self._cosmo_bkg = Background(cosmo)
        self._z_source_convention = z_source_convention
        if len(lens_redshift_list) > 0:
            z_lens_max = np.max(lens_redshift_list)
            if z_lens_max >= z_source_convention:
                raise ValueError('deflector redshifts higher or equal the source redshift convention (%s >= %s for the reduced lens'
                                 ' model quantities not allowed (leads to negative reduced deflection angles!'
                                 % (z_lens_max, z_source_convention))
        if not len(lens_model_list) == len(lens_redshift_list):
            raise ValueError("The length of lens_model_list does not correspond to redshift_list")

        self._lens_model_list = lens_model_list
        self._lens_redshift_list = lens_redshift_list
        self._lens_model = SinglePlane(lens_model_list, numerical_alpha_class=numerical_alpha_class)

        if len(lens_model_list) < 1:
            self._sorted_redshift_index = []
        else:
            self._sorted_redshift_index = self._index_ordering(lens_redshift_list)
        z_before = 0
        T_z = 0
        self._T_ij_list = []
        self._T_z_list = []
        self._reduced2physical_factor = []
        for idex in self._sorted_redshift_index:
            z_lens = self._lens_redshift_list[idex]
            if z_before == z_lens:
                delta_T = 0
            else:
                T_z = self._cosmo_bkg.T_xy(0, z_lens)
                delta_T = self._cosmo_bkg.T_xy(z_before, z_lens)
            self._T_ij_list.append(delta_T)
            self._T_z_list.append(T_z)
            factor = self._cosmo_bkg.D_xy(0, z_source_convention) / self._cosmo_bkg.D_xy(z_lens, z_source_convention)
            self._reduced2physical_factor.append(factor)
            z_before = z_lens

    def ray_shooting_partial(self, x, y, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
                             include_z_start=False, T_ij_start=None, T_ij_end=None):
        """
        ray-tracing through parts of the coin, starting with (x,y) co-moving distances and angles (alpha_x, alpha_y) at redshift z_start
        and then backwards to redshift z_stop

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param include_z_start: bool, if True, includes the computation of the deflection angle at the same redshift as
        the start of the ray-tracing. ATTENTION: deflection angles at the same redshift as z_stop will be computed always!
        This can lead to duplications in the computation of deflection angles.
        :param T_ij_start: transverse angular distance between the starting redshift to the first lens plane to follow.
        If not set, will compute the distance each time this function gets executed.
        :param T_ij_end: transverse angular distance between the last lens plane being computed and z_end.
        If not set, will compute the distance each time this function gets executed.
        :return: co-moving position and angles at redshift z_stop
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        alpha_x = np.array(alpha_x)
        alpha_y = np.array(alpha_y)
        z_lens_last = z_start
        first_deflector = True

        for i, idex in enumerate(self._sorted_redshift_index):
            z_lens = self._lens_redshift_list[idex]

            if self._start_condition(include_z_start, z_lens, z_start) and z_lens <= z_stop:
                if first_deflector is True:
                    if T_ij_start is None:
                        if z_start == 0:
                            delta_T = self._T_ij_list[0]
                        else:
                            delta_T = self._cosmo_bkg.T_xy(z_start, z_lens)
                    else:
                        delta_T = T_ij_start
                    first_deflector = False
                else:
                    delta_T = self._T_ij_list[i]
                x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)
                alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, i)

                z_lens_last = z_lens
        if T_ij_end is None:
            if z_lens_last == z_stop:
                delta_T = 0
            else:
                delta_T = self._cosmo_bkg.T_xy(z_lens_last, z_stop)
        else:
            delta_T = T_ij_end
        x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)
        return x, y, alpha_x, alpha_y

    def transverse_distance_start_stop(self, z_start, z_stop, include_z_start=False):
        """
        computes the transverse distance (T_ij) that is required by the ray-tracing between the starting redshift and
        the first deflector afterwards and the last deflector before the end of the ray-tracing.

        :param z_start: redshift of the start of the ray-tracing
        :param z_stop: stop of ray-tracing
        :return: T_ij_start, T_ij_end
        """
        z_lens_last = z_start
        first_deflector = True
        T_ij_start = None
        for i, idex in enumerate(self._sorted_redshift_index):
            z_lens = self._lens_redshift_list[idex]
            if self._start_condition(include_z_start, z_lens, z_start) and z_lens <= z_stop:
                if first_deflector is True:
                    T_ij_start = self._cosmo_bkg.T_xy(z_start, z_lens)
                    first_deflector = False
                z_lens_last = z_lens
        T_ij_end = self._cosmo_bkg.T_xy(z_lens_last, z_stop)
        return T_ij_start, T_ij_end

    def ray_shooting_partial_steps(self, x, y, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
                             include_z_start=False):
        """
        ray-tracing through parts of the coin, starting with (x,y) and angles (alpha_x, alpha_y) at redshift z_start
        and then backwards to redshift z_stop.

        This function differs from 'ray_shooting_partial' in that it returns the angular position of the ray
        at each lens plane.

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param keep_range: bool, if True, only computes the angular diameter ratio between the first and last step once
        :param check_convention: flag to check the image position convention (leave this alone)
        :return: co-moving position and angles at redshift z_stop
        """
        z_lens_last = z_start
        first_deflector = True

        pos_x, pos_y, redshifts, Tz_list = [], [], [], []
        pos_x.append(x)
        pos_y.append(y)
        redshifts.append(z_start)
        Tz_list.append(self._cosmo_bkg.T_xy(0, z_start))

        current_z = z_lens_last

        for i, idex in enumerate(self._sorted_redshift_index):

            z_lens = self._lens_redshift_list[idex]

            if self._start_condition(include_z_start,z_lens,z_start) and z_lens <= z_stop:

                if z_lens != current_z:
                    new_plane = True
                    current_z = z_lens

                else:
                    new_plane = False

                if first_deflector is True:
                    delta_T = self._cosmo_bkg.T_xy(z_start, z_lens)

                    first_deflector = False
                else:
                    delta_T = self._T_ij_list[i]
                x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)
                alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, i)
                z_lens_last = z_lens

                if new_plane:

                    pos_x.append(x)
                    pos_y.append(y)
                    redshifts.append(z_lens)
                    Tz_list.append(self._T_z_list[i])

        delta_T = self._cosmo_bkg.T_xy(z_lens_last, z_stop)

        x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)

        pos_x.append(x)
        pos_y.append(y)
        redshifts.append(z_stop)
        T_z_source = self._cosmo_bkg.T_xy(0, z_stop)
        Tz_list.append(T_z_source)

        return pos_x, pos_y, redshifts, Tz_list

    def arrival_time(self, theta_x, theta_y, kwargs_lens, z_stop, T_z_stop=None, T_ij_end=None):
        """
        light travel time relative to a straight path through the coordinate (0,0)
        Negative sign means earlier arrival time

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens:
        :return: travel time in unit of days
        """
        dt_grav = np.zeros_like(theta_x)
        dt_geo = np.zeros_like(theta_x)
        x = np.zeros_like(theta_x)
        y = np.zeros_like(theta_y)
        alpha_x = np.array(theta_x)
        alpha_y = np.array(theta_y)
        i = 0
        z_lens_last = 0
        for i, index in enumerate(self._sorted_redshift_index):
            z_lens = self._lens_redshift_list[index]
            if z_lens <= z_stop:
                T_ij = self._T_ij_list[i]
                x_new, y_new = self._ray_step(x, y, alpha_x, alpha_y, T_ij)
                if i == 0:
                    pass
                elif T_ij > 0:
                    T_j = self._T_z_list[i]
                    T_i = self._T_z_list[i-1]
                    beta_i_x, beta_i_y = x / T_i, y / T_i
                    beta_j_x, beta_j_y = x_new / T_j, y_new / T_j
                    dt_geo_new = self._geometrical_delay(beta_i_x, beta_i_y, beta_j_x, beta_j_y, T_i, T_j, T_ij)
                    dt_geo += dt_geo_new
                x, y = x_new, y_new
                dt_grav_new = self._gravitational_delay(x, y, kwargs_lens, i, z_lens)
                alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, i)

                dt_grav += dt_grav_new
                z_lens_last = z_lens
        if T_ij_end is None:
            T_ij_end = self._cosmo_bkg.T_xy(z_lens_last, z_stop)
        T_ij = T_ij_end
        x_new, y_new = self._ray_step(x, y, alpha_x, alpha_y, T_ij)
        if T_z_stop is None:
            T_z_stop = self._cosmo_bkg.T_xy(0, z_stop)
        T_j = T_z_stop
        T_i = self._T_z_list[i]
        beta_i_x, beta_i_y = x / T_i, y / T_i
        beta_j_x, beta_j_y = x_new / T_j, y_new / T_j
        dt_geo_new = self._geometrical_delay(beta_i_x, beta_i_y, beta_j_x, beta_j_y, T_i, T_j, T_ij)
        dt_geo += dt_geo_new
        return dt_grav + dt_geo

    @staticmethod
    def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        #sort_index = np.argsort(redshift_list[redshift_list < z_source])
        sort_index = np.argsort(redshift_list)
        #if len(sort_index) < 1:
        #    Warning("There is no lens object between observer at z=0 and source at z=%s" % z_source)
        return sort_index

    def _reduced2physical_deflection(self, alpha_reduced, index_lens):
        """
        alpha_reduced = D_ds/Ds alpha_physical

        :param alpha_reduced: reduced deflection angle
        :param z_lens: lens redshift
        :param z_source: source redshift
        :return: physical deflection angle
        """
        factor = self._reduced2physical_factor[index_lens]
        return alpha_reduced * factor

    def _gravitational_delay(self, x, y, kwargs_lens, idex, z_lens):
        """

        :param x: co-moving coordinate at the lens plane
        :param y: co-moving coordinate at the lens plane
        :param kwargs_lens: lens model keyword arguments
        :param z_lens: redshift of the deflector
        :param idex: index of the lens model
        :return: gravitational delay in units of days as seen at z=0
        """
        theta_x, theta_y = self._co_moving2angle(x, y, idex)
        potential = self._lens_model.potential(theta_x, theta_y, kwargs_lens, k=self._sorted_redshift_index[idex])
        delay_days = self._lensing_potential2time_delay(potential, z_lens, z_source=self._z_source_convention)
        return -delay_days

    @staticmethod
    def _geometrical_delay(beta_i_x, beta_i_y, beta_j_x, beta_j_y, T_i, T_j, T_ij):
        """

        :param beta_i_x: angle on the sky at plane i
        :param beta_i_y: angle on the sky at plane i
        :param beta_j_x: angle on the sky at plane j
        :param beta_j_y: angle on the sky at plane j
        :param T_i: transverse diameter distance to z_i
        :param T_j: transverse diameter distance to z_j
        :param T_ij: transverse diameter distance from z_i to z_j
        :return: excess delay relative to a straight line
        """
        d_beta_x = beta_j_x - beta_i_x
        d_beta_y = beta_j_y - beta_i_y
        tau_ij = T_i * T_j / T_ij * const.Mpc / const.c / const.day_s * const.arcsec**2
        return tau_ij * (d_beta_x ** 2 + d_beta_y ** 2) / 2

    def _lensing_potential2time_delay(self, potential, z_lens, z_source):
        """
        transforms the lensing potential (in units arcsec^2) to a gravitational time-delay as measured at z=0

        :param potential: lensing potential
        :param z_lens: redshift of the deflector
        :param z_source: redshift of source for the definition of the lensing quantities
        :return: gravitational time-delay in units of days
        """
        D_dt = self._cosmo_bkg.D_dt(z_lens, z_source)
        delay_days = const.delay_arcsec2days(potential, D_dt)
        return delay_days

    def _co_moving2angle(self, x, y, index):
        """
        transforms co-moving distances Mpc into angles on the sky (radian)

        :param x: co-moving distance
        :param y: co-moving distance
        :param index: index of plane
        :return: angles on the sky
        """
        T_z = self._T_z_list[index]
        theta_x = x / T_z
        theta_y = y / T_z
        return theta_x, theta_y

    @staticmethod
    def _ray_step(x, y, alpha_x, alpha_y, delta_T):
        """
        ray propagation with small angle approximation

        :param x: co-moving x-position
        :param y: co-moving y-position
        :param alpha_x: deflection angle in x-direction at (x, y)
        :param alpha_y: deflection angle in y-direction at (x, y)
        :param delta_T: transverse angular diameter distance to the next step
        :return: co-moving position at the next step (backwards)
        """
        x_ = x + alpha_x * delta_T
        y_ = y + alpha_y * delta_T
        return x_, y_

    @staticmethod
    def _ray_step_add(x, y, alpha_x, alpha_y, delta_T):
        """
        ray propagation with small angle approximation

        :param x: co-moving x-position
        :param y: co-moving y-position
        :param alpha_x: deflection angle in x-direction at (x, y)
        :param alpha_y: deflection angle in y-direction at (x, y)
        :param delta_T: transverse angular diameter distance to the next step
        :return: co-moving position at the next step (backwards)
        """
        x += alpha_x * delta_T
        y += alpha_y * delta_T
        return x, y

    def _add_deflection(self, x, y, alpha_x, alpha_y, kwargs_lens, index):
        """
        adds the physical deflection angle of a single lens plane to the deflection field

        :param x: co-moving distance at the deflector plane
        :param y: co-moving distance at the deflector plane
        :param alpha_x: physical angle (radian) before the deflector plane
        :param alpha_y: physical angle (radian) before the deflector plane
        :param kwargs_lens: lens model parameter kwargs
        :param index: index of the lens model to be added
        :param idex_lens: redshift of the deflector plane
        :return: updated physical deflection after deflector plane (in a backwards ray-tracing perspective)
        """
        theta_x, theta_y = self._co_moving2angle(x, y, index)
        alpha_x_red, alpha_y_red = self._lens_model.alpha(theta_x, theta_y, kwargs_lens, k=self._sorted_redshift_index[index])
        alpha_x_phys = self._reduced2physical_deflection(alpha_x_red, index)
        alpha_y_phys = self._reduced2physical_deflection(alpha_y_red, index)
        return alpha_x - alpha_x_phys, alpha_y - alpha_y_phys

    @staticmethod
    def _start_condition(inclusive, z_lens, z_start):

        if inclusive:
            return z_lens >= z_start
        else:
            return z_lens > z_start
