import hax
import numpy as np
from hax.minitrees import TreeMaker
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax.InterpolatingMap import InterpolatingMap
from pax import utils
from pax import exceptions
from scipy.stats import binom_test
from keras.models import model_from_json


class PositionReconstruction(TreeMaker):
    """Stores position-reconstruction-related variables.

    Provides:
       - s1_pattern_fit: s1 pattern fit computed with corrected
                         position and areas
       - s1_pattern_fit_hits: s1 pattern fit computed with corrected
                         position and hits

    """
    __version__ = '0.12'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.hits_per_channel[260]',
                      'peaks.n_saturated_per_channel',
                      'interactions.x', 'interactions.y', 'interactions.z']

    def __init__(self):

        hax.minitrees.TreeMaker.__init__(self)

        # We need to pull some stuff from the pax config
        self.pax_config = load_configuration("XENON1T")
        self.tpc_channels = list(range(0, 247 + 1))
        self.confused_s1_channels = []
        self.statistic = self.pax_config['BuildInteractions.BasicInteractionProperties']['s1_pattern_statistic']
        qes = np.array(self.pax_config['DEFAULT']['quantum_efficiencies'])

        self.pattern_fitter = PatternFitter(
            filename=utils.data_file_name(
                self.pax_config['WaveformSimulator']['s1_patterns_file']),
            zoom_factor=self.pax_config['WaveformSimulator'].get(
                's1_patterns_zoom_factor', 1),
            adjust_to_qe=qes[self.tpc_channels],
            default_errors=(self.pax_config['DEFAULT']['relative_qe_error'] +
                            self.pax_config['DEFAULT']['relative_gain_error'])
        )

        # For s1_aft variable
        # Don't yell at me for hardcoding the filename into hax because it was
        # also hardcoded in pax. Just kicking the can.
        aftmap_filename = utils.data_file_name('s1_aft_xyz_XENON1T_20170808.json')
        self.aft_map = InterpolatingMap(aftmap_filename)
        self.low_pe_threshold = 10

        # hax.minitrees.Treemaker.__init__(self)

        # load trained NN models
        nn_model_json = utils.data_file_name('tensorflow_nn_pos_XENON1T_20171211.json')
        json_file_nn = open(nn_model_json, 'r')
        loaded_model_json = json_file_nn.read()
        json_file_nn.close()
        loaded_nn_model = model_from_json(loaded_model_json)
        weights_file = utils.data_file_name('tensorflow_nn_pos_weights_XENON1T_20171211.h5')
        loaded_nn_model.load_weights(weights_file)
        loaded_nn_model.compile(loss='mean_squared_error', optimizer='adam')
        self.nn_tensorflow = loaded_nn_model
        self.list_bad_pmts = [1, 2, 12, 26, 34, 62, 65, 79, 86, 88, 102, 118, 130, 134, 135, 139, 148, 150, 152, 162, 178, 183, 190, 198, 206, 213, 214, 234, 239, 244, 27, 73, 91, 137, 167, 203]
        self.ntop_pmts = 127  # How to get this automatically?

    def get_data(self, dataset, event_list=None):

        data, _ = hax.minitrees.load_single_dataset(dataset, ['Corrections',
                                                              'Fundamentals'])
        self.x = data.x_3d_nn.values
        self.y = data.y_3d_nn.values
        self.z = data.z_3d_nn.values

        self.indices = list(data.event_number.values)

        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def extract_data(self, event):

        event_data = {
            "s1_pattern_fit_hax": None,
            "s1_area_fraction_top_probability_hax": None
        }

        # We first need the positions. This minitree is only valid when loading
        # Corrections since you need that to get the corrected positions
        if not len(event.interactions):
            return event_data
        e = event.event_number
        try:
            i = self.indices.index(e)
        except:
            return event_data
        interaction = event.interactions[0]
        s1 = event.peaks[interaction.s1]
        s2 = event.peaks[interaction.s2]

        # Position reconstruction based on NN from TensorFlow
        s2apc = np.array(list(s2.area_per_channel))
        s2apc_clean = []
        for i, s2_t in enumerate(s2apc):
            if i not in self.list_bad_pmts and i < self.ntop_pmts:
                s2apc_clean.append(s2_t)
        s2apc_clean = np.asarray(s2apc_clean)
        s2apc_clean_norm = s2apc_clean / s2apc_clean.sum()
        s2apc_clean_norm = s2apc_clean_norm.reshape(1, len(s2apc_clean_norm))
        predicted_xy_tensorflow = self.nn_tensorflow.predict(s2apc_clean_norm)
        event_data['x_observed_nn_tf'] = predicted_xy_tensorflow[0, 0] / 10.
        event_data['y_observed_nn_tf'] = predicted_xy_tensorflow[0, 1] / 10.

        # Want S1 AreaFractionTop Probability
        if s1.area < self.low_pe_threshold:
            s1_frac = s1.area / self.low_pe_threshold
            hits_top = s1.n_hits * s1.hits_fraction_top
            s1_top = s1.area * s1.area_fraction_top
            size_top = hits_top * (1. - s1_frac) + s1_top * s1_frac
            size_tot = s1.n_hits * (1. - s1_frac) + s1.area * s1_frac
        else:
            size_top = s1.area * s1.area_fraction_top
            size_tot = s1.area

        aft = self.aft_map.get_value(self.x[i], self.y[i], self.z[i])
        event_data['s1_area_fraction_top_probability_hax'] = binom_test(
            size_top, size_tot, aft)

        # Now do s1_pattern_fit
        apc = np.array(list(s1.area_per_channel))
        hpc = np.array(list(s1.hits_per_channel))

        # Get saturated channels
        confused_s1_channels = []
        for a, c in enumerate(s1.n_saturated_per_channel):
            if c > 0:
                confused_s1_channels.append(a)

        try:
            event_data['s1_pattern_fit_hax'] = self.pattern_fitter.compute_gof(
                (self.x[i], self.y[i], self.z[i]),
                apc[self.tpc_channels],
                pmt_selection=np.setdiff1d(self.tpc_channels,
                                           confused_s1_channels),
                statistic=self.statistic)

            event_data['s1_pattern_fit_hits_hax'] = self.pattern_fitter.compute_gof(
                (self.x[i], self.y[i], self.z[i]),
                hpc[self.tpc_channels],
                pmt_selection=np.setdiff1d(self.tpc_channels,
                                           confused_s1_channels),
                statistic=self.statistic)
        except exceptions.CoordinateOutOfRangeException:
            # pax does this too. happens when event out of TPC (usually z)
            return event_data

        return event_data
