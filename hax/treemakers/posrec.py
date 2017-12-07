import hax
import numpy as np
from hax.minitrees import TreeMaker
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax import utils
from pax import exceptions
from scipy.stats import binom_test

class PositionReconstruction(TreeMaker):
    """Stores position-reconstruction-related variables.
    
    Provides:
       - s1_pattern_fit: s1 pattern fit computed with corrected
                         position and areas
       - s1_pattern_fit_hist: s1 pattern fit computed with corrected
                         position and hits

    """
    __version__ = '0.10'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.hits_per_channel[260]',
                      'peaks.n_saturated_per_channel',
                      'interactions.x','interactions.y','interactions.z']

    def __init__(self):
        # We need to pull some stuff from the pax config
        self.pax_config = load_configuration("XENON1T")
        self.tpc_channels = list(range(0,   247+1))
        self.confused_s1_channels = []
        self.statistic = self.pax_config \
                         ['BuildInteractions.BasicInteractionProperties'] \
                         ['s1_pattern_statistic']
        qes = np.array(self.pax_config['DEFAULT']['quantum_efficiencies'])

        self.pattern_fitter = PatternFitter(
            filename=utils.data_file_name(
                self.pax_config['WaveformSimulator']['s1_patterns_file']),
            zoom_factor = self.pax_config['WaveformSimulator'].get(
                's1_patterns_zoom_factor', 1),
            adjust_to_qe=qes[self.tpc_channels],
            default_errors=(self.pax_config['DEFAULT']['relative_qe_error'] +
                            self.pax_config['DEFAULT']['relative_gain_error'])
        )

        # For s1_aft variable
        # Don't yell at me for hardcoding the filename into hax because it was
        # also hardcoded in pax. Just kicking the can.
        #aftmap_filename = utils.data_file_name('s1_aft_xyz_XENON1T_06Mar2017.json')
        aftmap_filename = utils.data_file_name('s1_aft_xyz_XENON1T_20170808.json')
        with open(aftmap_filename) as data_file:
            data = json.load(data_file)
        r_pts = np.array(data['r_pts'])
        z_pts = np.array(data['z_pts'])
        aft_vals = np.array(data['map']).reshape(len(r_pts), len(z_pts))
        self.aft_map = RectBivariateSpline(r_pts, z_pts, aft_vals)

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
            i=self.indices.index(e)
        except:
            return event_data
        interaction = event.interactions[0]
        s1 = event.peaks[interaction.s1]
        
        # Want S1 AreaFractionTop Probability
        event_data['s1_area_fraction_top_probability_hax'] = binom_test(
            np.round(s1.area_fraction_top * s1.area),
            np.round(s1.area),
            self.aft_map(np.sqrt(self.x[i]**2 + self.y[i]**2),
                        self.z[i])[0, 0])

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
        
