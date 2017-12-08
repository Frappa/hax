import hax
import numpy as np
from hax.minitrees import TreeMaker
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax.InterpolatingMap import InterpolatingMap
from pax import utils
from pax import exceptions
from scipy.stats import binom_test
import json

class PositionReconstruction(TreeMaker):
    """Stores position-reconstruction-related variables.
    
    Provides:
       - s1_pattern_fit: s1 pattern fit computed with corrected
                         position and areas
       - s1_pattern_fit_hist: s1 pattern fit computed with corrected
                         position and hits

    """
    __version__ = '0.12'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.hits_per_channel[260]',
                      'peaks.n_saturated_per_channel',
                      'interactions.x','interactions.y','interactions.z']

    def __init__(self):
        
        hax.minitrees.TreeMaker.__init__(self)
        
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
        self.aft_map = InterpolatingMap(aftmap_filename)
        self.low_pe_threshold=10
        #hax.minitrees.Treemaker.__init__(self)

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
        if s1.area < self.low_pe_threshold:
            s1_frac = s1.area/self.low_pe_threshold
            hits_top = s1.n_hits*s1.hits_fraction_top
            s1_top = s1.area*s1.area_fraction_top
            size_top = hits_top*(1.-s1_frac) + s1_top*s1_frac
            size_tot = s1.n_hits*(1.-s1_frac) + s1.area*s1_frac
        else:
            size_top = s1.area*s1.area_fraction_top
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
        
