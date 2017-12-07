import hax
import numpy as np
from hax.minitrees import TreeMaker
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax import utils
from pax import exceptions

class PositionReconstruction(TreeMaker):
    """Stores position-reconstruction-related variables.
    
    Provides:
       - s1_pattern_fit: s1 pattern fit computed with corrected
                         position and areas
    - s1_pattern_fit_hist: s1 pattern fit computed with corrected
                           position and hits

    """
    __version__ = '0.6'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.hits_per_channel[260]',
                      'peaks.n_saturated_per_channel',
                      'interactions.x','interactions.y','interactions.z']
    
    pax_config = load_configuration("XENON1T")
    tpc_channels = list(range(0,   247+1))
    confused_s1_channels = []
    statistic = pax_config['BuildInteractions.BasicInteractionProperties']['s1_pattern_statistic']
    qes = np.array(pax_config['DEFAULT']['quantum_efficiencies'])
    print(qes[tpc_channels])
    pattern_fitter = PatternFitter(
        filename=utils.data_file_name(            
            pax_config['WaveformSimulator']['s1_patterns_file']),
            zoom_factor = 1, #pax_config['WaveformSimulator'].get(
            #'s1_patterns_zoom_factor', 1),
            adjust_to_qe=qes[tpc_channels],
            default_errors=(pax_config['DEFAULT']['relative_qe_error'] +
                            pax_config['DEFAULT']['relative_gain_error'])
    )
    printer=False
    
    def get_data(self, dataset, event_list=None):
        data, _ = hax.minitrees.load_single_dataset(dataset, ['Corrections'])
        self.x = data.x_3d_nn.values
        self.y = data.y_3d_nn.values
        self.z = data.z_3d_nn.values

        self.xo = data.x_observed_tpf.values
        self.yo = data.y_observed_tpf.values
        self.zo = data.z_observed.values

        self.xc = data.x.values
        self.yc = data.y.values
        self.zc = data.z.values
        
        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def extract_data(self, event):
        event_data = {
            "s1_pattern_fit": None,
            "s1_pattern_fit_test": None,
            "s1_pattern_fit_hits": None
        }

        if not len(event.interactions):
            return event_data

        if len(event.interactions) != 0:
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            i = event.event_number
            apc = np.array(list(s1.area_per_channel))
            hpc = np.array(list(s1.hits_per_channel))

            event_data['x_interaction'] = interaction.x
            event_data['y_interaction'] = interaction.y
            event_data['z_interaction'] = interaction.z
            
            if self.printer==False:
                print(apc)
                print("%.2f, %.2f, %.2f"%(self.xo[i], self.yo[i], self.zo[i]))
                print("%.2f, %.2f, %.2f"%(interaction.x, interaction.y, interaction.z))
                self.printer=True
            # Get saturated channels
            confused_s1_channels = []
            for i, c in enumerate(s1.n_saturated_per_channel):
                if c > 0:
                    confused_s1_channels.append(i)
            #confused_s1_channels = []
            
            try:
                event_data['s1_pattern_fit_hax'] = self.pattern_fitter.compute_gof(
                    (self.x[i], self.y[i], self.z[i]),
                    apc[self.tpc_channels],
                    pmt_selection=np.setdiff1d(self.tpc_channels,
                                               confused_s1_channels),
                    statistic=self.statistic)

                event_data['s1_pattern_fit_test'] = self.pattern_fitter.compute_gof(
                    (self.xo[i], self.yo[i], self.zo[i]),
                    apc[self.tpc_channels],
                    pmt_selection=np.setdiff1d(self.tpc_channels,
                                               confused_s1_channels),
                    statistic=self.statistic)
                
                event_data['s1_pattern_fit_test_corr'] = self.pattern_fitter.compute_gof(
                    (self.xc[i], self.yc[i], self.zc[i]),
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
        
