from configparser import ConfigParser
import os
from hax.utils import HAX_DIR
from pax.plugins.io.ROOTClass import load_event_class

CONFIG = {}

# TODO: Test if you can actually reload this config
# Maybe other modules keep 'reference' to old config...?
def load_configuration(filename, **kwargs):
    global CONFIG
    configp = ConfigParser(inline_comment_prefixes='#', strict=True)
    configp.read(filename)

    # Evaluate the values in the ini file
    CONFIG = {}
    for key, value in configp['hax'].items():
        CONFIG[key] = eval(value, {'HAX_DIR': HAX_DIR, 'os': os})

    # Override with kwargs
    CONFIG.update(kwargs)

    # Load the pax class for the data format version
    load_event_class(os.path.join(CONFIG['pax_classes_dir'], 'pax_event_class_%d.cpp' % CONFIG['pax_class_version']))

load_configuration(os.path.join(HAX_DIR, 'hax.ini'))