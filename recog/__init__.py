# set up local configuration before anything else
import os
from ConfigParser import SafeConfigParser

def initialize():
    gdict = globals()
    this_dir = os.path.split(os.path.abspath(__file__))[0]
    conf_file = os.path.join(this_dir, 'conf/conf.txt')
    sf = SafeConfigParser()
    sf.read(conf_file)
    for section in sf.sections():
        gdict.update(sf.items(section))
    gdict['__init'] = True
    scratch = gdict['scratch']
    if not os.path.exists(scratch):
        os.mkdir(scratch)
    return

initialize()
