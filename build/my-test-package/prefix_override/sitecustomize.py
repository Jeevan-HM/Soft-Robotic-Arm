import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jeevan/Developer/Soft-Robotic-Arm/install/my-test-package'
