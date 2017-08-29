from switch_cnn_network import SwitchCnnNetwork
from simple_cnn_network import SimpleCnnNetwork
from mcnn import McnnNetwork
import sys

if len(sys.argv) != 2:
    print("Error")
    sys.exit()

job = sys.argv[1]



network = McnnNetwork()
network.construct_graphs()

if job == 'train':
    network.run_training()
else:
    network.run_tests()