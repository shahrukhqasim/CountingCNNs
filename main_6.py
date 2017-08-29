from switch_cnn_network import SwitchCnnNetwork
from simple_cnn_network_6 import SimpleCnnNetworkWithFlow


network = SimpleCnnNetworkWithFlow()
network.construct_graphs()
# network.run_training()
network.run_tests()