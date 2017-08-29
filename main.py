from switch_cnn_network import SwitchCnnNetwork
from simple_cnn_network import SimpleCnnNetwork


network = SimpleCnnNetwork()
network.construct_graphs()
#network.run_training()
network.run_tests()