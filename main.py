from net.network import meta_net


def main():
    net = meta_net
    net.build_network()
    net.gpu()
    net.run()

main()
