from udp_signal_receiver import UDPSignalReceiver


if __name__ == "__main__":
    receiver = UDPSignalReceiver(debug_mode=True)
    receiver.start()
