import time

from OCUliiRadar.decoder_np import OCULiiDecoderNetworkPackets

def decode_network_packets():
    data_path = 'F:\\20221217\\OCULiiRadar\\20221217_1\\20221217_1_udp.pcap'
    output_path = 'F:\\20221217\\OCULiiRadar_pcd'

    odnp = OCULiiDecoderNetworkPackets(pcap_path=data_path, output_path=output_path, pcd_file_type='pcd')

    t_last = time.time()
    while 1:
        odnp.decode()
        t = time.time()
        print('    {:.2f} s'.format(t - t_last))
        print('='*100)
        t_last = time.time()

if __name__ == '__main__':
    decode_network_packets()