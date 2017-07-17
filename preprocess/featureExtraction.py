# Jordan Ebel

import dpkt
import argparse
import sys
import ipaddress
import numpy as np


# parse input arguments
def parseArgs():
    parser = argparse.ArgumentParser(description='Parse a PCAP file')
    parser.add_argument('-f', '--file', type=file, help='.pcap file', required=True)
    parser.add_argument('-c', '--count', help='Packet count', required=False)
    parser.add_argument('-w', '--window_size', help='Window size', required=False)
    return parser.parse_args()


def cidrToSubnet(cidr):
    return ipaddress.IPv4Network(cidr)


def isYoutubePacket(ip):
    dest_ip_addr = ipaddress.IPv4Address(ip.dst)
    src_ip_addr = ipaddress.IPv4Address(ip.src)
    
    if dest_ip_addr in cidrToSubnet(unicode('173.194.0.0/16')) or \
       dest_ip_addr in cidrToSubnet(unicode('74.125.0.0/16')) or \
       dest_ip_addr in cidrToSubnet(unicode('192.178.0.0')) or \
       dest_ip_addr in cidrToSubnet(unicode('192.179.0.0/16')) or \
       dest_ip_addr in cidrToSubnet(unicode('172.217.0.0/16')) or \
       dest_ip_addr in cidrToSubnet(unicode('216.58.0.0/16')) or \
       src_ip_addr  in cidrToSubnet(unicode('173.194.0.0/16')) or \
       src_ip_addr  in cidrToSubnet(unicode('74.125.0.0/16')) or \
       src_ip_addr  in cidrToSubnet(unicode('192.178.0.0')) or \
       src_ip_addr  in cidrToSubnet(unicode('192.179.0.0/16')) or \
       src_ip_addr  in cidrToSubnet(unicode('172.217.0.0/16')) or \
       src_ip_addr  in cidrToSubnet(unicode('216.58.0.0/16')):

        if ip.p == 17:
            udp = ip.data
            if udp.dport == 443 or udp.sport == 443:
                return True
        elif ip.p == 6:
            tcp = ip.data
            if tcp.dport == 443 or tcp.sport == 443:
                return True

    return False


def isNetflixPacket(ip):
    dest_ip_addr = ipaddress.IPv4Address(ip.dst)
    src_ip_addr = ipaddress.IPv4Address(ip.src)
    
    if dest_ip_addr in cidrToSubnet(unicode('54.192.0.0/16')) or \
       dest_ip_addr in cidrToSubnet(unicode('23.246.0.0/16')) or \
       src_ip_addr  in cidrToSubnet(unicode('54.192.0.0/16')) or \
       src_ip_addr  in cidrToSubnet(unicode('23.246.0.0/16')):

        if ip.p == 17:
            udp = ip.data
            if udp.dport == 443 or udp.sport == 443:
                return True
        elif ip.p == 6:
            tcp = ip.data
            if tcp.dport == 443 or tcp.sport == 443:
                return True

    return False

def extractStandardFeatures(packet, timestamp, previous_timestamp):
    eth = dpkt.ethernet.Ethernet(packet)
    ip = eth.data

    # only IP packets are supported
    if eth.type != dpkt.ethernet.ETH_TYPE_IP:
        return -1

    inter_packet_time = timestamp - previous_timestamp
    packet_size = len(packet)
    ip_len = ip.len
    ip_header_len = ip.hl
    ip_off = ip.off
    ip_protocol = ip.p
    ip_ttl = ip.ttl

    return (inter_packet_time, packet_size, ip_len, ip_header_len, ip_off, ip_protocol, ip_ttl)

def extractWindowFeatures(packets):
    previous_arrival_time = 0
    interarrival_times = []
    ip_sizes = []
    ttls = []
    protocols = []

    # collect data for each packet
    for pktBuf in packets:

        timestamp = pktBuf[0]
        eth = dpkt.ethernet.Ethernet(pktBuf[1]);
        ip = eth.data

        # average arrival time
        if previous_arrival_time != 0:
            interarrival_times.append(timestamp - previous_arrival_time)
        previous_arrival_time = timestamp

        # average IP length
        ip_sizes.append(ip.len)

        # average TTL
        ttls.append(ip.ttl)

        # average protocol
        protocols.append(ip.p)

    return calcAverages(interarrival_times, ip_sizes, ttls, protocols)

    
def calcAverages(interarrival_times, ip_sizes, ttls, protocols):        

    if len(interarrival_times) > 0:
        mean_interarrival_time = np.mean(interarrival_times)
        var_interarrival_time = np.var(interarrival_times)
    else:
        mean_interarrival_time = 0
        var_interarrival_time = 0
    mean_ip_size = np.mean(ip_sizes)
    var_ip_size = np.var(ip_sizes)
    mean_ttl = np.mean(ttls)
    var_ttl = np.var(ttls)
    mean_protocol = np.mean(protocols)
    var_protocol = np.var(protocols)

    return (mean_interarrival_time, var_interarrival_time, mean_ip_size, var_ip_size, mean_ttl, var_ttl, mean_protocol, var_protocol)


def main():

    # parse args
    args = parseArgs()
    f = args.file
    if None != args.count:
        maxCount = int(args.count)
    else:
        maxCount = -1
    if None != args.window_size:
        window_size = int(args.window_size)
    else:
        window_size = 3

    # packet count locals
    count = 0
    previous_timestamp = 0
    target_count = 0
    total_count = 0

    # open files
    featureFile = open('featureMatrix.dat', 'w')
    categoryFile = open('category.dat', 'w')
    pcap = dpkt.pcap.Reader(f)
    pktList = pcap.readpkts()
    
    for timestamp, buf in pcap:
        total_count = total_count + 1

        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data

        # only IP packets are supported
        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
            continue

        if count == 0:
            previous_timestamp = timestamp

        # collect features
    	(inter_packet_time, packet_size, ip_len, ip_header_len, ip_off, ip_protocol, ip_ttl) = extractStandardFeatures(buf, timestamp, previous_timestamp)
    	previous_timestamp = timestamp

        # collect features from next window of packets
        # build array of next packet to extract features from
        skip_count = 0
    	window_packets = []
        for x in range(0, window_size):
            index = (total_count-1) + x + skip_count

       		# bounds check
            if (index >= len(pktList)):
	            break

            # only IP packets are supported
            # if non-IP packet found, skip it
            eth = dpkt.ethernet.Ethernet(pktList[index][1]);
            while eth.type != dpkt.ethernet.ETH_TYPE_IP:
                skip_count = skip_count + 1
                index = (total_count-1) + x + skip_count

          	    # bounds check
                if (index >= len(pktList)):
                    break

                eth = dpkt.ethernet.Ethernet(pktList[index][1]);

            # add to window arrays
            if (index < len(pktList)):
                window_packets.append(pktList[index])


        (mean_ia_time, var_ia_time, mean_ip, var_ip, mean_ttl, var_ttl, mean_p, var_p) = extractWindowFeatures(window_packets)

        # determine packet category
        category = 0 
        if isNetflixPacket(ip) or isYoutubePacket(ip):
            category = 1
            target_count = target_count + 1

        # write features and category to file
        featureFile.write('%f %d %d %d %d %d %f %f %f %f %f %f %f %f\n' % (inter_packet_time, packet_size, ip_len, ip_off, ip_protocol, ip_ttl, mean_ia_time, var_ia_time, mean_ip, var_ip, mean_ttl, var_ttl, mean_p, var_p))
        categoryFile.write('%d\n' % category)

        count=count+1
        # maximum packet check
        if maxCount != -1 and count >= maxCount:
            break
    
    print "Target count: ", target_count
    print "IP count:  ", count
    print "Total count: ", total_count
    featureFile.close()
    categoryFile.close()


if __name__ == '__main__':
    main()

