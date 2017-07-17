# Jordan Ebel

import argparse
import pcapy
import dpkt
import featureExtraction
import curses
import signal
import sys
from collections import deque
from sklearn import svm
from sklearn.externals import joblib

count = 0
previous_timestamp = 0
pred_total_0 = 0
pred_total_1 = 0
actual_total_0 = 0
actual_total_1 = 0
tp = 0
fn = 0
fp = 0
tn = 0


def parseArgs():
    parser = argparse.ArgumentParser(description='Real time streaming video classification')
    parser.add_argument('-w', '--window_size', help='Window size', required=False)
    return parser.parse_args()


def convert_header_to_ts(header):
    timestamp = header.getts()
    return timestamp[0] + 1e-6 * timestamp[1]


def process_packet(window_packets):
    global previous_timestamp, model, count
    
    (timestamp, packet) = window_packets[0]

    eth = dpkt.ethernet.Ethernet(packet)
    ip = eth.data

    # only IP packets are supported
    if eth.type != dpkt.ethernet.ETH_TYPE_IP:
        return
            
    # increment packet count
    count = count + 1

    # set timestamp the first time
    if previous_timestamp == 0:
        previous_timestamp = timestamp

    # collect features
    (inter_packet_time, packet_size, ip_len, ip_header_len, ip_off, ip_protocol, ip_ttl) = featureExtraction.extractStandardFeatures(packet, timestamp, previous_timestamp)
    previous_timestamp = timestamp

    # collect window features
    (mean_ia_time, var_ia_time, mean_ip, var_ip, mean_ttl, var_ttl, mean_p, var_p) = featureExtraction.extractWindowFeatures(window_packets)

    # make prediction
    features = [inter_packet_time, packet_size, ip_len, ip_off, ip_protocol, ip_ttl, mean_ia_time, var_ia_time, mean_ip, var_ip, mean_ttl, var_ttl, mean_p, var_p]
    pred = model.predict([features])

    # determine actual category
    actual = 0
    if featureExtraction.isNetflixPacket(ip) or featureExtraction.isYoutubePacket(ip):
        actual = 1

    return pred, actual


def update_stats(stdscr, pred, actual):
    global count
    global pred_total_0, pred_total_1
    global actual_total_0, actual_total_1
    global tp, fn, fp, tn
    
    true_p = 0
    false_p = 0
    true_n = 0
    false_n = 0
    
    # count errors
    if pred[0] == 0: 
        pred_total_0 = pred_total_0 + 1

        if actual == 0:
            actual_total_0 = actual_total_0 + 1
            tn = tn + 1
        else:
            actual_total_1 = actual_total_1 + 1
            fn = fn + 1
    else:
        pred_total_1 = pred_total_1 + 1

        if actual == 0:
            actual_total_0 = actual_total_0 + 1
            fp = fp + 1
        else:
            actual_total_1 = actual_total_1 + 1
            tp = tp + 1

    # calculate true / false rates
    if actual_total_0 > 0:
        true_n = float(tn) / actual_total_0
        false_p = float(fp) / actual_total_0
    if actual_total_1 > 0:
        true_p = float(tp) / actual_total_1
        false_n = float(fn) / actual_total_1


    # update display
    stdscr.addstr(7, 0, "Total packets: %5d" % count)
    stdscr.addstr(8, 0, "Accuracy     : %f" % ((tp+tn)/float(count)))

    stdscr.addstr(10,  0, "Pred.  video: %5d        Pred.  other: %5d" % (pred_total_1, pred_total_0))
    stdscr.addstr(11, 0, "Actual video: %5d        Actual other: %5d" % (actual_total_1, actual_total_0))

    stdscr.addstr(13, 0, "True positive : %f   True negative : %f" % (true_p, true_n))
    stdscr.addstr(14, 0, "False positive: %f   False negative: %f" % (false_p, false_n))

    stdscr.refresh()


def sig_handler(signal, frame):
    curses.endwin()
    sys.exit(0)


def main():
    global model 

    window_packets = deque()
    window_size = 3
    count = 0
    capture_device = 'enp0s3'

    # set up signal handler
    signal.signal(signal.SIGINT, sig_handler)

    # parse arguments
    args = parseArgs()
    if None != args.window_size and 0 < int(args.window_size):
        window_size = int(args.window_size)

    # load trained classifier
    model = joblib.load('model/model.pkl')

    # initialize display
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.noecho()
    curses.cbreak()
    window = stdscr.subwin(4, 50, 0, 0)
    window.addstr(1, 7, 'Streaming Video Traffic Classifier', curses.color_pair(1))
    window.addstr(2, 18, 'Jordan Ebel', curses.color_pair(1))
    window.border()
    window.refresh()
    stdscr.addstr(4, 0, 'Capture device: %s' % capture_device)

    # start live capture
    capture = pcapy.open_live(capture_device, 65536, 1, 0)

    # build window_size-1 window of packets first
    while (len(window_packets) < window_size-1):
        header, packet = capture.next()
        eth = dpkt.ethernet.Ethernet(packet)
        if eth.type == dpkt.ethernet.ETH_TYPE_IP:
            window_packets.append((convert_header_to_ts(header), packet))

    # make prediction on incoming packets
    while (1):
        header, packet = capture.next()

        eth = dpkt.ethernet.Ethernet(packet)
        if eth.type == dpkt.ethernet.ETH_TYPE_IP:
            window_packets.append((convert_header_to_ts(header), packet))
            pred, actual = process_packet(window_packets)
            update_stats(stdscr, pred, actual)
            window_packets.popleft()


main()

