import numpy as np
import random
random.seed(583004949)

def MED(bursts):
    intra_burst_delays = []
    for burst in bursts:
        timestamps = [packet[0] for packet in burst]
        intra_burst_delays.append(np.median(timestamps))
    return intra_burst_delays


def IMD(bursts):
    primary = MED(bursts)
    processed = [q-p for p, q in zip(primary[:-1], primary[1:])]

    return processed


def Variance(bursts):
    intra_burst_delays = []
    for burst in bursts:
        timestamps = [packet[0] for packet in burst]
        intra_burst_delays.append(np.var(timestamps))
    return intra_burst_delays


def IBD_FF(bursts):
    timestamps = [float(burst[0][0]) for burst in bursts]

    return np.diff(timestamps).tolist()


def IBD_IFF(bursts):
    incoming_bursts = [burst for burst in bursts if burst[0][1] == -1]
    timestamps = [float(burst[0][0]) for burst in incoming_bursts]
    return np.diff(timestamps).tolist()


def IBD_LF(bursts):
    timestamps_first = [float(burst[0][0]) for burst in bursts]
    timestamps_last = [float(burst[-1][0]) for burst in bursts]
    inter_burst_delays = [i-j for i, j in zip(timestamps_last,
                                              timestamps_first)]
    return inter_burst_delays


def IBD_OFF(bursts):
    outgoing_bursts = [burst for burst in bursts if burst[0][1] == 1]
    timestamps = [float(burst[0][0]) for burst in outgoing_bursts]
    return np.diff(timestamps).tolist()


def Burst_Length(bursts):
    timestamps_first = [float(burst[0][0]) for burst in bursts]
    timestamps_last = [float(burst[-1][0]) for burst in bursts]
    interval = [i-j for i, j in zip(timestamps_last, timestamps_first)]
    return interval