import numpy as np
import random
random.seed(583004949)

def extract_bursts(trace):
    bursts = []
    direction_counts = []
    direction = trace[0][1]
    burst = []
    count = 1
    for i, packet in enumerate(trace):
        if packet[1] != direction:
            bursts.append(burst)
            burst = [packet]
            direction_counts.append(count)
            direction = packet[1]
            count = 1
        else:
            burst.append(packet)
            count += 1
    bursts.append(burst)
    return bursts, direction_counts


def direction_counts(trace):
    counts = []
    direction = trace[0][1]
    count = 1
    for packet in trace:
        if packet[1] != direction:
            counts.append(count)
            direction = packet[1]
            count = 1
        else:
            count += 1
    return counts

def get_bin_sizes(feature_values, bin_input):
    bin_raw = []
    for v in feature_values.values():
        bin_raw.extend(v)
    bin_s = np.sort(bin_raw)
    bins = np.arange(0, 100 + 1, 100.0 / bin_input)

    final_bin = [np.percentile(bin_s, e) for e in bins]
    return final_bin


def slice_by_binsize(feature_values, bin_input):
    bin_for_all_instances = np.array(get_bin_sizes(feature_values, bin_input))
    d_new = {}
    for name, v in feature_values.items():
        d_new[name] = [[] for _ in range(bin_input)]

        bin_indices = np.digitize(np.array(v),
                                  bin_for_all_instances[:bin_input],
                                  right=True)
        for i in range(bin_indices.size):
            if bin_indices[i] > bin_input:
                d_new[name][-1].append(v[i])
            elif bin_indices[i] == 0:
                d_new[name][0].append(v[i])
            else:
                d_new[name][bin_indices[i]-1].append(v[i])
    return d_new


def get_statistics(feature_values, bin_input):
    sliced_dic = slice_by_binsize(feature_values, bin_input)
    bin_length = {
        key: [len(value) for value in values] for key, values in
        sliced_dic.items()
    }
    return bin_length


def normalize_data(feature_values, bin_input):
    to_be_norm = get_statistics(feature_values, bin_input)
    normed = {
        key: [float(value)/sum(values) for value in values]
        if sum(values) > 0 else values
        for key, values in to_be_norm.items()
    }
    return normed


def final_format_by_class(feature_values, bin_input):
    # norm_data = normalized_data(traces, bin_input)
    norm_data = get_statistics(feature_values, bin_input)
    final = {}
    for k in norm_data:
        c = k.split('-')[0]
        if c not in final:
            final[c] = [norm_data[k]]
        else:
            final[c].append(norm_data[k])
    return final

def padding_neural(feature_values):
    directed_neural = feature_values
    max_length = max(len(elements) for elements in directed_neural.values())
    print ("Maximum Length",max_length)
    for key, value in directed_neural.iteitems():
        if len(value)<max_length:
            zeroes_needed = max_length - len(value)
            value += [0] * zeroes_needed

    return directed_neural
