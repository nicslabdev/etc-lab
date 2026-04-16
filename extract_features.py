import os
import argparse
import numpy as np
import re
import json
from collections import Counter
from glob import glob
import random
from tqdm import tqdm
from scapy.all import rdpcap
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.inet6 import IPv6

def extract_features(packets, N=None, use_ip_layer=False, remove_ip_port=True, udp_padding=True,
                     filter_ipv6=True, remove_tcp_options=False, remove_checksums=False):
    features = []
    if N is None or N == 0:
        max_len = max(len(bytes(pkt[IP])) if IP in pkt else len(bytes(pkt)) for pkt in packets)
    else:
        max_len = N

    for pkt in packets:
        if filter_ipv6 and IPv6 in pkt:
            continue

        if use_ip_layer and IP in pkt:
            raw_bytes = bytes(pkt[IP])[:max_len]
        else:
            raw_bytes = bytes(pkt)[:max_len]

        ip_header_len = None
        if IP in pkt and len(raw_bytes) > 0:
            ip_header_len = (raw_bytes[0] & 0x0F) * 4

        if remove_tcp_options and TCP in pkt and len(raw_bytes) > 33:
            tcp_offset = ip_header_len

            if len(raw_bytes) > tcp_offset + 12:
                tcp_header_len = ((raw_bytes[tcp_offset + 12] >> 4) & 0xF) * 4

                if tcp_header_len > 20:
                    options_start = tcp_offset + 20
                    options_end = tcp_offset + tcp_header_len
                    if len(raw_bytes) >= options_end:
                        raw_bytes = raw_bytes[:options_start] + raw_bytes[options_end:]

        if remove_ip_port and len(raw_bytes) > 24:
            raw_bytes = raw_bytes[:12] + raw_bytes[24:]

        if remove_checksums:
            if TCP in pkt and ip_header_len is not None:
                tcp_checksum_pos = ip_header_len + 16
                if remove_ip_port and tcp_checksum_pos >= 24:
                    tcp_checksum_pos -= 12
                if len(raw_bytes) >= tcp_checksum_pos + 2:
                    raw_bytes = raw_bytes[:tcp_checksum_pos] + raw_bytes[tcp_checksum_pos + 2:]

            if len(raw_bytes) >= 12:
                raw_bytes = raw_bytes[:10] + raw_bytes[12:]

        if udp_padding and UDP in pkt and len(raw_bytes) > 28:
            raw_bytes = raw_bytes[:28] + b'\x00' * 12 + raw_bytes[28:]

        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        padded_array = np.pad(byte_array, (0, max_len - len(byte_array)), 'constant')
        features.append(padded_array)

    return np.array(features)

"""def bitization(features, bit_type=1):
    if bit_type == 1:
        return np.unpackbits(features.astype(np.uint8), axis=1).astype(np.float32)
    elif bit_type in [2, 4, 8]:
        factor = 256 // (2 ** bit_type)
        scaled = (features // factor).astype(np.float32)
        return scaled / (2**bit_type - 1)
    else:
        raise ValueError("bit_type must be one of: 1, 2, 4, or 8")"""
    
def bitization(features, bit_type=1):
    if bit_type not in [1, 2, 4, 8]:
        raise ValueError("bit_type must be 1, 2, 4, or 8")

    if bit_type == 1:
        # Each byte → 8 bits → 8 float32 values
        return np.unpackbits(features, axis=1).astype(np.float32)

    else:
        # For bit_type = 2, 4, 8
        values_per_byte = 8 // bit_type
        masks = (2 ** bit_type) - 1  # used for normalization

        # Creamos un array nuevo más grande donde pondremos los valores divididos
        n_samples, n_bytes = features.shape
        output = np.zeros((n_samples, n_bytes * values_per_byte), dtype=np.float32)

        for i in range(values_per_byte):
            shift = (values_per_byte - 1 - i) * bit_type
            part = (features >> shift) & masks
            output[:, i::values_per_byte] = part  # assign interleaved columns

        return output / masks  # normalize to [0, 1]

def balance_classes(X, y):
    label_counts = Counter(y)
    min_count = min(label_counts.values())

    balanced_X, balanced_y = [], []
    for label in label_counts:
        indices = [i for i, lbl in enumerate(y) if lbl == label]
        sampled_indices = random.sample(indices, min_count)
        balanced_X.extend(X[i] for i in sampled_indices)
        balanced_y.extend(y[i] for i in sampled_indices)

    return np.array(balanced_X), np.array(balanced_y)

def main():
    parser = argparse.ArgumentParser(
        description="Extract and optionally balance and bitize network packet features from .pcapng/.pcap files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pcap_dir", type=str, help="Path to the folder containing .pcap or .pcapng files")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (used in output filename)")
    parser.add_argument("--N", type=int, default=100, help="Sliding window size in bytes. Use 0 to extract entire packet.")
    parser.add_argument("--bit_type", type=int, default=8, choices=[1, 2, 4, 8], help="Bitization type: 1, 2, 4, or 8")
    parser.add_argument("--balance", action="store_true", help="Whether to balance classes to the smallest size")
    parser.add_argument("--noopt", action="store_true", help="Remove TCP options from packets")
    parser.add_argument("--nocs", action="store_true", help="Remove IP and TCP checksums from packets")
    parser.add_argument("--background_dir", type=str, default=None,
                        help="Path to background .pcap/.pcapng files for open-world binary classification. "
                             "If set, files in pcap_dir are labeled 'foreground' and files in background_dir are labeled 'background'.")

    args = parser.parse_args()

    dataset_name = args.dataset
    N = args.N
    bit_type = args.bit_type
    balance = args.balance
    noopt = args.noopt
    nocs = args.nocs
    pcap_dir = args.pcap_dir
    background_dir = args.background_dir

    filename_parts = [f"{dataset_name}_N{N}", f"BIT{bit_type}", "grouped"]
    if balance:
        filename_parts.append("balanced")
    if noopt:
        filename_parts.append("noopt")
    if nocs:
        filename_parts.append("nocs")
    if background_dir is not None:
        filename_parts.append("openworld")
    output_filename = "_".join(filename_parts) + ".npz"
    output_path = os.path.join("features", output_filename)
    os.makedirs("features", exist_ok=True)

    print(f"Saving output to: {output_path}")

    # Open-world mode: binary classification with foreground/background folders
    if background_dir is not None:
        foreground_files = glob(os.path.join(pcap_dir, '*.pcap')) + glob(os.path.join(pcap_dir, '*.pcapng'))
        background_files = glob(os.path.join(background_dir, '*.pcap')) + glob(os.path.join(background_dir, '*.pcapng'))

        if not foreground_files:
            raise FileNotFoundError(f"No .pcap or .pcapng files found in foreground dir: {pcap_dir}")
        if not background_files:
            raise FileNotFoundError(f"No .pcap or .pcapng files found in background dir: {background_dir}")

        pcaps_labels = {}
        for file in foreground_files:
            pcaps_labels[file] = "foreground"
        for file in background_files:
            pcaps_labels[file] = "background"

    else:
        grouping_file = "label_groups.json"
        if os.path.exists(grouping_file):
            with open(grouping_file, 'r') as f:
                grouping_map = json.load(f)
        else:
            grouping_map = {}

        pcap_files = glob(os.path.join(pcap_dir, '*.pcap')) + glob(os.path.join(pcap_dir, '*.pcapng'))
        if not pcap_files:
            raise FileNotFoundError(f"No .pcap or .pcapng files found in: {pcap_dir}")

        keyword_labels = {}

        # Sort label group keys by length (desc) to prioritize longer, more specific prefixes
        grouping_keys = sorted(grouping_map.keys(), key=len, reverse=True)

        # Assign labels to files based on grouping_map or fallback rule
        for file in pcap_files:
            filename = os.path.basename(file)
            filename_base = os.path.splitext(filename)[0].lower()

            # Look for the longest matching prefix from label_groups
            keyword = None
            for key in grouping_keys:
                if filename_base.startswith(key.lower()):
                    keyword = key
                    break

            # Default rule if no match found in label_groups
            if keyword is None:
                keyword = re.split(r'[_\.]', filename)[0].lower()

            label = grouping_map.get(keyword, keyword.title())
            keyword_labels[filename] = label  # key is full filename for exact match in next loop

        pcaps_labels = {}
        for file in pcap_files:
            filename = os.path.basename(file)
            label = keyword_labels.get(filename)
            if label is None:
                raise ValueError(f"Could not determine label for file: {filename}")
            pcaps_labels[file] = label

    X, y, groups = [], [], []

    print("Extracting packets...")
    for pcap_file, label in tqdm(pcaps_labels.items(), desc="Processing pcap files"):
        packets = rdpcap(pcap_file)
        features = extract_features(
            packets, N if N > 0 else None, use_ip_layer=True,
            remove_ip_port=True, udp_padding=True, filter_ipv6=True,
            remove_tcp_options=noopt, remove_checksums=nocs
        )
        X.extend(features)
        y.extend([label] * len(features))
        groups.extend([os.path.basename(pcap_file)] * len(features))

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    if balance:
        print("Balancing classes...")
        print("WARNING: balancing is applied only to X and y. The groups array is NOT re-balanced, so it will no longer be aligned with the final samples.")
        print("WARNING: if you plan to use group-based train/test splitting later, do not use --balance with this extracted dataset.")
        X, y = balance_classes(X, y)

    print(f"Applying BITization: BIT-{bit_type}")
    X = bitization(X, bit_type=bit_type)

    np.savez_compressed(output_path, X=X, y=y, groups=groups)
    print(f"Features saved successfully to '{output_path}'")

if __name__ == "__main__":
    main()