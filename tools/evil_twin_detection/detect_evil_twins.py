import subprocess
import re
from collections import defaultdict

# ANSI escape codes
RED = "\033[91m"
RESET = "\033[0m"

# List of trusted BSSID MAC addresses (lowercase format)
TRUSTED_BSSIDS = {
    
}

def scan_networks():
    """
    Uses netsh to scan available Wi-Fi networks.
    """
    try:
        return subprocess.check_output(
            ['netsh', 'wlan', 'show', 'networks', 'mode=Bssid'],
            encoding='utf-8'
        )
    except subprocess.CalledProcessError as e:
        print(f"{RED}[ERROR] Network scan failed: {e}{RESET}")
        return ""

def parse_networks(output):
    """
    Parses SSID and BSSID entries from netsh output.
    """
    networks = defaultdict(set)
    current_ssid = None

    ssid_regex = re.compile(r"^\s*SSID\s+\d+\s+:\s+(.*)")
    bssid_regex = re.compile(r"^\s*BSSID\s+\d+\s+:\s+([0-9a-fA-F:-]{17})")

    for line in output.splitlines():
        ssid_match = ssid_regex.match(line)
        bssid_match = bssid_regex.match(line)

        if ssid_match:
            current_ssid = ssid_match.group(1).strip()
        elif bssid_match and current_ssid:
            bssid = bssid_match.group(1).strip().lower()
            networks[current_ssid].add(bssid)

    return networks

def detect_evil_twins(networks):
    """
    Compares discovered networks against trusted BSSIDs.
    Alerts if untrusted BSSIDs are present under the same SSID.
    """
    print("\n[*] Analysis complete.\n")
    for ssid, bssids in networks.items():
        untrusted = {b for b in bssids if b not in TRUSTED_BSSIDS}
        if len(bssids) > 1 and untrusted:
            print(f"{RED}[!!] Potential Evil Twin Detected for SSID '{ssid}'{RESET}")
            for b in sorted(bssids):
                mark = "[TRUSTED]" if b in TRUSTED_BSSIDS else "[UNTRUSTED]"
                color = "" if b in TRUSTED_BSSIDS else RED
                print(f"{color}     - {mark} BSSID: {b}{RESET}")
        else:
            print(f"[OK] SSID '{ssid}' passed check. BSSIDs:")
            for b in sorted(bssids):
                mark = "[TRUSTED]" if b in TRUSTED_BSSIDS else "[UNTRUSTED]"
                print(f"     - {mark} BSSID: {b}")

if __name__ == "__main__":
    print("[*] Scanning nearby Wi-Fi networks...")
    wifi_output = scan_networks()
    parsed_networks = parse_networks(wifi_output)
    detect_evil_twins(parsed_networks)
