import hashlib
import base64
import pandas as pd
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")

kpn_df = pd.read_excel(kpn_path)
excel_kpns = set(kpn_df['KPN명호'].astype(str).str.strip())
print(f"Loaded {len(excel_kpns)} unique KPN hashes from Excel.")

# Let's try to match these hashes by brute-forcing KPN numbers from 1 to 2000
# Formats to try:
# "KPN1", "KPN01", "KPN001", "KPN0001", "1", "0001", "KPN-1", "KPN-001", etc.
found = {}
for i in range(1, 3000):
    candidates = [
        f"KPN{i}",
        f"KPN{i:02d}",
        f"KPN{i:03d}",
        f"KPN{i:04d}",
        f"{i}",
        f"{i:04d}",
        f"KPN-{i}",
        f"KPN {i}",
        f"KPN-{i:03d}",
        f"KPN {i:03d}"
    ]
    for cand in candidates:
        cand_bytes = cand.encode('utf-8')
        
        # Try MD5
        h_md5 = hashlib.md5(cand_bytes).digest()
        b_md5 = base64.b64encode(h_md5).decode('utf-8')
        if b_md5 in excel_kpns:
            found[b_md5] = (cand, "MD5")
            
        # Try SHA-1 (first 16 bytes or 20 bytes)
        h_sha1 = hashlib.sha1(cand_bytes).digest()
        b_sha1 = base64.b64encode(h_sha1).decode('utf-8')
        if b_sha1 in excel_kpns:
            found[b_sha1] = (cand, "SHA1")
            
        # Try SHA-256 (first 16 bytes or 32 bytes)
        h_sha256 = hashlib.sha256(cand_bytes).digest()
        b_sha256 = base64.b64encode(h_sha256).decode('utf-8')
        if b_sha256 in excel_kpns:
            found[b_sha256] = (cand, "SHA256")
            
        # Let's also try SHA-256 truncated to 16 bytes
        b_sha256_trunc = base64.b64encode(h_sha256[:16]).decode('utf-8')
        if b_sha256_trunc in excel_kpns:
            found[b_sha256_trunc] = (cand, "SHA256_trunc16")

print(f"Brute force complete. Found {len(found)} matches.")
if len(found) > 0:
    print("Samples of matches:")
    for k, v in list(found.items())[:10]:
        print(f"Hash: {k} -> Original: {v[0]} (via {v[1]})")
else:
    print("No matches found. Let's inspect some hash strings to understand their format.")
    sample_hashes = list(excel_kpns)[:5]
    for sh in sample_hashes:
        decoded = base64.b64decode(sh)
        print(f"Hash: {sh} -> decoded bytes (hex): {decoded.hex()} (len={len(decoded)})")
