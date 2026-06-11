#!/usr/bin/env python3
"""Robust per-file downloader for the G-Refer Google Drive data folder.

gdown --folder repeatedly aborts on the large .pt/.pth files (Drive rate limits),
so we download each file individually by ID with retries and skip files that are
already present with a plausible size.
"""
import os
import sys
import time
import gdown

ROOT = os.path.dirname(os.path.abspath(__file__))

# (file_id, relative_destination_path)
MANIFEST = [
    # ---- data/ (place under G-Refer root) ----
    ("1mQ7YhXU5w5F_t-z_UpQy7WSULY52BnoW", "data/amazon/data_trn.pt"),
    ("1mpK0AZ3Sov1pw6olEYadMdgLT-OOsfSA", "data/amazon/data_tst.pt"),
    ("1mjETeapSBWXba_GcRyGEYLxF_LCPDf5G", "data/amazon/dense_retrieval_results_trn.json"),
    ("1D63EY8n4PPIsnaLKH_20da-oYMIJ3J32", "data/amazon/dense_retrieval_results_tst.json"),
    ("1njo3D0nkH329q2GnRyL5AI-zpP96RK7r", "data/amazon/item_profile.json"),
    ("1ap1uq57Was4wdgh6L_WoE9N6LH4Mc83r", "data/amazon/total_trn.csv"),
    ("1WBK7D14Xa-IH-N3pMMJFHoM374IqEcIs", "data/amazon/total_tst.csv"),
    ("1SZDMjvnyC9yaxVk0gcbcfSQyrvir5Efm", "data/amazon/user_profile.json"),

    ("1oTfrzMWIpLkZtVuDUxsP6nuNuyM-eakB", "data/google/data_trn.pt"),
    ("1Q_AV94oJweYmsaG7hC1S8O7lp5zQxPzn", "data/google/data_tst.pt"),
    ("1eMFRSd6xqDU0qU4t5sOpSZzCdN3o-jNl", "data/google/dense_retrieval_results_trn.json"),
    ("1k1mhjh3ra4coFdVvAMk15csIWhnD8z6w", "data/google/dense_retrieval_results_tst.json"),
    ("14sOLw9J4hPx2S-dIORCSr6LlzVHSoJxv", "data/google/item_profile.json"),
    ("1oURCiOoLsPkclWQBNdV1--zVAlRnMKye", "data/google/total_trn.csv"),
    ("1pfXIwhl8UdtXPqPR7GojNTot3WlmEQ9a", "data/google/total_tst.csv"),
    ("1KBpGEAF1Ul-PFxJBxV_q9kLOB5kaCm-V", "data/google/user_profile.json"),

    ("19LNmaaZA6a1EzsxturUyNj1IA81j7eWE", "data/yelp/data_trn.pt"),
    ("1tucl8wK2EHHn3XRqUqLcFSGJpcswI8Pe", "data/yelp/data_tst.pt"),
    ("1v4XA-6MYODgQ_Hb6ohyFYgCZtUQ4btR-", "data/yelp/dense_retrieval_results_trn.json"),
    ("16WcK3r5rM_90rQHhXMtEnznmUl_ghRVF", "data/yelp/dense_retrieval_results_tst.json"),
    ("1iD-h8K3oUiTqDDk9N5Vd9a-wEpHhnz6m", "data/yelp/item_profile.json"),
    ("168eXB_j1D1TebhrzCi2TtsCNXoUF4MXs", "data/yelp/total_trn.csv"),
    ("1_KUYHAFuNBpD1NcUTnRz_23fkUROm2ry", "data/yelp/total_tst.csv"),
    ("1ptvIPfWN59frtBhgI12LfPzQk5MaJUqi", "data/yelp/user_profile.json"),

    # ---- raft_data/ (place under G-Refer root) ----
    ("1ukiJ9GQDe8sPhiAj0TBsMBM1hd1SUj0V", "raft_data/amazon/eval.json"),
    ("1w7HV77V_ixhjj7t80Ew8w6GwqG9kxkh6", "raft_data/amazon/test.json"),
    ("1qWIx2u2jkPbIidURke-0qk0LuhdbjS_d", "raft_data/amazon/train.json"),
    ("1jBA6Zd42WGss820RQX4ffGW76iVhPrbo", "raft_data/google/eval.json"),
    ("1Zb1Vqm6zmlsyolRhEmLRw4oxWeWTMg9-", "raft_data/google/test.json"),
    ("11_ss-4GTFBaHn8ZRKdO_s063M5grn3rV", "raft_data/google/train.json"),
    ("1OuvP-1yw1SOM3AC97SNHLSoz4iJLWNs4", "raft_data/yelp/eval.json"),
    ("11hmc1UD8m_VfyO5zWdgHrlD9bb48GoGQ", "raft_data/yelp/test.json"),
    ("1-QmnSVwe73bg3mFAab0_GgNQT31_kLOa", "raft_data/yelp/train.json"),

    # ---- saved_explanations/ (place under path_retriever/) ----
    ("1ECZD18jQqnaUIVzjEyzLIyX8iYM62CUk", "path_retriever/saved_explanations/pagelink_amazon_model_trn_pred_edge_to_comp_g_edge_mask"),
    ("1eQgBldeH4mBZI0LlySQHfihJtjMhp-rb", "path_retriever/saved_explanations/pagelink_amazon_model_trn_pred_edge_to_paths"),
    ("1fKmmYMcJ1VdBImVqUvRsDYU3J4z93p6d", "path_retriever/saved_explanations/pagelink_amazon_model_tst_pred_edge_to_comp_g_edge_mask"),
    ("1OUodQWBKHgx885cNu14UGSGUxsYloIXK", "path_retriever/saved_explanations/pagelink_amazon_model_tst_pred_edge_to_paths"),
    ("1g53C8JVkyXqCboCNVpKL6gSiHPvtaKD1", "path_retriever/saved_explanations/pagelink_google_model_trn_pred_edge_to_comp_g_edge_mask"),
    ("12FX0GS0nV_Ot98Hb4oqpMPRuBzZ3_1wi", "path_retriever/saved_explanations/pagelink_google_model_trn_pred_edge_to_paths"),
    ("16G00I9szynC34K12GCQHR7bhMNQgeqA_", "path_retriever/saved_explanations/pagelink_google_model_tst_pred_edge_to_comp_g_edge_mask"),
    ("13tgXAHY72y26h4V0EmAGsm6879gkKmE3", "path_retriever/saved_explanations/pagelink_google_model_tst_pred_edge_to_paths"),
    ("1K3Ico5BOHUwt8rO2RnvZA8XFlUUKkxFG", "path_retriever/saved_explanations/pagelink_yelp_model_trn_pred_edge_to_comp_g_edge_mask"),
    ("1x6oxhbeNUC0FmGT8I8-ZDQ-WLjCPqZ-l", "path_retriever/saved_explanations/pagelink_yelp_model_trn_pred_edge_to_paths"),
    ("1_SKDlTNUnTlz5sShygQhvY6e9GbWVz7N", "path_retriever/saved_explanations/pagelink_yelp_model_tst_pred_edge_to_comp_g_edge_mask"),
    ("1HxEDzywSsKZJTKPCjE7KiAUMYlKxHpWM", "path_retriever/saved_explanations/pagelink_yelp_model_tst_pred_edge_to_paths"),

    # ---- saved_models/ (place under path_retriever/) ----
    ("1vDEt4e99BK1smU1Yr4Co3zinTWjEBOxF", "path_retriever/saved_models/amazon_model_trn.pth"),
    ("125sYamzCRrQN8g5N7xlQIC7TdrnEH4nV", "path_retriever/saved_models/amazon_model_tst.pth"),
    ("1mwxXU95mIxAWCBGZTHRkA_cn2y0weq7B", "path_retriever/saved_models/google_model_trn.pth"),
    ("1RUCFGe8xZHd2ASYQ5OauHUwBjZNXtnOH", "path_retriever/saved_models/google_model_tst.pth"),
    ("14gUtMoiBbEzz_qUN7AIDqrjx7C0r_mGP", "path_retriever/saved_models/yelp_model_trn.pth"),
    ("1eq0gWr9s2u5JHPoVeykh5TSWpV12w2E_", "path_retriever/saved_models/yelp_model_tst.pth"),
]

# Step-7 (RAFT) only needs raft_data/ + (later) saved_models/saved_explanations;
# the large data/*.pt files are only for graph-retrieval steps 1-6. Fetch the
# small/important files first so a flaky Drive connection yields what we need.
PRIORITY = ("raft_data/", "saved_explanations", "saved_models")
MANIFEST.sort(key=lambda x: (0 if any(p in x[1] for p in PRIORITY) else 1))

MIN_BYTES = 1024  # treat files smaller than this as failed/placeholder


def main():
    failed = []
    for i, (fid, rel) in enumerate(MANIFEST, 1):
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
            print(f"[{i}/{len(MANIFEST)}] SKIP (exists) {rel} ({os.path.getsize(dst)} bytes)")
            continue
        ok = False
        for attempt in range(1, 5):
            try:
                print(f"[{i}/{len(MANIFEST)}] DL  {rel} (attempt {attempt})")
                gdown.download(id=fid, output=dst, quiet=False, resume=True)
                if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
                    ok = True
                    break
            except Exception as e:
                print(f"    error: {e}")
            time.sleep(5 * attempt)
        if not ok:
            print(f"    !! FAILED {rel}")
            failed.append(rel)
    print("\n==== SUMMARY ====")
    if failed:
        print(f"FAILED {len(failed)} files:")
        for f in failed:
            print("  ", f)
        sys.exit(1)
    print("All files downloaded.")


if __name__ == "__main__":
    main()
