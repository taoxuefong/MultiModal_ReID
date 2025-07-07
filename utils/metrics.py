import numpy as np

def compute_rank1_mAP(distmat, query_ids, gallery_ids):
    # distmat: [num_query, num_gallery]
    # query_ids, gallery_ids: list or np.array
    num_query = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    all_cmc = []
    all_AP = []
    for i in range(num_query):
        cmc = matches[i]
        if not np.any(cmc):
            continue
        cmc = cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:20])
        # AP
        num_rel = cmc.sum()
        tmp_cmc = matches[i].astype(np.float32)
        d_recall = 1.0 / num_rel
        precision = tmp_cmc.cumsum() / (np.arange(len(tmp_cmc)) + 1)
        AP = (precision * tmp_cmc).sum() * d_recall
        all_AP.append(AP)
    if len(all_cmc) == 0:
        return 0, 0
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / len(all_cmc)
    mAP = np.mean(all_AP)
    return cmc[0], mAP 