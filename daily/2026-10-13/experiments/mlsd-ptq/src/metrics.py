"""Line AP on a labelled subset; explicit protocol, not the paper benchmark."""

import numpy as np


def numerical(reference, actual):
    a = np.asarray(reference, np.float64)
    b = np.asarray(actual, np.float64)
    error = b - a
    power = np.sum(a * a)
    noise = np.sum(error * error)
    return {
        "snr_db": None
        if noise == 0
        else float(10 * (np.log10(max(power, 1e-300)) - np.log10(noise))),
        "exact_match": bool(noise == 0),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "relative_l2": float(np.sqrt(noise / max(power, 1e-300))),
    }


def lines(raw, minimum_score=0.01):
    heat = 1 / (1 + np.exp(-np.clip(raw[:, :, 0], -80, 80)))
    padded = np.pad(heat, 1, constant_values=0)
    maximum = np.maximum.reduce(
        [padded[y : y + 256, x : x + 256] for y in range(3) for x in range(3)]
    )
    heat = np.where(heat == maximum, heat, 0)
    ids = np.argsort(-heat.reshape(-1), kind="stable")[:200]
    result = []
    for index in ids:
        y, x = divmod(int(index), 256)
        score = float(heat[y, x])
        d = raw[y, x, 1:5]
        if score >= minimum_score and np.linalg.norm(d[:2] - d[2:]) > 20:
            result.append(
                (np.array([[x + d[0], y + d[1]], [x + d[2], y + d[3]]], np.float64) / 2, score)
            )
    return result


def endpoint_distances(predicted, truth):
    same = np.sum((predicted[:, None] - truth[None]) ** 2, axis=(2, 3))
    reverse = np.sum((predicted[:, None] - truth[None, :, ::-1]) ** 2, axis=(2, 3))
    return np.minimum(same, reverse)


def line_metrics(outputs, ground_truth):
    scores = []
    true_positive = []
    gt_count = 0
    counts = []
    for raw, row in zip(outputs, ground_truth):
        w, h = row["image_size"]
        truth = np.asarray(row["lines"], np.float64).reshape(-1, 2, 2) * np.array(
            [128 / w, 128 / h]
        )
        gt_count += len(truth)
        predictions = lines(raw)
        counts.append(len(predictions))
        if not predictions:
            continue
        if not len(truth):
            scores.extend(score for _, score in predictions)
            true_positive.extend([False] * len(predictions))
            continue
        predicted = np.stack([p for p, _ in predictions])
        distance = endpoint_distances(predicted, truth)
        nearest = np.argmin(distance, axis=1)
        used = np.zeros(len(truth), bool)
        for i, (_, score) in enumerate(predictions):
            j = nearest[i]
            matched = bool(distance[i, j] < 10 and not used[j])
            if matched:
                used[j] = True
            scores.append(score)
            true_positive.append(matched)
    scores = np.asarray(scores)
    tp = np.asarray(true_positive, np.float64)
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    scores = scores[order]
    cumulative = np.cumsum(tp)
    recall = cumulative / max(gt_count, 1)
    precision = cumulative / np.arange(1, len(tp) + 1)
    mrec = np.r_[0, recall, 1]
    mpre = np.r_[0, precision, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]) * 100
    chosen = scores >= 0.1
    matched = tp[chosen].sum()
    return {
        "sAP10_subset_percent": float(ap),
        "precision_at_0_1": float(matched / max(chosen.sum(), 1)),
        "recall_at_0_1": float(matched / max(gt_count, 1)),
        "ground_truth_lines": gt_count,
        "predicted_line_counts": counts,
        "protocol": "NMS 3x3, top200, score>=0.01, map length>20; endpoint-swap invariant squared distance sum<10 on 128x128 coordinates; one-to-one greedy matches; continuous AP. Small York subset, not paper protocol.",
    }
