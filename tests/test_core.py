import numpy as np
import pytest
from src.boxes import xyxy_to_cxcywh, cxcywh_to_xyxy, box_area, intersection_area, union_area
from src.iou import calculate_iou, iou_matrix
from src.nms import non_max_suppression, non_max_suppression_simple, class_aware_nms, confidence_filter
from src.metrics import precision_recall, pr_curve, average_precision
from src.evaluation import Evaluator


# ---------------------------------------------------------------- boxes
def test_box_conversion_round_trip():
    b = np.array([[100, 50, 300, 250], [0, 0, 10, 20]], float)
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b)
    assert np.allclose(xyxy_to_cxcywh([100, 50, 300, 250]), [200, 150, 200, 200])  # worked example from the notes


def test_area_degenerate_is_zero():
    assert box_area([5, 5, 5, 10]) == 0
    assert box_area([10, 10, 0, 0]) == 0


# ---------------------------------------------------------------- IoU
@pytest.mark.parametrize("a,b,expected", [
    ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),                   # identical
    ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),                 # disjoint
    ([0, 0, 10, 10], [10, 0, 20, 10], 0.0),                  # touching edge only
    ([0, 0, 10, 10], [5, 0, 15, 10], 50 / 150),              # half overlap
    ([0, 0, 10, 10], [2, 2, 8, 8], 36 / 100),                # one box inside the other
    ([0, 0, 0, 0], [0, 0, 0, 0], 0.0),                       # zero-area boxes
])
def test_iou_known_values(a, b, expected):
    assert calculate_iou(a, b) == pytest.approx(expected)


def test_iou_symmetric_and_matrix_matches_scalar():
    rng = np.random.RandomState(0)
    a = np.sort(rng.rand(15, 2, 2) * 100, axis=1).reshape(15, 4)[:, [0, 2, 1, 3]]
    b = np.sort(rng.rand(12, 2, 2) * 100, axis=1).reshape(12, 4)[:, [0, 2, 1, 3]]
    m = iou_matrix(a, b)
    for i in range(15):
        for j in range(12):
            assert m[i, j] == pytest.approx(calculate_iou(a[i], b[j]))
            assert calculate_iou(a[i], b[j]) == pytest.approx(calculate_iou(b[j], a[i]))


# ---------------------------------------------------------------- NMS
BOXES = [[10, 10, 100, 100], [15, 15, 95, 95], [150, 150, 200, 200]]
SCORES = [0.95, 0.91, 0.40]


def test_nms_example_from_notes():
    assert non_max_suppression(BOXES, SCORES, 0.5) == [0, 2]


def test_nms_threshold_direction():
    # IoU(A,B)=0.5: kept only when threshold >= 0.5 (suppress if IoU > threshold)
    boxes = [[0, 0, 10, 10], [0, 0, 10, 15]]            # IoU = 100/150 = 0.667
    assert len(non_max_suppression(boxes, [0.9, 0.8], 0.5)) == 1
    assert len(non_max_suppression(boxes, [0.9, 0.8], 0.7)) == 2
    assert len(non_max_suppression(boxes, [0.9, 0.8], 0.3)) == 1


def test_vectorised_nms_equals_simple_nms():
    rng = np.random.RandomState(1)
    xy = rng.rand(200, 2) * 100
    wh = rng.rand(200, 2) * 40 + 5
    boxes = np.concatenate([xy, xy + wh], 1)
    scores = rng.rand(200)
    for t in (0.3, 0.5, 0.7):
        assert non_max_suppression(boxes, scores, t) == non_max_suppression_simple(boxes, scores, t)


def test_class_aware_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    assert len(class_aware_nms(boxes, [0.9, 0.8], [0, 1], 0.5)) == 2
    assert len(class_aware_nms(boxes, [0.9, 0.8], [0, 0], 0.5)) == 1


def test_confidence_filter():
    b, s = confidence_filter(BOXES, SCORES, 0.5)
    assert len(b) == 2 and np.allclose(s, [0.95, 0.91])


def test_empty_inputs():
    assert non_max_suppression(np.zeros((0, 4)), np.zeros(0), 0.5) == []


# ---------------------------------------------------------------- metrics
def test_precision_recall_cat_example():
    # notes: TP=2, FP=1, FN=1 -> both 2/3
    p, r = precision_recall(2, 1, 1)
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    assert precision_recall(0, 0, 0) == (0.0, 0.0)


def test_ap_perfect_and_known_curve():
    rec, prec = pr_curve([1, 1, 1], 3)
    assert average_precision(rec, prec) == pytest.approx(1.0)
    # TP, FP, TP over 2 GT: precision 1, .5, .667 ; recall .5, .5, 1
    rec, prec = pr_curve([1, 0, 1], 2)
    assert average_precision(rec, prec) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert average_precision([], []) == 0.0


def test_ap_11_point_perfect():
    rec, prec = pr_curve([1, 1], 2)
    assert average_precision(rec, prec, "11_point") == pytest.approx(1.0)


# ---------------------------------------------------------------- evaluator
def _toy():
    gts = {"a": {"boxes": np.array([[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90]], float),
                 "labels": np.array([0, 0, 0]), "difficult": np.array([False, False, True])}}
    dets = {"a": {"boxes": np.array([[0, 0, 10, 10],      # TP
                                     [1, 1, 10, 10],      # duplicate of GT 0 -> FP
                                     [50, 50, 60, 60],    # TP
                                     [80, 80, 90, 90],    # matches difficult GT -> ignored
                                     [200, 200, 210, 210]], float),   # background -> FP
                  "scores": np.array([0.9, 0.8, 0.7, 0.6, 0.5]), "labels": np.zeros(5, int)}}
    return gts, dets


def test_evaluator_counts_follow_voc_protocol():
    gts, dets = _toy()
    m = Evaluator(gts, dets, num_classes=1).metrics(0.5)
    assert (m["tp"], m["fp"], m["fn"], m["n_gt"]) == (2, 2, 0, 2)       # difficult GT not counted
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(1.0)
    # sorted TP,FP,TP,FP -> precision 1,.5,.667,.5 ; AP = 0.5*1 + 0.5*(2/3)
    assert m["map"] == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_evaluator_confidence_threshold_truncates():
    gts, dets = _toy()
    m = Evaluator(gts, dets, num_classes=1).metrics(0.5, conf_thr=0.85)
    assert (m["tp"], m["fp"]) == (1, 0) and m["recall"] == pytest.approx(0.5)
