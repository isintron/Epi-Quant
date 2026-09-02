"""Neighbour-set agreement, as a second opinion on a tracking link.

One shared scoring function, used by three callers that previously had none:
`suggest_match_candidates` (as a confirmer), `find_propagation_stop_frame` (as
a parallel stop condition), and the accept-and-revert loop (as a score
orthogonal to what it searches).

Tissue Analyzer's equivalent is `pairwise_score`
(`tracking_error_detector_and_fixer.py:621`), the Dice coefficient of two
cells' neighbour sets. It looks only at *who is around a cell*, never at where
the cell is -- which is exactly why it is worth having beside the centroid
criteria, all three of which look only at position.

**The asymmetry is the design constraint, and it is measured.** On 7,007 GBE
links, judged independently by each cell's best-IoU successor (baseline
disagreement 11.52%):

- topology alone is NOT a detector -- the topology-only-flagged group disagrees
  at 11.8% / 15.5% / 13.2% at `dice <= 0.5 / 0.6 / 0.75`, indistinguishable
  from the 11.5% baseline;
- both criteria flagged together reaches **94.3%**, 8.2x baseline;
- as a decision rule on the 781 geometry-flagged links, pure geometry agrees
  with the judge 66.8%; when topology also says swap (n=98) that rises to
  **81.6%**; when topology says keep, the judge agrees with the current link
  only **30.5%**.

So: **topology is a good confirmer and a bad vetoer.** Never let it alone block
a re-assignment. `suggest_match_candidates` uses it as an AND (rank a candidate
up when both agree) and never demotes on a topology veto; `find_propagation_stop_frame`
uses it as an OR (stop at whichever criterion stops first), because stopping is
conservative while re-assigning is aggressive.
"""

from typing import Mapping, Optional, Set

import adjacency

# TA's repair tier, not its look tier. `take_decision` flags at 0.75 and
# `optimize_score` only attempts a repair at <= 0.5; conflating the two is what
# produced the 64.8%-vs-22.5% dead-band discrepancy in an earlier round.
REPAIR_THRESHOLD = 0.5
LOOK_THRESHOLD = 0.75


def neighbour_dice(
    graph_a: Mapping[int, Set[int]],
    graph_b: Mapping[int, Set[int]],
    label_a: int,
    label_b: int,
    label_to_track_a: Mapping[int, Optional[int]],
    label_to_track_b: Mapping[int, Optional[int]],
) -> Optional[float]:
    """Dice of `label_a`'s neighbour set at frame A against `label_b`'s at
    frame B, compared in TRACK space. `None` when neither has a neighbour that
    can be identified across the two frames.

    Track space, not label space: segmentation renumbers labels every frame, so
    a raw-label comparison would score a perfectly stable neighbourhood at 0.
    Untracked neighbours are dropped from both sides rather than counted as
    mismatches -- with no cross-frame identity they can neither confirm nor
    deny.

    **Unweighted set Dice, deliberately.** The contact-weighted variant was
    measured worse (mean 0.8464 with 18.5% below 0.7) because it penalises a
    contact that is preserved but shrinking -- which is the normal state of a
    bond approaching a T1, not an error.
    """
    set_a = {label_to_track_a[n] for n in graph_a.get(label_a, set())
             if label_to_track_a.get(n) is not None}
    set_b = {label_to_track_b[n] for n in graph_b.get(label_b, set())
             if label_to_track_b.get(n) is not None}
    if not set_a and not set_b:
        return None
    return 2.0 * len(set_a & set_b) / (len(set_a) + len(set_b))


def swap_verdict(
    graph_prev: Mapping[int, Set[int]],
    graph_now: Mapping[int, Set[int]],
    label_to_track_prev: Mapping[int, Optional[int]],
    label_to_track_now: Mapping[int, Optional[int]],
    current_label: int,
    rival_label: int,
    anchor_label: int,
) -> dict:
    """Does the neighbourhood favour keeping `anchor`'s link to `current`, or
    swapping it to `rival`?

    Returns `{"s_current", "s_rival", "verdict"}` with `verdict` in
    `{"swap", "keep", "unknown"}`. `unknown` when either score is `None` --
    reported rather than silently resolved, because a cell with no identifiable
    neighbours has not voted.

    Used as a CONFIRMER only. A `keep` verdict must never block a re-assignment
    the geometry proposed: measured, when topology says keep, the independent
    judge still sides with the current link only 30.5% of the time.
    """
    s_current = neighbour_dice(graph_prev, graph_now, anchor_label, current_label,
                               label_to_track_prev, label_to_track_now)
    s_rival = neighbour_dice(graph_prev, graph_now, anchor_label, rival_label,
                             label_to_track_prev, label_to_track_now)
    if s_current is None or s_rival is None:
        verdict = "unknown"
    elif s_rival > s_current:
        verdict = "swap"
    else:
        verdict = "keep"
    return {"s_current": s_current, "s_rival": s_rival, "verdict": verdict}


def settled_by_topology(
    graph_prev: Mapping[int, Set[int]],
    graph_now: Mapping[int, Set[int]],
    label_to_track_prev: Mapping[int, Optional[int]],
    label_to_track_now: Mapping[int, Optional[int]],
    label_prev: int,
    label_now: int,
    threshold: float = REPAIR_THRESHOLD,
) -> bool:
    """Has this cell's neighbourhood settled, by TA's repair tier (0.5)?

    Used as a parallel stop condition for smart-stop propagation, taken as
    `min(geometric stop, topology stop)`. 0.5 and not 0.75 on purpose: 0.75 is
    the tier at which TA merely *looks* at a cell, and using it here would stop
    propagation on cells TA considers clean.
    """
    score = neighbour_dice(graph_prev, graph_now, label_prev, label_now,
                           label_to_track_prev, label_to_track_now)
    return score is None or score >= threshold


def mean_neighbour_dice(
    graph_a: Mapping[int, Set[int]],
    graph_b: Mapping[int, Set[int]],
    pairs,
    label_to_track_a: Mapping[int, Optional[int]],
    label_to_track_b: Mapping[int, Optional[int]],
) -> Optional[float]:
    """Mean `neighbour_dice` over `pairs` of `(label_a, label_b)`.

    This is the score for the accept-and-revert loop. It is deliberately NOT
    the quantity smart-stop searches over: that decides by centroid-based
    neighbour settlement, so scoring its result by the same family would be
    self-confirming -- the failure this project has already made three times.
    """
    scores = [s for s in (neighbour_dice(graph_a, graph_b, a, b, label_to_track_a, label_to_track_b)
                          for a, b in pairs) if s is not None]
    if not scores:
        return None
    return float(sum(scores) / len(scores))


if __name__ == "__main__":
    # A cell whose neighbourhood is preserved scores 1.0 even though every raw
    # label changed -- the property that makes track space mandatory.
    graph_prev = {1: {2, 3, 4}, 2: {1}, 3: {1}, 4: {1}}
    graph_now = {9: {8, 7, 6}, 8: {9}, 7: {9}, 6: {9}}
    l2t_prev = {1: 100, 2: 200, 3: 300, 4: 400}
    l2t_now = {9: 100, 8: 200, 7: 300, 6: 400}
    assert neighbour_dice(graph_prev, graph_now, 1, 9, l2t_prev, l2t_now) == 1.0
    print("preserved neighbourhood scores 1.0 despite every raw label changing")

    # One neighbour exchanged -> 2*2/(3+3)
    swapped_now = {9: {8, 7, 5}, 8: {9}, 7: {9}, 5: {9}}
    l2t_swapped = {9: 100, 8: 200, 7: 300, 5: 500}
    score = neighbour_dice(graph_prev, swapped_now, 1, 9, l2t_prev, l2t_swapped)
    assert abs(score - 2 / 3) < 1e-9, score
    print("one neighbour exchanged scores %.3f" % score)

    # A cell with no identifiable neighbours has not voted.
    assert neighbour_dice({1: set()}, {9: set()}, 1, 9, {1: 100}, {9: 100}) is None
    print("a cell with no identifiable neighbours returns None, not 0.0")

    # swap_verdict: the rival with the better-preserved neighbourhood wins,
    # and an unscoreable pair is reported as unknown rather than resolved.
    verdict = swap_verdict(graph_prev, {9: {8, 7, 6}, 5: {2}}, l2t_prev,
                           {9: 100, 8: 200, 7: 300, 6: 400, 5: 500, 2: 999},
                           current_label=5, rival_label=9, anchor_label=1)
    print("swap_verdict:", verdict)
    assert verdict["verdict"] == "swap", verdict
    unknown = swap_verdict({1: set()}, {9: set(), 5: set()}, {1: 100},
                           {9: 100, 5: 500}, current_label=5, rival_label=9, anchor_label=1)
    assert unknown["verdict"] == "unknown", unknown
    print("  -> an unscoreable comparison is 'unknown', never a silent 'keep'")

    # settled_by_topology uses the REPAIR tier, not the look tier: a score of
    # 0.667 is settled at 0.5 and unsettled at 0.75.
    assert settled_by_topology(graph_prev, swapped_now, l2t_prev, l2t_swapped, 1, 9, threshold=0.5)
    assert not settled_by_topology(graph_prev, swapped_now, l2t_prev, l2t_swapped, 1, 9, threshold=0.75)
    print("settled_by_topology: 0.667 is settled at the 0.5 repair tier, not at the 0.75 look tier")

    print("\nAll topology_score self-tests passed.")
