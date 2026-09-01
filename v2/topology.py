"""T1 transitions / neighbour exchanges.

Ported from TissueMiner (`topology/CountT1.R`, and the orientation nematic in
`commons/BaseQueryFunctions.R:414-420`), which is BSD-3-Clause licensed:

    Copyright (c) 2015, Scientific Computing Facility @ MPI-CBG
    All rights reserved. Redistribution and use in source and binary forms,
    with or without modification, are permitted provided that the conditions
    in `third_party/tissue_miner_LICENSE` are met. THIS SOFTWARE IS PROVIDED
    BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
    IMPLIED WARRANTIES ARE DISCLAIMED.

The full licence text is in `third_party/tissue_miner_LICENSE`. TissueMiner's
authors ask (as a courtesy, not a licence condition) that work using it cite
Etournay et al. 2016, eLife 5:e14334.

What is ported and what is not
------------------------------
TissueMiner derives its per-frame neighbour list from a half-edge
(`directed_bonds`) table in a SQLite database built by its own C++ parser.
None of that is needed here: `adjacency.build_neighbor_graph` already produces
the neighbour sets, and the tracking table already supplies persistent ids. So
what is ported is the *criterion*, not the plumbing -- a time-shifted set
difference over neighbour pairs, plus the nematic.

TissueMiner also collapses a division by relabelling both daughters with the
mother's id for one frame, which needs a lineage table Epi-Quant's tracking
does not produce (a division here simply looks like one track ending and
another starting). Its OTHER artefact filter -- requiring both cells to be
present in both frames -- IS ported, and removes most division and
appearance/disappearance noise. A division whose daughters both persist can
still register; `detect_t1_transitions`' docstring says so.

Like TissueMiner, this counts **half-events per cell**, not four-cell
quartets: a canonical T1 among A,B (losing contact) and C,D (gaining it)
appears as one lost pair and one gained pair, incrementing `lost` for A and B
and `gained` for C and D. Losses are not matched to their partner gains.
"""

from typing import Dict, Mapping, Optional, Set

import numpy as np

# Contacts are compared in track-id space, so a cell must be tracked in BOTH
# frames to say anything about its neighbourhood changing.
__all__ = [
    "contacts_in_track_space", "detect_t1_transitions", "t1_counts_per_cell",
    "annotate_persistence", "validate_rosette_quartet",
]


def contacts_in_track_space(
    graph: Mapping[int, Set[int]], label_to_track: Mapping[int, Optional[int]],
    previous_contacts: Optional[Set[tuple]] = None,
    enter_px: Optional[float] = None, exit_px: Optional[float] = None,
) -> Set[tuple]:
    """`{(low_track, high_track), ...}` -- one entry per cell-cell contact,
    keyed by persistent track id rather than by the frame's own labels.

    Raw labels are reassigned arbitrarily by segmentation each frame, so a
    set difference over them would be meaningless. Untracked cells are
    dropped: with no cross-frame identity, a contact involving one can be
    neither confirmed nor denied.

    Hysteresis (a Schmitt trigger)
    ------------------------------
    With `enter_px`/`exit_px` set (`enter_px > exit_px`) and `graph` in the
    weighted form from `adjacency.build_neighbor_graph(..., return_weights=True)`,
    a contact is:

        present      once its strength rises above `enter_px`
        absent       once its strength falls below `exit_px`
        UNCHANGED    in between -- it keeps `previous_contacts`' verdict

    **This is structurally different from a hard threshold, and the difference
    is the whole point.** Round 9 measured that a single interface-length
    cutoff makes flicker WORSE, not better (20 px -> 1,832 gains against 1,505,
    with reversion rising to 31.6%): a smoothly shrinking bond crosses one
    threshold repeatedly as noise pushes it either side. Two thresholds with a
    dead zone between them cannot do that -- a bond that has fallen below
    `exit_px` must climb all the way back above `enter_px` to be counted
    present again, so noise of less than the band width produces no event at
    all.

    `previous_contacts` is the previous frame's result, which is what carries
    the state; without it the band resolves to "present above `enter_px`" and
    the trigger degenerates to a hard threshold at `enter_px`.
    """
    use_hysteresis = enter_px is not None and exit_px is not None
    if use_hysteresis and not (enter_px > exit_px):
        raise ValueError(f"enter_px ({enter_px}) must exceed exit_px ({exit_px}) for hysteresis")

    contacts = set()
    for label, neighbours in graph.items():
        a = label_to_track.get(label)
        if a is None:
            continue
        for neighbour in neighbours:
            b = label_to_track.get(neighbour)
            if b is None or b == a:
                continue
            pair = (min(a, b), max(a, b))
            if not use_hysteresis:
                contacts.add(pair)
                continue
            strength = neighbours[neighbour] if isinstance(neighbours, Mapping) else None
            if strength is None:
                contacts.add(pair)
            elif strength > enter_px:
                contacts.add(pair)
            elif strength >= exit_px and previous_contacts is not None and pair in previous_contacts:
                contacts.add(pair)   # inside the band: hold the previous verdict
    return contacts


def detect_t1_transitions(
    graph_t: Mapping[int, Set[int]],
    graph_t1: Mapping[int, Set[int]],
    label_to_track_t: Mapping[int, Optional[int]],
    label_to_track_t1: Mapping[int, Optional[int]],
    centroids_t: Mapping[int, tuple] = None,
) -> dict:
    """Neighbour exchanges between two consecutive frames.

    Returns `{"gained": [...], "lost": [...]}` where each entry is
    `{"tracks": (a, b), "type": "gain"|"loss", "nematic": (qxx, qxy),
    "phi": float, "center": (x, y)}` -- `nematic`/`phi`/`center` present only
    when `centroids_t` (a `{track_id: (x, y)}` map for frame t) is given.

    A contact present at `t` but not `t+1` is a LOSS; present at `t+1` but not
    `t` is a GAIN. Both cells must appear in both frames, which is
    TissueMiner's filter for appearance/disappearance artefacts -- without it
    a cell entering the field of view registers a gain against every
    neighbour it lands beside.

    **The orientation is a nematic (a headless axis), and the sign flip for
    gains is the load-bearing part.** From the two cells' centroid separation
    `(dx, dy)`: `Qxx = (dx^2 - dy^2)/2`, `Qxy = dx*dy`, normalised to unit
    length, then NEGATED for a gain. That rotates a gain's axis by 90 degrees
    so it lies along the newly formed bond rather than along the separation of
    the cells forming it. Without the flip, the losses and gains of one T1
    point across each other and cancel when averaged; with it they reinforce,
    which is what makes a mean nematic over a region meaningful -- the
    measurement that matters for convergent extension.

    Caveat: a cell division whose daughters both persist can still register as
    a T1, since Epi-Quant has no lineage table to collapse it with.
    """
    present_t = {t for t in label_to_track_t.values() if t is not None}
    present_t1 = {t for t in label_to_track_t1.values() if t is not None}
    present_both = present_t & present_t1

    contacts_t = contacts_in_track_space(graph_t, label_to_track_t)
    contacts_t1 = contacts_in_track_space(graph_t1, label_to_track_t1)

    def _judgeable(pair):
        return pair[0] in present_both and pair[1] in present_both

    events = {"gained": [], "lost": []}
    for pair in sorted(contacts_t - contacts_t1):
        if _judgeable(pair):
            events["lost"].append({"tracks": pair, "type": "loss"})
    for pair in sorted(contacts_t1 - contacts_t):
        if _judgeable(pair):
            events["gained"].append({"tracks": pair, "type": "gain"})

    if centroids_t:
        for entry in events["lost"] + events["gained"]:
            a, b = entry["tracks"]
            if a not in centroids_t or b not in centroids_t:
                continue
            (x1, y1), (x2, y2) = centroids_t[a], centroids_t[b]
            dx, dy = x2 - x1, y2 - y1
            qxx, qxy = 0.5 * (dx * dx - dy * dy), dx * dy
            norm = float(np.hypot(qxx, qxy))
            if norm <= 0:
                continue
            sign = -1.0 if entry["type"] == "gain" else 1.0
            unit_xx, unit_xy = sign * qxx / norm, sign * qxy / norm
            entry["nematic"] = (unit_xx, unit_xy)
            entry["phi"] = float(0.5 * np.arctan2(unit_xy, unit_xx))
            entry["center"] = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
    return events


def annotate_persistence(events: dict, contacts_after: list, persist_frames: int = 1) -> dict:
    """Mark each event in `detect_t1_transitions`' result with whether the
    change it describes actually held. Sets three fields on every entry, in
    place, and returns `events`:

        "state"            "persistent" | "transient" | "unjudgeable"
        "persistent"       True only for "persistent" (kept for callers)
        "persisted_frames" how many further frames it was confirmed for

    `contacts_after` is the contact set (from `contacts_in_track_space`) for
    each of the frames FOLLOWING the `t+1` the events were detected against,
    nearest first. A gain persists while its new contact is still present; a
    loss persists while the broken contact stays absent.

    **Why there are three states and not two.** Near the end of a movie there
    are not `persist_frames` frames left to look at. The obvious
    `horizon = min(persist_frames, len(contacts_after))` silently weakens the
    requirement to whatever lookahead happens to exist, and when
    `contacts_after` is empty the horizon becomes 0, `held >= 0` is always
    true, and **every event in those frame pairs is marked persistent
    unconditionally**. That is not a neutral default: the unjudgeable frames
    all sit at the end of the movie, which on `embryos_GBE_GAP43` is where
    germband extension is most active. Measured, at `persist_frames=1` it
    inflated the settled count by 55 gains and 39 losses; at 2, by 100 and 86.
    An event that cannot be judged is now reported as such instead of being
    quietly counted as settled.

    **Nothing is dropped.** A one-frame flicker may be noise or may be a real
    reversible exchange, and this module cannot tell the two apart -- so the
    caller gets the evidence and decides, the same curation split `lineage.py`
    uses for divisions and delaminations.

    TissueMiner has no equivalent. `topology/CountT1.R` is a raw time-shifted
    set difference; its only filters are the margin cell, presence in both
    frames, and the division relabel. It does not need one, because its
    outputs are ROI rates and nematic averages (`mqf_cg_roi_rate_T1`,
    `mqf_fg_unit_nematics_T1`) in which a gain and its immediate reversal
    cancel. Epi-Quant draws individual events on individual frames, which is
    exactly where a flicker stops cancelling and starts looking like a finding.

    Measured on `embryos_GBE_GAP43`, counting only judgeable events: requiring
    one further frame takes gains 1,505 -> 1,047 and losses 1,545 -> 1,225;
    requiring two takes them to 806 and 1,074.

    `persist_frames=0` marks everything persistent with nothing unjudgeable --
    the unfiltered TissueMiner behaviour.
    """
    horizon = int(persist_frames)
    judgeable = len(contacts_after) >= horizon
    for entry in events["gained"] + events["lost"]:
        pair = entry["tracks"]
        wanted = entry["type"] == "gain"  # a gain must stay a contact; a loss must stay absent
        held = 0
        for contacts in contacts_after[:horizon]:
            if (pair in contacts) != wanted:
                break
            held += 1
        entry["persisted_frames"] = held
        if held >= horizon:
            # Confirmed for the full horizon -- reachable even when the
            # lookahead is short only if the horizon is 0.
            entry["state"] = "persistent"
        elif judgeable or held < len(contacts_after):
            # It actually reversed within the frames available, which is a
            # verdict regardless of how many more frames there would have been.
            entry["state"] = "transient"
        else:
            entry["state"] = "unjudgeable"
        entry["persistent"] = entry["state"] == "persistent"
    return events


def validate_rosette_quartet(
    events: dict,
    contacts_t: Set[tuple],
    contacts_t1: Set[tuple],
) -> dict:
    """napariCellFlow's four-cell rosette check, as a NAMED ALTERNATIVE.

    Marks each gained event `rosette: True` when a lost pair can be found that
    forms a canonical T1 quartet with it: the four cells distinct, and all four
    connecting edges (each lost cell to each gained cell) present in BOTH
    frames. Annotates; drops nothing.

    Ported criterion, not ported plumbing, from napariCellFlow
    (`napariCellFlow/edge_analysis.py:183 _validate_t1_transition`): reject
    unless `len(lost_edge | gained_edge) == 4` and every connecting edge exists
    at t and t+1.

    **`detect_t1_transitions` is deliberately left untouched.** It is the
    ported TissueMiner criterion and the module docstring draws a line around
    what is ported; this is a different definition of the event, not a filter
    on that one, so it lives beside it with its own provenance.

    On `embryos_GBE_GAP43` it accepts 43.2% of gains and rejects 56.8%. That
    56.8% is an UPPER bound on what is false, not an estimate of it: a real T1
    with one untracked cell, or with a ring edge broken by segmentation, is
    rejected too.

    **napariCellFlow's `min_edge_length` / `min_overlap_pixels` filters are
    deliberately NOT adopted.** Those are hard length cutoffs, and Round 9
    measured a hard cutoff making flicker worse (20 px -> 1,827 gains and 31.7%
    reversal against 1,505 and 21.0% unfiltered). `contacts_in_track_space`'s
    hysteresis is the tool for that, not a threshold.
    """
    lost_pairs = [entry["tracks"] for entry in events["lost"]]
    neighbours_t: Dict[int, Set[int]] = {}
    for a, b in contacts_t:
        neighbours_t.setdefault(a, set()).add(b)
        neighbours_t.setdefault(b, set()).add(a)

    accepted = 0
    for entry in events["gained"]:
        c, d = entry["tracks"]
        shared = neighbours_t.get(c, set()) & neighbours_t.get(d, set())
        entry["rosette"] = False
        for a, b in lost_pairs:
            if a not in shared or b not in shared:
                continue
            if len({a, b, c, d}) != 4:
                continue
            ring = {(min(p, q), max(p, q)) for p, q in ((a, c), (c, b), (b, d), (d, a))}
            if ring <= contacts_t and ring <= contacts_t1:
                entry["rosette"] = True
                entry["rosette_partner"] = (a, b)
                accepted += 1
                break
    return {
        "gained": len(events["gained"]), "rosette": accepted,
        "rejected": len(events["gained"]) - accepted,
    }


def t1_counts_per_cell(events: dict) -> Dict[int, dict]:
    """`{track_id: {"gained": n, "lost": n}}` from `detect_t1_transitions`.

    Both members of a pair are credited, matching TissueMiner's per-cell
    half-event counting.
    """
    counts: Dict[int, dict] = {}
    for key, field in (("gained", "gained"), ("lost", "lost")):
        for entry in events[key]:
            for track in entry["tracks"]:
                counts.setdefault(track, {"gained": 0, "lost": 0})[field] += 1
    return counts


if __name__ == "__main__":
    # A canonical T1: four cells around a shared vertex. A-B are in contact at
    # t and separate; C-D are apart at t and come together. Neighbour sets are
    # written directly, so this tests the criterion rather than the geometry.
    #        t                    t+1
    #      C                    C
    #   A --- B      ->      A  |  B
    #      D                    D
    graph_t = {1: {2, 3, 4}, 2: {1, 3, 4}, 3: {1, 2}, 4: {1, 2}}
    graph_t1 = {1: {3, 4}, 2: {3, 4}, 3: {1, 2, 4}, 4: {1, 2, 3}}
    identity = {i: i for i in (1, 2, 3, 4)}
    centroids = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (5.0, -5.0), 4: (5.0, 5.0)}

    events = detect_t1_transitions(graph_t, graph_t1, identity, identity, centroids)
    print("lost :", [e["tracks"] for e in events["lost"]])
    print("gained:", [e["tracks"] for e in events["gained"]])
    assert [e["tracks"] for e in events["lost"]] == [(1, 2)], events["lost"]
    assert [e["tracks"] for e in events["gained"]] == [(3, 4)], events["gained"]
    print("  -> exactly one lost pair (A-B) and one gained pair (C-D).")

    loss_phi, gain_phi = events["lost"][0]["phi"], events["gained"][0]["phi"]
    print("nematic angle: loss=%.1f deg  gain=%.1f deg" % (np.degrees(loss_phi), np.degrees(gain_phi)))
    # A-B separate horizontally, C-D approach vertically. The gain's sign flip
    # must bring the two axes into ALIGNMENT, not leave them perpendicular.
    lost_vec = events["lost"][0]["nematic"]
    gain_vec = events["gained"][0]["nematic"]
    alignment = lost_vec[0] * gain_vec[0] + lost_vec[1] * gain_vec[1]
    print("  nematic alignment (dot product): %+.2f" % alignment)
    assert alignment > 0.9, (
        "gain and loss of one T1 must reinforce, not cancel -- this is what the sign flip buys"
    )
    print("  -> the two halves of the T1 reinforce, so a regional mean is meaningful.")

    counts = t1_counts_per_cell(events)
    print("per-cell counts:", counts)
    assert counts[1] == {"gained": 0, "lost": 1} and counts[2] == {"gained": 0, "lost": 1}, counts
    assert counts[3] == {"gained": 1, "lost": 0} and counts[4] == {"gained": 1, "lost": 0}, counts
    print("  -> both cells of each pair are credited (half-event counting).")

    # A cell that only appears at t+1 must not register a gain against
    # everything it lands next to.
    graph_new = {1: {2, 5}, 2: {1, 5}, 5: {1, 2}}
    l2t_new = {1: 1, 2: 2, 5: 5}
    arriving = detect_t1_transitions({1: {2}, 2: {1}}, graph_new, {1: 1, 2: 2}, l2t_new)
    print("\narriving cell:", arriving["gained"])
    assert arriving["gained"] == [], "a cell absent at t cannot have gained a neighbour"
    print("  -> the both-frames presence filter suppresses arrival artefacts.")

    # --- persistence: the T1 above is real, so C-D stay together. A flicker
    # is the same detection followed by an immediate reversal.
    stable_after = [contacts_in_track_space(graph_t1, identity)] * 2
    flicker_after = [contacts_in_track_space(graph_t, identity)] * 2  # snaps back to t
    real = annotate_persistence(
        detect_t1_transitions(graph_t, graph_t1, identity, identity), stable_after, persist_frames=2,
    )
    flick = annotate_persistence(
        detect_t1_transitions(graph_t, graph_t1, identity, identity), flicker_after, persist_frames=2,
    )
    print("\npersistence: stable T1 gain=%s  flickering gain=%s"
          % (real["gained"][0]["state"], flick["gained"][0]["state"]))
    assert real["gained"][0]["state"] == "persistent" and real["lost"][0]["state"] == "persistent", real
    assert flick["gained"][0]["state"] == "transient" and flick["lost"][0]["state"] == "transient", flick
    assert flick["gained"][0]["persisted_frames"] == 0, flick["gained"][0]
    assert len(flick["gained"]) == 1 and len(flick["lost"]) == 1, "annotate, never drop"
    unfiltered = annotate_persistence(
        detect_t1_transitions(graph_t, graph_t1, identity, identity), flicker_after, persist_frames=0,
    )
    assert unfiltered["gained"][0]["state"] == "persistent", "persist_frames=0 is the unfiltered TissueMiner behaviour"
    print("  -> flickers are flagged, still returned, and persist_frames=0 flags nothing.")

    # The end of the movie: no lookahead at all. The old
    # `horizon = min(persist_frames, len(contacts_after))` made `held >= 0`
    # trivially true here and marked every such event settled.
    ran_out = annotate_persistence(
        detect_t1_transitions(graph_t, graph_t1, identity, identity), [], persist_frames=2,
    )
    print("no lookahead left: gain=%s" % ran_out["gained"][0]["state"])
    assert ran_out["gained"][0]["state"] == "unjudgeable", ran_out["gained"][0]
    assert not ran_out["gained"][0]["persistent"], "an unjudgeable event must not count as settled"
    # Partial lookahead that already contradicts the event IS a verdict.
    early = annotate_persistence(
        detect_t1_transitions(graph_t, graph_t1, identity, identity),
        [contacts_in_track_space(graph_t, identity)], persist_frames=3,
    )
    assert early["gained"][0]["state"] == "transient", early["gained"][0]
    print("  -> insufficient lookahead is 'unjudgeable', but an observed reversal still decides.")

    # --- hysteresis: a bond shrinking through the band cannot flicker
    weighted_wide = {1: {2: 40}, 2: {1: 40}}
    weighted_band = {1: {2: 20}, 2: {1: 20}}   # inside 25/15 -> hold previous
    weighted_thin = {1: {2: 5}, 2: {1: 5}}
    ident = {1: 1, 2: 2}
    wide = contacts_in_track_space(weighted_wide, ident, enter_px=25, exit_px=15)
    held = contacts_in_track_space(weighted_band, ident, previous_contacts=wide, enter_px=25, exit_px=15)
    gone = contacts_in_track_space(weighted_thin, ident, previous_contacts=held, enter_px=25, exit_px=15)
    back = contacts_in_track_space(weighted_band, ident, previous_contacts=gone, enter_px=25, exit_px=15)
    print("\nhysteresis: wide=%s band(after wide)=%s thin=%s band(after thin)=%s"
          % (bool(wide), bool(held), bool(gone), bool(back)))
    assert wide and held and not gone and not back, (wide, held, gone, back)
    print("  -> inside the band the previous verdict is held, in BOTH directions:")
    print("     a bond that fell out must climb back past enter_px, not just into the band.")
    try:
        contacts_in_track_space(weighted_band, ident, enter_px=15, exit_px=25)
    except ValueError:
        print("  -> enter_px <= exit_px rejected (that would be a hard threshold).")
    else:
        raise AssertionError("enter_px <= exit_px must be rejected")

    # --- rosette quartet: the canonical T1 passes, a lone gain does not
    ct = contacts_in_track_space(graph_t, identity)
    ct1 = contacts_in_track_space(graph_t1, identity)
    quartet = detect_t1_transitions(graph_t, graph_t1, identity, identity)
    summary = validate_rosette_quartet(quartet, ct, ct1)
    print("\nrosette check on the canonical T1:", summary)
    assert summary["rosette"] == 1 and quartet["gained"][0]["rosette"], summary
    lone = {"gained": [{"tracks": (3, 4), "type": "gain"}], "lost": []}
    lone_summary = validate_rosette_quartet(lone, ct, ct1)
    assert lone_summary["rosette"] == 0, lone_summary
    assert len(lone["gained"]) == 1, "annotate, never drop"
    print("  -> a gain with no matching loss is rejected, and still returned.")

    print("\nAll topology self-tests passed.")
