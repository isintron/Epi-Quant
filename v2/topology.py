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
    "annotate_persistence", "validate_rosette_quartet", "flatten_transient",
    "validate_quartet_permutation",
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
    enter_px: Optional[float] = None,
    exit_px: Optional[float] = None,
    previous_contacts_t: Optional[Set[tuple]] = None,
    previous_contacts_t1: Optional[Set[tuple]] = None,
) -> dict:
    """Neighbour exchanges between two consecutive frames.

    Returns `{"gained": [...], "lost": [...]}` where each entry is
    `{"tracks": (a, b), "type": "gain"|"loss", "nematic": (qxx, qxy),
    "phi": float, "center": (x, y)}` -- `nematic`/`phi`/`center` present only
    when `centroids_t` (a `{track_id: (x, y)}` map for frame t) is given.

    A contact present at `t` but not `t+1` is a LOSS; present at `t+1` but not
    `t` is a GAIN. Both cells must appear in both frames, without which a cell
    entering the field of view registers a gain against every neighbour it
    lands beside.

    **That filter is ours, not TissueMiner's** -- this docstring used to
    attribute it to them, wrongly. Measured on `WT_1`, 40 frame pairs of their
    own truth adjacency and tracking: we find 2,028 losses and 2,489 gains
    against `t1DataFilt`'s 2,129 and 2,620, we find **nothing they do not**,
    and the 101 / 131 they have and we do not are **exactly** the events in
    which a cell is absent from one of the two frames. So we are their
    criterion plus this filter; the reasoning for it stands, the attribution
    did not (Round 30 §D2).

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

    **`enter_px`/`exit_px` reach the detection here (Round 30 §A).** They did
    not until this round: this function built its two contact sets with a bare
    `contacts_in_track_space`, and the only caller that passed a band was
    `EpiQuantAPI._contacts_at`, whose sole consumer is `annotate_persistence`'s
    lookahead. Turning the band on therefore detected events on raw adjacency
    and judged them persistent on hysteretic adjacency -- two different
    definitions of "contact" on the two sides of one verdict, with the filter
    under test present on only one of them (`CLAUDE.md` rule 5). The band is
    still off by default and nothing changes when it is.

    Both contact sets carry state: `previous_contacts_t` is the band's verdict
    at `t-1`, and `previous_contacts_t1` defaults to the `t` set computed here,
    which is the chain the Schmitt trigger needs. Pass `previous_contacts_t1`
    explicitly only to break that chain deliberately.

    A band on an UNWEIGHTED graph is a no-op -- `contacts_in_track_space` sees
    `strength is None` and admits the pair unconditionally -- so a caller that
    sets it must pass `adjacency.build_neighbor_graph(..., return_weights=True)`.
    """
    present_t = {t for t in label_to_track_t.values() if t is not None}
    present_t1 = {t for t in label_to_track_t1.values() if t is not None}
    present_both = present_t & present_t1

    contacts_t = contacts_in_track_space(
        graph_t, label_to_track_t, previous_contacts=previous_contacts_t,
        enter_px=enter_px, exit_px=exit_px,
    )
    contacts_t1 = contacts_in_track_space(
        graph_t1, label_to_track_t1,
        previous_contacts=previous_contacts_t1 if previous_contacts_t1 is not None else contacts_t,
        enter_px=enter_px, exit_px=exit_px,
    )

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


def annotate_persistence(events: dict, contacts_after: list, persist_frames: int = 1,
                         detected_frame: Optional[int] = None) -> dict:
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

    Measured on `embryos_GBE_GAP43` (30 s/frame), counting only settled
    events, all rows from one run each, Round 30. The last row is Finegan et
    al. 2019's requirement: that movie is 30 s/frame germband, the same rate
    and tissue she set it on, so 5 frames IS her 150 s, unconverted.

        persist_frames   seconds   unregistered        mask-registered
                                   gains   losses      gains   losses
                     0        0    1,468    1,532        477      467
                     1       30    1,017    1,227        294      317
                     2       60      780    1,084        201      259
                     5      150      318      784        111      193

    **Two things this table says that one column could not.** First, the
    registration convention changes the raw count by 3.2x -- more than any
    persistence requirement does -- so a count quoted without it says little
    (`drift_mode='mask'` is what `embryos_GBE_GAP43`'s ground truth was
    collected under; measured agreement 99.7% against 85.5% unregistered).
    Second, the earlier record of 1,047/1,225 and 806/1,074 for rows 1 and 2
    was measured in an earlier round under settings it did not state; these
    rows replace them and differ by 2-3%.

    With `detected_frame`, two more fields are set on every entry:

        "detected_frame"  the frame the exchange was first seen at
        "settled_frame"   where it FIRST holds for the full horizon, which is
                          `detected_frame` for a persistent event and a later
                          frame for one that flickers and then settles; `None`
                          if it never settles inside `contacts_after`

    Finegan records a T1 at the frame the swap settles, not where it first
    appeared. Both are kept rather than one replaced by the other: their
    difference is the flicker delay, which is itself a diagnostic.
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
        if detected_frame is not None:
            entry["detected_frame"] = int(detected_frame)
            entry["settled_frame"] = _settles_at(
                pair, wanted, contacts_after, horizon, int(detected_frame))
    return events


def _settles_at(pair, wanted: bool, contacts_after: list, horizon: int, detected_frame: int):
    """The first frame from which `pair` holds `wanted` for `horizon`
    consecutive frames, or `None`."""
    if horizon <= 0:
        return detected_frame
    for start in range(0, len(contacts_after) - horizon + 1):
        window = contacts_after[start:start + horizon]
        if all((pair in contacts) == wanted for contacts in window):
            # contacts_after[0] describes frame detected_frame + 2.
            return detected_frame if start == 0 else detected_frame + 2 + start
    return None


def flatten_transient(events: dict) -> dict:
    """Finegan's smoothing, as an EXPLICIT step rather than a hidden filter.

    Returns `{"gained": [...], "lost": [...], "removed": {...}, "moved": int}`:
    an exchange that never settles is dropped -- the interface stays with the
    pair that held it before and after the interlude -- and one that settles
    later is kept, at its `settled_frame`.

    Not called by anything in the pipeline by default. `annotate_persistence`
    deliberately drops nothing, and this returns a NEW structure rather than
    editing one, so the unfiltered event table is still there afterwards: a
    filter whose removals cannot be seen cannot be judged too aggressive
    (`docs/schmitt-trigger.md` section 6).

    Requires `annotate_persistence(..., detected_frame=...)` to have run;
    without `settled_frame` there is nothing to flatten and it raises rather
    than quietly returning the input.
    """
    kept = {"gained": [], "lost": []}
    removed = {"gained": 0, "lost": 0}
    moved = 0
    for key in ("gained", "lost"):
        for entry in events[key]:
            if "settled_frame" not in entry:
                raise ValueError("flatten_transient needs annotate_persistence(detected_frame=...)")
            if entry["settled_frame"] is None:
                removed[key] += 1
                continue
            if entry["settled_frame"] != entry.get("detected_frame"):
                moved += 1
            kept[key].append(entry)
    return {**kept, "removed": removed, "moved": moved}


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


def validate_quartet_permutation(
    events: dict,
    contacts_t: Set[tuple],
    contacts_t1: Set[tuple],
) -> dict:
    """Benoit et al. 2020's four-cell permutation pruning, as a SECOND named
    alternative (PLoS Comput Biol 10.1371/journal.pcbi.1008049).

    Four cells can be paired three ways -- (ab|cd), (ac|bd), (ad|bc) -- and a
    T1 is a move from one of those partitions to a DIFFERENT one, with the
    four ring edges shared by both states. So of the six ordered pairings only
    two are T1-compatible, and a quartet whose start or end state is not a
    closed ring is an intermediate, not an event. That intermediate is what
    Benoit reports Tissue Analyzer mis-labelling: 45 false T1s against their 6
    on the same embryo data, *"intermediate chain-like topologies between
    valid start and end states"*.

    Marks each gained event `quartet: True` with its `quartet_partner`;
    annotates, drops nothing, and `detect_t1_transitions` is untouched.

    **Requires both states to be complete**: all five of the six quartet edges
    present at `t` (the ring plus the lost pair) and all five at `t+1` (the
    ring plus the gained pair). That is the difference from
    `validate_rosette_quartet` above, which requires the ring in both frames
    but says nothing about the diagonal in the frame where it should be
    absent. The two therefore agree whenever the event came out of
    `detect_t1_transitions`, which guarantees those absences -- measured on
    `embryos_GBE_GAP43` and reported in README Round 30 §G. Kept separate
    anyway, because they are two papers' criteria and a later change to one
    must not silently become a change to the other.

    **What this does NOT port**: the DTW template matching, the 3-frame
    sliding mean on interface length, and the 2 micron minimum. The first
    needs labelled training events this project does not have; the last needs
    a pixel size the repo does not record anywhere (`docs/schmitt-trigger.md`
    section 7).
    """
    lost_pairs = [entry["tracks"] for entry in events["lost"]]
    accepted = 0
    for entry in events["gained"]:
        c, d = entry["tracks"]
        entry["quartet"] = False
        for a, b in lost_pairs:
            if len({a, b, c, d}) != 4:
                continue
            ring = {(min(p, q), max(p, q)) for p, q in ((a, c), (c, b), (b, d), (d, a))}
            before = ring | {(min(a, b), max(a, b))}
            after = ring | {(min(c, d), max(c, d))}
            if before <= contacts_t and after <= contacts_t1 \
                    and (min(c, d), max(c, d)) not in contacts_t \
                    and (min(a, b), max(a, b)) not in contacts_t1:
                entry["quartet"] = True
                entry["quartet_partner"] = (a, b)
                accepted += 1
                break

    # Round 32 T1-2: the check judges GAINS only, so filtering on it hides a
    # gain's axis and leaves its paired loss drawn alone -- the maintainer's
    # "cells lose their T1 axes" was exactly the blue gain axis gone and the
    # orange loss axis still there. Mark each loss with whether the filter
    # orphaned it: it is PAIRED with at least one gain (the T1 geometry --
    # four distinct cells, the lost pair both neighbours at `t` of each cell
    # of the gained pair, the same `shared` test the rosette check uses) and
    # none of those gains was accepted. A loss with no paired gain at all is
    # not the filter's doing and is not marked. Annotation for display only:
    # the filter's own logic is unchanged.
    neighbours_t: Dict[int, Set[int]] = {}
    for p, q in contacts_t:
        neighbours_t.setdefault(p, set()).add(q)
        neighbours_t.setdefault(q, set()).add(p)
    accepted_partners = {entry["quartet_partner"] for entry in events["gained"]
                         if entry.get("quartet")}
    for loss in events["lost"]:
        a, b = loss["tracks"]
        paired = [gain for gain in events["gained"]
                  if len({a, b, *gain["tracks"]}) == 4
                  and {a, b} <= (neighbours_t.get(gain["tracks"][0], set())
                                 & neighbours_t.get(gain["tracks"][1], set()))]
        loss["quartet_orphan"] = bool(paired) and (a, b) not in accepted_partners
    return {
        "gained": len(events["gained"]), "quartet": accepted,
        "rejected": len(events["gained"]) - accepted,
        "orphaned_losses": sum(1 for loss in events["lost"] if loss["quartet_orphan"]),
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

    # --- R30 G1: Benoit's quartet-permutation check, beside the rosette one
    permutation = detect_t1_transitions(graph_t, graph_t1, identity, identity)
    perm_summary = validate_quartet_permutation(permutation, ct, ct1)
    print("\nR30 G1 quartet-permutation check on the canonical T1:", perm_summary)
    assert perm_summary["quartet"] == 1, perm_summary
    # The difference from the rosette check is the state completeness
    # requirement, so a quartet missing one ring edge at t+1 must fail here.
    broken_t1 = {1: {3}, 2: {3, 4}, 3: {1, 2, 4}, 4: {2, 3}}
    broken = detect_t1_transitions(graph_t, broken_t1, identity, identity)
    assert validate_quartet_permutation(broken, ct,
                                        contacts_in_track_space(broken_t1, identity))["quartet"] == 0
    print("  -> an incomplete end state is rejected; the event is annotated, not dropped.")
    # R32 T1-2: the loss paired with that rejected gain is the orphan the
    # display has to hide too -- and the accepted canonical T1's loss is not.
    assert broken["lost"] and broken["lost"][0]["quartet_orphan"] is True, broken["lost"]
    assert permutation["lost"][0]["quartet_orphan"] is False, permutation["lost"]
    print("  -> R32: the loss left behind by a rejected gain is marked quartet_orphan; "
          "an accepted T1's loss is not.")

    # --- R30 E2/E3: the settle frame, and flattening as an explicit step
    flicker = detect_t1_transitions(graph_t, graph_t1, identity, identity)
    # Reverses at t+2, comes back at t+3 and holds: detected at 7, settles at 10.
    after = [ct, ct1, ct1]
    annotate_persistence(flicker, after, persist_frames=1, detected_frame=7)
    entry = flicker["gained"][0]
    assert entry["detected_frame"] == 7, entry
    assert entry["settled_frame"] == 10, entry        # contacts_after[1] is frame 10
    assert entry["state"] == "transient", entry
    flat = flatten_transient(flicker)
    assert flat["moved"] == 2 and flat["removed"] == {"gained": 0, "lost": 0}, flat
    never = detect_t1_transitions(graph_t, graph_t1, identity, identity)
    annotate_persistence(never, [ct, ct, ct], persist_frames=1, detected_frame=7)
    assert never["gained"][0]["settled_frame"] is None
    assert flatten_transient(never)["removed"] == {"gained": 1, "lost": 1}
    assert len(never["gained"]) == 1, "flatten returns a new structure; the raw table stays"
    print("R30 E: an exchange detected at frame 7 that settles at 10 keeps both frames; "
          "flatten_transient moves it, drops one that never settles, and leaves the "
          "original event table intact.")

    print("\nAll topology self-tests passed.")
