# Code style: AI-code. AI-generated read-only frozen-replay input diagnostics.
"""Inspect raw tracks and hits without changing reconstruction or selections.

The independent CSC association reproduces HitTruthAssociation.h's diagnostic
policy. Agreement with stored selected rows must be established on the actual
replay; this Python implementation is not proof of the frozen binary's source.
"""
from collections import Counter, defaultdict
import hashlib
import itertools
import math
from pathlib import Path


TRACK_BANKS = ("displacedStandAloneMuons", "shiftTraversingMuons", "shiftCosmicMuons")
SELECTED_FIELDS = ("recoAlgorithm", "sourceIndex", "hitSimTrackId", "hitGenPartIdx",
                   "hitTruthLayers", "hitTruthMatchedLayers", "hitTruthAmbiguousLayers", "hitTruthPurity")


def require(value, message):
    if not value:
        raise ValueError(message)


def associate_csc_hits(measurements, crossings):
    """Vote by distinct layer: chi2<=25, unique winner, >=3 votes, purity>=.75.

    Identical measured coordinates are deduplicated. Multiple measurements or
    compatible SimTracks make a layer ambiguous. ALL crossing particle types
    are used. No nearest-angle assignment or truth-based selection is made.
    """
    layers, by_layer = defaultdict(list), defaultdict(list)
    for hit in measurements:
        if not any(hit["position_cm"] == other["position_cm"] for other in layers[hit["layer"]]):
            layers[hit["layer"]].append(hit)
    for crossing in crossings:
        by_layer[crossing["layer"]].append(crossing)
    votes, ambiguous = Counter(), 0
    for layer, hits in layers.items():
        if len(hits) != 1:
            ambiguous += 1
            continue
        hit = hits[0]
        xx, xy, yy = hit["error_cm2"]
        determinant = xx * yy - xy * xy
        if not (xx > 0. and yy > 0. and determinant > 0. and math.isfinite(determinant)):
            continue
        compatible = set()
        for crossing in by_layer[layer]:
            px, py, pz = crossing["momentum_GeV"]
            if not math.isfinite(pz) or abs(pz) < 1.e-12:
                continue
            dz = hit["position_cm"][2] - crossing["position_cm"][2]
            dx = hit["position_cm"][0] - crossing["position_cm"][0] - dz * px / pz
            dy = hit["position_cm"][1] - crossing["position_cm"][1] - dz * py / pz
            chi2 = (yy * dx * dx - 2. * xy * dx * dy + xx * dy * dy) / determinant
            if math.isfinite(chi2) and 0. <= chi2 <= 25.:
                compatible.add(crossing["track_id"])
        if len(compatible) == 1:
            votes[next(iter(compatible))] += 1
        elif len(compatible) > 1:
            ambiguous += 1
    matched = max(votes.values(), default=0)
    winners = [track for track, count in votes.items() if count == matched]
    purity = matched / len(layers) if layers else 0.
    track = winners[0] if len(winners) == 1 and matched >= 3 and purity >= .75 else -1
    return dict(track=track, total=len(layers), matched=matched, ambiguous=ambiguous,
                purity=purity, votes_by_track=dict(sorted(votes.items())))


def validate_selected_association(association, selected):
    """Close identities/counts exactly; Nano purity is a quantized Float."""
    for raw, stored in (("track", "hitSimTrackId"), ("total", "hitTruthLayers"),
                        ("matched", "hitTruthMatchedLayers"), ("ambiguous", "hitTruthAmbiguousLayers")):
        require(association[raw] == selected[stored], "Selected CSC association mismatch: " + stored)
    difference = abs(association["purity"] - selected["hitTruthPurity"])
    # At precision=10, one Float mantissa step is at most 2^-11 in [0,1).
    # Integer vote counts above remain exact; this does not relax any link.
    require(math.isfinite(difference) and difference <= 2.**-11,
            "Selected CSC association mismatch: quantized hitTruthPurity")
    return dict(identity_and_layer_counts_exact=True, purity_comparison="quantized Float, not exact equality",
                purity_absolute_difference=difference, purity_absolute_tolerance=2.**-11)


def json_safe_report(report):
    """Retain invalid numerical fields as null plus their exact path/category."""
    invalid = []
    def clean(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            invalid.append(dict(path=path, value="nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"))
            return None
        if isinstance(value, dict):
            return {key: clean(item, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, path + "/" + str(index)) for index, item in enumerate(value)]
        return value
    result = clean(report, "")
    result["nonfinite_field_values"] = invalid
    return result


def _xyz(point):
    return [float(point.x()), float(point.y()), float(point.z())]


def _product(event, typename, label):
    from DataFormats.FWLite import Handle
    handle = Handle(typename)
    event.getByLabel(label, handle)
    require(handle.isValid(), "Missing required product: " + str(label) + " (" + typename + ")")
    return handle, handle.product()


def _identity(event):
    aux = event.eventAuxiliary()
    return [int(aux.run()), int(aux.luminosityBlock()), int(aux.event())]


def _leaf_hits(hit, ROOT):
    if not hit.isValid():
        return []
    components = hit.recHits()
    if len(components):
        return list(itertools.chain.from_iterable(_leaf_hits(child, ROOT) for child in components if child))
    error, detid = hit.localPositionError(), hit.geographicalId()
    row = dict(layer=int(detid.rawId()), position_cm=_xyz(hit.localPosition()),
               error_cm2=[float(error.xx()), float(error.xy()), float(error.yy())])
    row["is_CSC_layer"] = int(detid.det()) == 2 and int(detid.subdetId()) == 2
    if row["is_CSC_layer"]:
        csc = ROOT.CSCDetId(row["layer"])
        row.update(endcap=int(csc.endcap()), station=int(csc.station()), chamber=int(csc.chamber()))
        row["is_CSC_layer"] = int(csc.layer()) > 0
    return [row]


def _nano_rows(path, ROOT):
    source = ROOT.TFile.Open(str(path), "READ")
    require(source and not source.IsZombie() and not source.TestBit(ROOT.TFile.kRecovered), "Unreadable Nano")
    try:
        tree = source.Get("Events")
        required = ["run", "luminosityBlock", "event", "nShiftMuon"] + ["ShiftMuon_" + x for x in SELECTED_FIELDS]
        require(tree and all(tree.GetBranch(name) for name in required), "Missing selected-row Nano branches")
        rows = []
        for event in tree:
            selected = []
            for index in range(int(event.nShiftMuon)):
                selected.append({name: (float if name == "hitTruthPurity" else int)(
                    getattr(event, "ShiftMuon_" + name)[index]) for name in SELECTED_FIELDS})
            rows.append(dict(event_id=[int(event.run), int(event.luminosityBlock), int(event.event)], selected=selected))
        return rows
    finally:
        source.Close()


def audit_reco_inputs(sim_path, digi_path, reco_path, nano_path):
    """Return all raw tracks/CSC SimHits and a gated selected-row truth join.

    Missing products/owned hits or any selected association mismatch fail. Valid
    empty collections remain explicitly present, never fabricated as a fallback.
    Digi links are retained products with counts only, not a pulse-overlap study.
    """
    import ROOT
    from DataFormats.FWLite import Events
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    ROOT.gInterpreter.Declare('#include "DataFormats/MuonDetId/interface/CSCDetId.h"')
    nano = _nano_rows(nano_path, ROOT)
    require(nano and len({tuple(x["event_id"]) for x in nano}) == len(nano), "Empty/repeated Nano originals")
    rows = []
    readers = [Events(str(path)) for path in (sim_path, digi_path, reco_path)]
    for index, native in enumerate(itertools.zip_longest(*readers)):
        require(index < len(nano) and all(x is not None for x in native), "Detector tier event counts differ")
        sim, digi, reco = native
        event_id = nano[index]["event_id"]
        require(all(_identity(x) == event_id for x in native), "Detector/Nano event identities or order differ")
        handles = []
        def get(event, typename, label):
            handle, product = _product(event, typename, label)
            handles.append(handle)
            return product
        tracks = get(sim, "std::vector<SimTrack>", ("g4SimHits", "", "SHIFTSIM"))
        vertices = get(sim, "std::vector<SimVertex>", ("g4SimHits", "", "SHIFTSIM"))
        hits = get(sim, "std::vector<PSimHit>", ("g4SimHits", "MuonCSCHits", "SHIFTSIM"))
        crossings = []
        for hit in hits:
            crossing = dict(layer=int(hit.detUnitId()), track_id=int(hit.trackId()),
                encoded_event_id=int(hit.eventId().rawId()), particle_type=int(hit.particleType()),
                position_cm=_xyz(hit.entryPoint()), exit_position_cm=_xyz(hit.exitPoint()),
                momentum_GeV=_xyz(hit.momentumAtEntry()), time_of_flight_ns=float(hit.timeOfFlight()))
            detid = ROOT.CSCDetId(crossing["layer"])
            crossing.update(endcap=int(detid.endcap()), station=int(detid.station()), chamber=int(detid.chamber()))
            crossings.append(crossing)
        products = dict(SimTracks=dict(present=True, objects=int(tracks.size())),
                        SimVertices=dict(present=True, objects=int(vertices.size())),
                        CSC_SimHits=dict(present=True, objects=int(hits.size())))
        for instance in ("MuonCSCWireDigiSimLinks", "MuonCSCStripDigiSimLinks"):
            links = get(digi, "edm::DetSetVector<StripDigiSimLink>", ("simMuonCSCDigis", instance))
            products[instance] = dict(present=True, detector_sets=int(links.size()),
                                     links=sum(int(detset.size()) for detset in links))
        for typename, label in (("CSCRecHit2DCollection", "csc2DRecHits"), ("CSCSegmentCollection", "cscSegments")):
            product = get(reco, typename, label)
            products[label] = dict(present=True, objects=int(product.size()))
        raw_tracks = []
        for algorithm, label in enumerate(TRACK_BANKS):
            bank = get(reco, "std::vector<reco::Track>", (label, "", "RECO"))
            products[label] = dict(present=True, tracks=int(bank.size()))
            for source_index, track in enumerate(bank):
                extra = track.extra()
                require(extra.isNonnull() and extra.isAvailable(), "Missing TrackExtra: " + label)
                require(extra.get().recHitsOk(), "Missing owned track hits: " + label)
                leaves = []
                for number in range(int(track.recHitsSize())):
                    ref = track.recHit(number)
                    require(ref.isNonnull() and ref.isAvailable(), "Missing owned RecHit: " + label)
                    leaves.extend(_leaf_hits(ref.get(), ROOT))
                measured = [x for x in leaves if x["is_CSC_layer"]]
                stations = {str(side): sorted({x["station"] for x in measured if x["endcap"] == side}) for side in (1, 2)}
                raw_tracks.append(dict(recoAlgorithm=algorithm, sourceIndex=source_index, module=label,
                    charge=int(track.charge()), p_GeV=float(track.p()), momentum_GeV=[float(track.px()), float(track.py()), float(track.pz())],
                    valid_hits=int(track.numberOfValidHits()), owned_hits=int(track.recHitsSize()),
                    CSC_stations_by_endcap=stations, valid_leaf_hits=leaves,
                    independent_CSC_association=associate_csc_hits(measured, crossings)))
        lookup = {(x["recoAlgorithm"], x["sourceIndex"]): x for x in raw_tracks}
        selected, seen = [], set()
        for selected_index, stored in enumerate(nano[index]["selected"]):
            key = (stored["recoAlgorithm"], stored["sourceIndex"])
            require(key in lookup and key not in seen, "Invalid/repeated selected raw-track identity")
            seen.add(key)
            proof = validate_selected_association(lookup[key]["independent_CSC_association"], stored)
            selected.append(dict(selected_index=selected_index, stored=stored, closure=proof))
        crossing_events = defaultdict(set)
        for crossing in crossings:
            crossing_events[crossing["track_id"]].add(crossing["encoded_event_id"])
        rows.append(dict(event_id=event_id, products=products, raw_tracks=raw_tracks, selected=selected,
            encoded_CSC_event_ids=sorted({x["encoded_event_id"] for x in crossings}),
            track_id_event_id_collisions=[dict(track_id=track, encoded_event_ids=sorted(ids))
                                          for track, ids in sorted(crossing_events.items()) if len(ids) > 1],
            CSC_SimHits=crossings,
            SimTracks=[dict(track_id=int(t.trackId()), encoded_event_id=int(t.eventId().rawId()), pdg=int(t.type()),
                GEN_barcode=int(t.genpartIndex()), vertex_index=int(t.vertIndex()), momentum_GeV=_xyz(t.momentum())) for t in tracks],
            SimVertices=[dict(index=i, parent_track_id=int(v.parentIndex()), process_type=int(v.processType()),
                position_cm=_xyz(v.position()), time_seconds=float(v.position().t())) for i, v in enumerate(vertices)]))
    require(len(rows) == len(nano), "Detector tier has fewer events than Nano")
    return json_safe_report(dict(schema="shift-raw-reco-input-audit-v1", complete=True, events=rows, event_count=len(rows),
        inputs={name: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                for name, path in zip(("SIM", "GENRAW", "AOD", "Nano"), (sim_path, digi_path, reco_path, nano_path))},
        association_policy="Independent HitTruthAssociation.h policy; all CSC PSimHit particle types; selected link/count closure required.",
        association_event_scope="Votes key trackId only, matching the existing header; this is not an event-aware association.",
        multiple_encoded_CSC_events=any(len(row["encoded_CSC_event_ids"]) > 1 for row in rows),
        CSC_endcap_convention={"1": "plus-z", "2": "minus-z"},
        digi_scope="Mandatory wire/strip DigiSimLinks checked and counted; pulses/channel-overlap not analyzed.",
        limitations=["No proof of the frozen producer source or duplicate-component membership.",
                     "A missing second selected row does not prove that two physical muons merged.",
                     "Stored hitGenPartIdx is reported; GEN index assignment is not recomputed by this diagnostic.",
                     "Hit association uses local measurement compatibility, not nearest GEN angles."]))
