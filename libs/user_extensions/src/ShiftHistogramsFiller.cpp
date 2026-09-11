#include "ShiftHistogramsFiller.hpp"

#include "ConfigManager.hpp"

#include <map>
#include <cmath>
#include <set>

using namespace std;

namespace {

vector<string> const dimuonCategories = {
    "", "Good", "Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other"};

string GetDimuonTopologyCategory(int topologyMin, int topologyMax) {
  static map<pair<int, int>, string> const namedCategories = {
      {{0, 2}, "Near-Both"},
      {{0, 3}, "Near-Far"},
      {{2, 2}, "Both-Both"},
      {{2, 3}, "Both-Far"},
  };
  auto const category = namedCategories.find({topologyMin, topologyMax});
  return category == namedCategories.end() ? "Other" : category->second;
}

}  // namespace

ShiftHistogramsFiller::ShiftHistogramsFiller(shared_ptr<HistogramsHandler> histogramsHandler_) : histogramsHandler(histogramsHandler_) {
  auto& config = ConfigManager::GetInstance();
  eventProcessor = make_unique<EventProcessor>();
  config.GetValue("enableTruthDiagnostics", enableTruthDiagnostics);
}

ShiftHistogramsFiller::~ShiftHistogramsFiller() {}

void ShiftHistogramsFiller::Fill(const shared_ptr<Event> event) {
  FillRecoLevel(event);
  FillDetectorDiagnostics(event);
  if (enableTruthDiagnostics) {
    FillGenLevel(event);
    FillRecoVsGen2D(event);
    FillResolutionPlots(event);
    FillEfficiencies(event);
  }
}

int ShiftHistogramsFiller::HitTruthIndex(const shared_ptr<PhysicsObject>& muon) const {
  if (!muon->HasBranch("hitGenPartIdx"))
    throw runtime_error("SHIFT truth diagnostics require hitGenPartIdx; angular matching is not a fallback. "
                        "For collision data set enableTruthDiagnostics=False.");
  return muon->GetAs<int>("hitGenPartIdx");
}

const ShiftHistogramsFiller::GenJPsiCandidate* ShiftHistogramsFiller::MatchDimuon(
    const shared_ptr<PhysicsObject>& dimuon, const shared_ptr<PhysicsObjects>& muons,
    const vector<GenJPsiCandidate>& candidates) const {
  int const i = dimuon->GetAs<int>("muonIdx1"), j = dimuon->GetAs<int>("muonIdx2");
  if (i < 0 || j < 0 || i == j || size_t(i) >= muons->size() || size_t(j) >= muons->size()) return nullptr;
  int const a = HitTruthIndex(muons->at(i)), b = HitTruthIndex(muons->at(j));
  if (a < 0 || b < 0 || a == b) return nullptr;
  for (auto const& candidate : candidates)
    if ((a == candidate.muonMinusIdx && b == candidate.muonPlusIdx) ||
        (b == candidate.muonMinusIdx && a == candidate.muonPlusIdx)) return &candidate;
  return nullptr;
}

void ShiftHistogramsFiller::FillDetectorDiagnostics(const shared_ptr<Event> event) {
  auto muons = event->GetCollection("ShiftMuon");
  for (auto const& muon : *muons) {
    histogramsHandler->FillUnweighted("TargetDiagnostics_constrainedStatus", muon->GetAs<int>("constrainedStatus"));
    if (enableTruthDiagnostics) {
      histogramsHandler->FillUnweighted("TruthDiagnostics_hitMatched", HitTruthIndex(muon) >= 0);
      histogramsHandler->FillUnweighted("TruthDiagnostics_legacyMatched", muon->GetAs<int>("genPartIdx") >= 0);
    }
    for (string const variable : {"constrainedEtaErr", "constrainedPhiErr", "targetPredictedEtaErr",
          "targetPredictedPhiErr", "targetResidualX", "targetResidualY", "targetPullX", "targetPullY"})
      if (muon->HasBranch(variable) && muon->GetAs<int>("constrainedValid"))
        histogramsHandler->Fill("TargetDiagnostics_" + variable, muon->GetAs<float>(variable));
    int const compatibleDT = muon->GetAs<int>("nCompatibleDTSegments");
    int const addedDT = muon->GetAs<int>("nAddedDTRefitHits");
    int const compatibleTracker = muon->GetAs<int>("nCompatiblePixelHits") +
                                  muon->GetAs<int>("nCompatibleStripHits");
    int const addedTracker = muon->GetAs<int>("nAddedTrackerRefitHits");
    if (compatibleDT > 0)
      histogramsHandler->FillUnweighted("DetectorDiagnostics_dtAttachmentFraction",
                                        static_cast<double>(addedDT) / compatibleDT);
    int const truthMatchedDT = enableTruthDiagnostics ? muon->GetAs<int>("nAddedDTTruthChamberMatches") : -1;
    if (addedDT > 0 && truthMatchedDT >= 0)
      histogramsHandler->FillUnweighted("DetectorDiagnostics_dtTruthChamberPurity",
                                        static_cast<double>(truthMatchedDT) / addedDT);
    if (compatibleTracker > 0)
      histogramsHandler->FillUnweighted("DetectorDiagnostics_trackerAttachmentFraction",
                                        static_cast<double>(addedTracker) / compatibleTracker);
    histogramsHandler->FillUnweighted("DetectorDiagnostics_timingMeasurements",
                                      muon->GetAs<int>("nTimingMeasurements"));
    histogramsHandler->FillUnweighted("DetectorDiagnostics_timingDeltaChi2",
                                      muon->GetAs<float>("timingDeltaChi2"));
    histogramsHandler->FillUnweighted("DetectorDiagnostics_combinedTimingDeltaChi2",
                                      muon->GetAs<float>("combinedTimingDeltaChi2"));
    int const crossedHBHE = muon->GetAs<int>("nCrossedHBHERecHits");
    int const crossedHO = muon->GetAs<int>("nCrossedHORecHits");
    if (crossedHBHE > 0)
      histogramsHandler->FillUnweighted("DetectorDiagnostics_hbheValidTimeFraction",
                                        static_cast<double>(muon->GetAs<int>("nValidCrossedHBHETimes")) /
                                            crossedHBHE);
    if (crossedHO > 0)
      histogramsHandler->FillUnweighted("DetectorDiagnostics_hoValidTimeFraction",
                                        static_cast<double>(muon->GetAs<int>("nValidCrossedHOTimes")) /
                                            crossedHO);
  }
}

void ShiftHistogramsFiller::FillEfficiencies(const shared_ptr<Event> event) {
  auto genParticles = event->GetCollection("GenPart");
  auto recoMuons = event->GetCollection("ShiftMuon");
  auto recoDimuons = event->GetCollection("ShiftDimuonVertex");
  auto genCandidates = GetGenJPsiCandidates(genParticles);

  map<int, vector<size_t>> recoMuonIndicesByGenIndex;
  for (size_t recoIndex = 0; recoIndex < recoMuons->size(); ++recoIndex) {
    int const genIndex = HitTruthIndex(recoMuons->at(recoIndex));
    if (genIndex >= 0 && static_cast<size_t>(genIndex) < genParticles->size())
      recoMuonIndicesByGenIndex[genIndex].push_back(recoIndex);
  }

  vector<pair<string, int>> const muonCategories = {
      {"NearEndcapOnly", 0},
      {"NearEndcapAndBarrel", 1},
      {"BothEndcaps", 2},
      {"FarEndcapOnly", 3},
      {"Unclassified", 4},
  };
  auto fillMuon = [this](string const& prefix, shared_ptr<NanoGenParticle> const& muon, bool pass) {
    map<string, double> const values = {
        {"pt", muon->GetAs<float>("pt")},   {"pz", muon->GetAs<float>("pz")},
        {"eta", muon->GetAs<float>("eta")}, {"phi", muon->GetAs<float>("phi")},
        {"vz", muon->GetAs<float>("vz")},
    };
    for (auto const& [variable, value] : values) {
      histogramsHandler->FillUnweighted(prefix + "_" + variable + "_total", value);
      if (pass)
        histogramsHandler->FillUnweighted(prefix + "_" + variable + "_pass", value);
    }
  };

  for (auto const& candidate : genCandidates) {
    for (auto const& [genIndex, genMuon] : vector<pair<int, shared_ptr<NanoGenParticle>>>{
             {candidate.muonMinusIdx, candidate.muonMinus}, {candidate.muonPlusIdx, candidate.muonPlus}}) {
      auto const matchIt = recoMuonIndicesByGenIndex.find(genIndex);
      bool const matched = matchIt != recoMuonIndicesByGenIndex.end();
      map<int, bool> categoryMatched;
      if (matched) {
        for (size_t const recoIndex : matchIt->second) {
          auto const recoMuon = recoMuons->at(recoIndex);
          int const topology = recoMuon->GetAs<int>("topology");
          categoryMatched[topology] = true;
        }
      }
      fillMuon("ShiftMuonEfficiency", genMuon, matched);
      for (auto const& [category, topology] : muonCategories)
        fillMuon("ShiftMuon" + category + "Efficiency", genMuon, categoryMatched[topology]);
    }
  }

  map<string, set<PhysicsObject const*>> dimuonCategoryMembers;
  for (auto const& category : dimuonCategories) {
    auto const collection = event->GetCollection("ShiftDimuonVertex" + category);
    for (auto const& dimuon : *collection)
      dimuonCategoryMembers[category].insert(dimuon.get());
  }
  auto fillDimuon = [this](string const& prefix, GenJPsiCandidate const& candidate, bool pass) {
    map<string, double> const values = {
        {"pt", candidate.momentum.Pt()},   {"pz", candidate.momentum.Pz()},
        {"eta", candidate.momentum.Eta()}, {"phi", candidate.momentum.Phi()},
        {"vz", candidate.vertex.Z()},
    };
    for (auto const& [variable, value] : values) {
      histogramsHandler->FillUnweighted(prefix + "_" + variable + "_total", value);
      if (pass)
        histogramsHandler->FillUnweighted(prefix + "_" + variable + "_pass", value);
    }
  };

  for (auto const& candidate : genCandidates) {
    bool matched = false;
    map<string, bool> categoryMatched;
    for (auto const& recoDimuon : *recoDimuons) {
      int const firstRecoIndex = recoDimuon->GetAs<int>("muonIdx1");
      int const secondRecoIndex = recoDimuon->GetAs<int>("muonIdx2");
      if (firstRecoIndex < 0 || secondRecoIndex < 0 ||
          static_cast<size_t>(firstRecoIndex) >= recoMuons->size() ||
          static_cast<size_t>(secondRecoIndex) >= recoMuons->size())
        continue;
      set<int> const recoGenIndices = {
          HitTruthIndex(recoMuons->at(firstRecoIndex)),
          HitTruthIndex(recoMuons->at(secondRecoIndex)),
      };
      set<int> const truthGenIndices = {candidate.muonMinusIdx, candidate.muonPlusIdx};
      if (recoGenIndices.size() != 2 || recoGenIndices != truthGenIndices)
        continue;

      matched = true;
      for (auto const& category : dimuonCategories)
        categoryMatched[category] = categoryMatched[category] ||
                                    dimuonCategoryMembers[category].count(recoDimuon.get());
    }
    for (auto const& category : dimuonCategories)
      fillDimuon("ShiftDimuonVertex" + category + "Efficiency", candidate,
                 category.empty() ? matched : categoryMatched[category]);
  }
}

void ShiftHistogramsFiller::FillGenLevel(const shared_ptr<Event> event) {
  auto genParticles = event->GetCollection("GenPart");

  for (auto const& candidate : GetGenJPsiCandidates(genParticles)) {
    auto const& genDimuon = candidate.momentum;
    auto const& genDimuonVertex = candidate.vertex;

    histogramsHandler->Fill("GenDimuon_pt", genDimuon.Pt());
    histogramsHandler->Fill("GenDimuon_pz", genDimuon.Pz());
    histogramsHandler->Fill("GenDimuon_eta", genDimuon.Eta());
    histogramsHandler->Fill("GenDimuon_phi", genDimuon.Phi());
    histogramsHandler->Fill("GenDimuon_mass", genDimuon.M());
    histogramsHandler->Fill("GenDimuon_vx", genDimuonVertex.X());
    histogramsHandler->Fill("GenDimuon_vy", genDimuonVertex.Y());
    histogramsHandler->Fill("GenDimuon_vz", genDimuonVertex.Z());
  }
}

void ShiftHistogramsFiller::FillRecoLevel(const shared_ptr<Event> event) {
  auto dimuons = event->GetCollection("ShiftDimuonVertex");

  map<string, int> const categoryIndices = {
      {"Near-Both", 0}, {"Near-Far", 1}, {"Both-Both", 2}, {"Both-Far", 3}, {"Other", 4},
  };
  map<int, string> labels;
  for (auto const& [category, index] : categoryIndices)
    labels[index] = category;

  auto const recoMuons = event->GetCollection("ShiftMuon");
  vector<GenJPsiCandidate> truthCandidates;
  if (enableTruthDiagnostics) truthCandidates = GetGenJPsiCandidates(event->GetCollection("GenPart"));
  for (auto const& dimuon : *dimuons) {
    if (enableTruthDiagnostics)
      histogramsHandler->FillUnweighted("TruthDiagnostics_dimuonHitMatched", MatchDimuon(dimuon, recoMuons, truthCandidates) != nullptr);
    string const category = GetDimuonTopologyCategory(
        dimuon->GetAs<int>("topologyMin"), dimuon->GetAs<int>("topologyMax"));
    histogramsHandler->Fill("ShiftDimuonVertex_topologyCategory", categoryIndices.at(category));
  }

  histogramsHandler->SetHistogramLabels("ShiftDimuonVertex_topologyCategory", labels);
}

void ShiftHistogramsFiller::FillRecoVsGen2D(const shared_ptr<Event> event) {
  // single muon
  auto genParticles = event->GetCollection("GenPart");
  auto recoShiftMuons = event->GetCollection("ShiftMuon");

  for (size_t i = 0; i < recoShiftMuons->size(); i++) {
    auto recoMuon = recoShiftMuons->at(i);
    int genPartIdx = HitTruthIndex(recoMuon);
    if (genPartIdx < 0 || size_t(genPartIdx) >= genParticles->size()) continue;
    auto genMuon = genParticles->at(genPartIdx);

    histogramsHandler->Fill("RecoVsGenMuon_pt", recoMuon->GetAs<float>("pt"), genMuon->GetAs<float>("pt"));
    histogramsHandler->Fill("RecoVsGenMuon_pz", recoMuon->GetAs<float>("pz"), genMuon->GetAs<float>("pz"));
    histogramsHandler->Fill("RecoVsGenMuon_eta", recoMuon->GetAs<float>("eta"), genMuon->GetAs<float>("eta"));
    histogramsHandler->Fill("RecoVsGenMuon_phi", recoMuon->GetAs<float>("phi"), genMuon->GetAs<float>("phi"));
    histogramsHandler->Fill("RecoVsGenMuon_vx", recoMuon->GetAs<float>("vx"), genMuon->GetAs<float>("vx"));
    histogramsHandler->Fill("RecoVsGenMuon_vy", recoMuon->GetAs<float>("vy"), genMuon->GetAs<float>("vy"));
    histogramsHandler->Fill("RecoVsGenMuon_vz", recoMuon->GetAs<float>("vz"), genMuon->GetAs<float>("vz"));
  }

  // dimuon
  auto const genCandidates = GetGenJPsiCandidates(genParticles);

  auto recoShiftDimuons = event->GetCollection("ShiftDimuonVertex");
  for (size_t i = 0; i < recoShiftDimuons->size(); i++) {
    auto recoDimuon = recoShiftDimuons->at(i);
    auto const match = MatchDimuon(recoDimuon, recoShiftMuons, genCandidates);
    if (!match) continue;
    auto const& genJPsiVec = match->momentum;
    auto const& genJPsiVertex = match->vertex;

    histogramsHandler->Fill("RecoVsGenDimuon_pt", recoDimuon->GetAs<float>("pt"), genJPsiVec.Pt());
    histogramsHandler->Fill("RecoVsGenDimuon_pz", recoDimuon->GetAs<float>("pz"), genJPsiVec.Pz());
    histogramsHandler->Fill("RecoVsGenDimuon_eta", recoDimuon->GetAs<float>("eta"), genJPsiVec.Eta());
    histogramsHandler->Fill("RecoVsGenDimuon_phi", recoDimuon->GetAs<float>("phi"), genJPsiVec.Phi());
    histogramsHandler->Fill("RecoVsGenDimuon_minv", recoDimuon->GetAs<float>("mass"), genJPsiVec.M());
    histogramsHandler->Fill("RecoVsGenDimuon_vx", recoDimuon->GetAs<float>("vx"), genJPsiVertex.X());
    histogramsHandler->Fill("RecoVsGenDimuon_vy", recoDimuon->GetAs<float>("vy"), genJPsiVertex.Y());
    histogramsHandler->Fill("RecoVsGenDimuon_vz", recoDimuon->GetAs<float>("vz"), genJPsiVertex.Z());
  }
}

void ShiftHistogramsFiller::FillResolutionPlots(const shared_ptr<Event> event) {
  // plot pt, pz, eta, phi, minv resolutions for reco vs gen muons and dimuons
  // Implementation for resolution plots
  auto genMuons = event->GetCollection("GenMuon");
  auto genParticles = event->GetCollection("GenPart");

  vector<string> shiftMuonTypes = {
      "NearEndcapOnly", "NearEndcapAndBarrel", "BothEndcaps", "FarEndcapOnly", "Unclassified"};
  map<string, shared_ptr<PhysicsObjects>> recoShiftMuons;
  for (const auto& type : shiftMuonTypes) {
    recoShiftMuons[type] = event->GetCollection("ShiftMuon" + type);
  }

  // Fill single muon resolution plots
  for (const auto& [name, recoCollection] : recoShiftMuons) {
    for (size_t i = 0; i < recoCollection->size(); i++) {
      auto recoMuon = recoCollection->at(i);
      int genPartIdx = HitTruthIndex(recoMuon);
      if (genPartIdx < 0 || genPartIdx >= genParticles->size()) continue;
      auto genMuon = asNanoGenParticle(genParticles->at(genPartIdx));
      double const de = recoMuon->GetAs<float>("eta") - genMuon->GetAs<float>("eta");
      double const dp = remainder(recoMuon->GetAs<float>("phi") - genMuon->GetAs<float>("phi"), 2. * acos(-1.));
      double const re = -recoMuon->GetAs<float>("eta") - genMuon->GetAs<float>("eta");
      double const rp = remainder(recoMuon->GetAs<float>("phi") + acos(-1.) - genMuon->GetAs<float>("phi"), 2. * acos(-1.));
      bool const reverse = hypot(re, rp) < hypot(de, dp);
      string const diagnostics = "MuonResolution" + name + "_";
      histogramsHandler->Fill(diagnostics + "directionReversed", reverse);
      histogramsHandler->Fill(diagnostics + "axisDeltaEta", reverse ? re : de);
      histogramsHandler->Fill(diagnostics + "axisDeltaPhi", reverse ? rp : dp);
      // This is a diagnostic axis comparison only: NEVER flip or cut the reco state.
      if (recoMuon->GetAs<int>("constrainedValid") && recoMuon->HasBranch("constrainedEtaErr")) {
        for (string const angle : {"Eta", "Phi"}) {
          double const error = recoMuon->GetAs<float>("constrained" + angle + "Err");
          double residual = recoMuon->GetAs<float>("constrained" + angle) -
                            genMuon->GetAs<float>(angle == "Eta" ? "eta" : "phi");
          if (angle == "Phi") residual = remainder(residual, 2. * acos(-1.));
          if (error > 0.) histogramsHandler->Fill(diagnostics + "constrained" + angle + "Pull", residual / error);
        }
      }

      histogramsHandler->Fill("MuonResolution" + name + "_deltaEta",
                              recoMuon->GetAs<float>("eta") - genMuon->GetAs<float>("eta"));
      histogramsHandler->Fill("MuonResolution" + name + "_deltaPhi",
                              std::remainder(recoMuon->GetAs<float>("phi") - genMuon->GetAs<float>("phi"), 2. * std::acos(-1.)));

      // CMS-DP-2015-015 uses the signed inverse-pT (curvature) residual.
      // PDG IDs +13/-13 denote mu-/mu+, hence the opposite sign for charge.
      double const genPt = genMuon->GetAs<float>("pt");
      double const recoPt = recoMuon->GetAs<float>("pt");
      int const genCharge = genMuon->GetPdgId() > 0 ? -1 : 1;
      int const recoCharge = recoMuon->GetAs<int>("charge");
      histogramsHandler->Fill(diagnostics + "chargeMisidentified", recoCharge != genCharge);
      if (genPt > 0. && recoPt > 0.) {
        double const genQOverPt = genCharge / genPt;
        double const recoQOverPt = recoCharge / recoPt;
        histogramsHandler->Fill("MuonResolution" + name + "_qOverPt",
                                (recoQOverPt - genQOverPt) / genQOverPt);
      }

      histogramsHandler->Fill("MuonResolution" + name + "_pt", (recoMuon->GetAs<float>("pt") - genMuon->GetAs<float>("pt")) / genMuon->GetAs<float>("pt"));
      histogramsHandler->Fill("MuonResolution" + name + "_pz", (recoMuon->GetAs<float>("pz") - genMuon->GetAs<float>("pz")) / genMuon->GetAs<float>("pz"));
      histogramsHandler->Fill("MuonResolution" + name + "_eta", (recoMuon->GetAs<float>("eta") - genMuon->GetAs<float>("eta")) / genMuon->GetAs<float>("eta"));
      histogramsHandler->Fill("MuonResolution" + name + "_phi", (recoMuon->GetAs<float>("phi") - genMuon->GetAs<float>("phi")) / genMuon->GetAs<float>("phi"));
      histogramsHandler->Fill("MuonResolution" + name + "_vx", (recoMuon->GetAs<float>("vx") - genMuon->GetAs<float>("vx")) / genMuon->GetAs<float>("vx"));
      histogramsHandler->Fill("MuonResolution" + name + "_vy", (recoMuon->GetAs<float>("vy") - genMuon->GetAs<float>("vy")) / genMuon->GetAs<float>("vy"));
      histogramsHandler->Fill("MuonResolution" + name + "_vz", (recoMuon->GetAs<float>("vz") - genMuon->GetAs<float>("vz")) / genMuon->GetAs<float>("vz"));

      // Invalid constrained fits are stored as zeros in NanoAOD. Filling them would manufacture a spike at residual -1 and bias every constrained
      // scale plot, so require the explicit validity bit.
      if (recoMuon->GetAs<int>("constrainedValid")) {
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedDeltaEta",
                                recoMuon->GetAs<float>("constrainedEta") - genMuon->GetAs<float>("eta"));
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedDeltaPhi",
                                std::remainder(recoMuon->GetAs<float>("constrainedPhi") - genMuon->GetAs<float>("phi"), 2. * std::acos(-1.)));
        double const constrainedPt = recoMuon->GetAs<float>("constrainedPt");
        if (genPt > 0. && constrainedPt > 0.) {
          double const genQOverPt = genCharge / genPt;
          double const constrainedQOverPt = recoCharge / constrainedPt;
          histogramsHandler->Fill("MuonResolution" + name + "_constrainedQOverPt",
                                  (constrainedQOverPt - genQOverPt) / genQOverPt);
        }
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedPt", (recoMuon->GetAs<float>("constrainedPt") - genMuon->GetAs<float>("pt")) / genMuon->GetAs<float>("pt"));
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedPz", (recoMuon->GetAs<float>("constrainedPz") - genMuon->GetAs<float>("pz")) / genMuon->GetAs<float>("pz"));
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedEta", (recoMuon->GetAs<float>("constrainedEta") - genMuon->GetAs<float>("eta")) / genMuon->GetAs<float>("eta"));
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedPhi", (recoMuon->GetAs<float>("constrainedPhi") - genMuon->GetAs<float>("phi")) / genMuon->GetAs<float>("phi"));
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedVx", (recoMuon->GetAs<float>("constrainedVx") - genMuon->GetAs<float>("vx")) / genMuon->GetAs<float>("vx"));
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedVy", (recoMuon->GetAs<float>("constrainedVy") - genMuon->GetAs<float>("vy")) / genMuon->GetAs<float>("vy"));
        histogramsHandler->Fill("MuonResolution" + name + "_constrainedVz", (recoMuon->GetAs<float>("constrainedVz") - genMuon->GetAs<float>("vz")) / genMuon->GetAs<float>("vz"));
      }
    }
  }

  // This aggregate is intentionally limited to the q/pT comparison. Filling
  // only its qOverPt histogram keeps all other resolution products unchanged.
  auto const singleEndcapMuons = event->GetCollection("ShiftMuonSingleEndcap");
  for (auto const& recoMuon : *singleEndcapMuons) {
    int const genPartIdx = HitTruthIndex(recoMuon);
    if (genPartIdx < 0 || static_cast<size_t>(genPartIdx) >= genParticles->size())
      continue;
    auto const genMuon = asNanoGenParticle(genParticles->at(genPartIdx));
    double const genPt = genMuon->GetAs<float>("pt");
    double const recoPt = recoMuon->GetAs<float>("pt");
    if (genPt <= 0. || recoPt <= 0.)
      continue;
    int const genCharge = genMuon->GetPdgId() > 0 ? -1 : 1;
    int const recoCharge = recoMuon->GetAs<int>("charge");
    double const genQOverPt = genCharge / genPt;
    double const recoQOverPt = recoCharge / recoPt;
    histogramsHandler->Fill("MuonResolutionSingleEndcap_qOverPt",
                            (recoQOverPt - genQOverPt) / genQOverPt);
  }

  // Pair identity comes from recorded hits, independently of fit residuals.
  auto const genCandidates = GetGenJPsiCandidates(genParticles);
  auto const recoMuons = event->GetCollection("ShiftMuon");

  for (auto const& category : dimuonCategories) {
    auto const recoShiftDimuons = event->GetCollection("ShiftDimuonVertex" + category);
    for (auto const& recoDimuon : *recoShiftDimuons) {
      auto const match = MatchDimuon(recoDimuon, recoMuons, genCandidates);
      if (!match) continue;
      auto const& genJPsiVec = match->momentum;
      auto const& genJPsiVertex = match->vertex;
      string const histogramPrefix = "DimuonResolution" + category + "_";
      histogramsHandler->Fill(histogramPrefix + "deltaEta", recoDimuon->GetAs<float>("eta") - genJPsiVec.Eta());
      histogramsHandler->Fill(histogramPrefix + "deltaPhi",
                              std::remainder(recoDimuon->GetAs<float>("phi") - genJPsiVec.Phi(), 2. * std::acos(-1.)));
      histogramsHandler->Fill(histogramPrefix + "pt", (recoDimuon->GetAs<float>("pt") - genJPsiVec.Pt()) / genJPsiVec.Pt());
      histogramsHandler->Fill(histogramPrefix + "pz", (recoDimuon->GetAs<float>("pz") - genJPsiVec.Pz()) / genJPsiVec.Pz());
      histogramsHandler->Fill(histogramPrefix + "eta", (recoDimuon->GetAs<float>("eta") - genJPsiVec.Eta()) / genJPsiVec.Eta());
      histogramsHandler->Fill(histogramPrefix + "phi", (recoDimuon->GetAs<float>("phi") - genJPsiVec.Phi()) / genJPsiVec.Phi());
      histogramsHandler->Fill(histogramPrefix + "minv", (recoDimuon->GetAs<float>("mass") - genJPsiVec.M()) / genJPsiVec.M());
      histogramsHandler->Fill(histogramPrefix + "vx", (recoDimuon->GetAs<float>("vx") - genJPsiVertex.X()) / genJPsiVertex.X());
      histogramsHandler->Fill(histogramPrefix + "vy", (recoDimuon->GetAs<float>("vy") - genJPsiVertex.Y()) / genJPsiVertex.Y());
      histogramsHandler->Fill(histogramPrefix + "vz", (recoDimuon->GetAs<float>("vz") - genJPsiVertex.Z()) / genJPsiVertex.Z());

      if (!recoDimuon->GetAs<int>("constrainedValid")) continue;

      histogramsHandler->Fill(histogramPrefix + "constrainedDeltaEta", recoDimuon->GetAs<float>("constrainedEta") - genJPsiVec.Eta());
      histogramsHandler->Fill(histogramPrefix + "constrainedDeltaPhi",
                              std::remainder(recoDimuon->GetAs<float>("constrainedPhi") - genJPsiVec.Phi(), 2. * std::acos(-1.)));

      histogramsHandler->Fill(histogramPrefix + "constrainedPt", (recoDimuon->GetAs<float>("constrainedPt") - genJPsiVec.Pt()) / genJPsiVec.Pt());
      histogramsHandler->Fill(histogramPrefix + "constrainedPz", (recoDimuon->GetAs<float>("constrainedPz") - genJPsiVec.Pz()) / genJPsiVec.Pz());
      histogramsHandler->Fill(histogramPrefix + "constrainedEta", (recoDimuon->GetAs<float>("constrainedEta") - genJPsiVec.Eta()) / genJPsiVec.Eta());
      histogramsHandler->Fill(histogramPrefix + "constrainedPhi", (recoDimuon->GetAs<float>("constrainedPhi") - genJPsiVec.Phi()) / genJPsiVec.Phi());
      histogramsHandler->Fill(histogramPrefix + "constrainedMinv", (recoDimuon->GetAs<float>("constrainedMass") - genJPsiVec.M()) / genJPsiVec.M());
      histogramsHandler->Fill(histogramPrefix + "constrainedVx", (recoDimuon->GetAs<float>("constrainedVx") - genJPsiVertex.X()) / genJPsiVertex.X());
      histogramsHandler->Fill(histogramPrefix + "constrainedVy", (recoDimuon->GetAs<float>("constrainedVy") - genJPsiVertex.Y()) / genJPsiVertex.Y());
      histogramsHandler->Fill(histogramPrefix + "constrainedVz", (recoDimuon->GetAs<float>("constrainedVz") - genJPsiVertex.Z()) / genJPsiVertex.Z());
    }
  }
}

vector<ShiftHistogramsFiller::GenJPsiCandidate> ShiftHistogramsFiller::GetGenJPsiCandidates(
    const shared_ptr<PhysicsObjects> genParticles) {
  struct DaughterIndices {
    vector<int> minus;
    vector<int> plus;
  };
  map<int, DaughterIndices> daughtersByMother;
  for (size_t index = 0; index < genParticles->size(); ++index) {
    auto const particle = asNanoGenParticle(genParticles->at(index));
    if (abs(particle->GetPdgId()) != 13 || particle->GetAs<int>("status") != 1)
      continue;
    int motherIndex = particle->GetMotherIndex();
    // A final-state muon can follow one or more same-PDG copies after photon
    // radiation. Follow ancestry, never angular proximity, to its J/psi.
    set<int> visited{static_cast<int>(index)};
    while (motherIndex >= 0 && static_cast<size_t>(motherIndex) < genParticles->size() &&
           genParticles->at(motherIndex)->GetAs<int>("pdgId") == particle->GetPdgId()) {
      if (!visited.insert(motherIndex).second) { motherIndex = -1; break; }
      motherIndex = asNanoGenParticle(genParticles->at(motherIndex))->GetMotherIndex();
    }
    if (motherIndex < 0 || static_cast<size_t>(motherIndex) >= genParticles->size())
      continue;
    if (abs(genParticles->at(motherIndex)->GetAs<int>("pdgId")) != 443)
      continue;
    if (particle->GetPdgId() == 13)
      daughtersByMother[motherIndex].minus.push_back(index);
    else
      daughtersByMother[motherIndex].plus.push_back(index);
  }

  vector<GenJPsiCandidate> candidates;
  constexpr float muonMass = 0.1056583745;
  for (auto const& [motherIndex, daughters] : daughtersByMother) {
    if (daughters.minus.size() != 1 || daughters.plus.size() != 1) {
      warn() << "Expected exactly one stable mu- and mu+ daughter for GenPart J/psi index " << motherIndex
             << ", found " << daughters.minus.size() << " and " << daughters.plus.size() << ". Skipping." << endl;
      continue;
    }
    int const minusIndex = daughters.minus.front();
    int const plusIndex = daughters.plus.front();
    auto const minus = asNanoGenParticle(genParticles->at(minusIndex));
    auto const plus = asNanoGenParticle(genParticles->at(plusIndex));
    TLorentzVector const momentum = minus->GetFourVector(muonMass) + plus->GetFourVector(muonMass);
    TVector3 const vertex(
        0.5 * (minus->GetAs<float>("vx") + plus->GetAs<float>("vx")),
        0.5 * (minus->GetAs<float>("vy") + plus->GetAs<float>("vy")),
        0.5 * (minus->GetAs<float>("vz") + plus->GetAs<float>("vz")));
    candidates.push_back({motherIndex, minusIndex, plusIndex, minus, plus, momentum, vertex});
  }
  return candidates;
}
