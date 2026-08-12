#include "ShiftHistogramsFiller.hpp"

#include "ConfigManager.hpp"

using namespace std;

ShiftHistogramsFiller::ShiftHistogramsFiller(shared_ptr<HistogramsHandler> histogramsHandler_) : histogramsHandler(histogramsHandler_) {
  auto& config = ConfigManager::GetInstance();
  eventProcessor = make_unique<EventProcessor>();
}

ShiftHistogramsFiller::~ShiftHistogramsFiller() {}

void ShiftHistogramsFiller::Fill(const shared_ptr<Event> event) {
  FillGenLevel(event);
  FillRecoLevel(event);
  FillRecoVsGen2D(event);
  FillResolutionPlots(event);
}

void ShiftHistogramsFiller::FillGenLevel(const shared_ptr<Event> event) {
  auto genParticles = event->GetCollection("GenPart");

  auto [genDimuon, genDimuonVertex] = GetGenJPsiDimuonVector(genParticles);

  histogramsHandler->Fill("GenDimuon_pt", genDimuon.Pt());
  histogramsHandler->Fill("GenDimuon_pz", genDimuon.Pz());
  histogramsHandler->Fill("GenDimuon_eta", genDimuon.Eta());
  histogramsHandler->Fill("GenDimuon_phi", genDimuon.Phi());
  histogramsHandler->Fill("GenDimuon_mass", genDimuon.M());
  histogramsHandler->Fill("GenDimuon_vx", genDimuonVertex.X());
  histogramsHandler->Fill("GenDimuon_vy", genDimuonVertex.Y());
  histogramsHandler->Fill("GenDimuon_vz", genDimuonVertex.Z());
}

void ShiftHistogramsFiller::FillRecoLevel(const shared_ptr<Event> event) {
  
}

void ShiftHistogramsFiller::FillRecoVsGen2D(const shared_ptr<Event> event) {
  // single muon
  auto genParticles = event->GetCollection("GenPart");
  auto recoShiftMuons = event->GetCollection("ShiftMuon");

  for (size_t i = 0; i < recoShiftMuons->size(); i++) {
    auto recoMuon = recoShiftMuons->at(i);
    int genPartIdx = recoMuon->Get("genPartIdx");
    if (genPartIdx < 0) {
      warn() << "Reco muon has no corresponding gen muon, skipping." << endl;
      continue;
    }
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
  auto [genJPsiVec, genJPsiVertex] = GetGenJPsiDimuonVector(genParticles);
  if (genJPsiVec.Pt() == 0) return;

  auto recoShiftDimuons = event->GetCollection("ShiftDimuonVertex");
  for (size_t i = 0; i < recoShiftDimuons->size(); i++) {
    auto recoDimuon = recoShiftDimuons->at(i);

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
  auto recoShiftDimuons = event->GetCollection("ShiftDimuonVertex");

  vector<string> shiftMuonTypes = {"DoubleTraversing", "Traversing", "DSA", "Cosmic"};
  map<string, shared_ptr<PhysicsObjects>> recoShiftMuons;
  for (const auto& type : shiftMuonTypes) {
    recoShiftMuons[type] = event->GetCollection("ShiftMuon" + type);
  }

  // Fill single muon resolution plots
  for (const auto& [name, recoCollection] : recoShiftMuons) {
    for (size_t i = 0; i < recoCollection->size(); i++) {
      auto recoMuon = recoCollection->at(i);
      int genPartIdx = recoMuon->Get("genPartIdx");
      if (genPartIdx < 0 || genPartIdx >= genParticles->size()) continue;
      auto genMuon = asNanoGenParticle(genParticles->at(genPartIdx));

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

  // Fill dimuon resolution plots
  auto [genJPsiVec, genJPsiVertex] = GetGenJPsiDimuonVector(genParticles);

  for (size_t i = 0; i < recoShiftDimuons->size(); i++) {
    auto recoDimuon = recoShiftDimuons->at(i);

    histogramsHandler->Fill("DimuonResolution_pt", (recoDimuon->GetAs<float>("pt") - genJPsiVec.Pt()) / genJPsiVec.Pt());
    histogramsHandler->Fill("DimuonResolution_pz", (recoDimuon->GetAs<float>("pz") - genJPsiVec.Pz()) / genJPsiVec.Pz());
    histogramsHandler->Fill("DimuonResolution_eta", (recoDimuon->GetAs<float>("eta") - genJPsiVec.Eta()) / genJPsiVec.Eta());
    histogramsHandler->Fill("DimuonResolution_phi", (recoDimuon->GetAs<float>("phi") - genJPsiVec.Phi()) / genJPsiVec.Phi());
    histogramsHandler->Fill("DimuonResolution_minv", (recoDimuon->GetAs<float>("mass") - genJPsiVec.M()) / genJPsiVec.M());
    histogramsHandler->Fill("DimuonResolution_vx", (recoDimuon->GetAs<float>("vx") - genJPsiVertex.X()) / genJPsiVertex.X());
    histogramsHandler->Fill("DimuonResolution_vy", (recoDimuon->GetAs<float>("vy") - genJPsiVertex.Y()) / genJPsiVertex.Y());
    histogramsHandler->Fill("DimuonResolution_vz", (recoDimuon->GetAs<float>("vz") - genJPsiVertex.Z()) / genJPsiVertex.Z());
  }
}

pair<TLorentzVector, TVector3> ShiftHistogramsFiller::GetGenJPsiDimuonVector(const shared_ptr<PhysicsObjects> genParticles) {
  shared_ptr<NanoGenParticle> muon1 = nullptr;
  shared_ptr<NanoGenParticle> muon2 = nullptr;

  for (size_t i = 0; i < genParticles->size(); i++) {
    auto particle = asNanoGenParticle(genParticles->at(i));
    if (particle->IsMotherJPsi(genParticles)) {
      if (!muon1)
        muon1 = particle;
      else
        muon2 = particle;
    }
    if (muon1 && muon2) break;  // found both muons, no need to continue
  }

  if (!muon1 || !muon2) {
    warn() << "Could not find both muons from JPsi decay." << endl;
    return make_pair(TLorentzVector(), TVector3());  // return a zero vector and zero vertex
  }

  float muonMass = 0.1056583745;  // GeV/c^2
  TLorentzVector genJPsiVec = muon1->GetFourVector(muonMass) + muon2->GetFourVector(muonMass);
  TVector3 muon1Vertex(muon1->GetAs<float>("vx"), muon1->GetAs<float>("vy"), muon1->GetAs<float>("vz"));

  return make_pair(genJPsiVec, muon1Vertex);
}
