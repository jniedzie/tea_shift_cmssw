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

  auto genDimuon = GetGenJPsiDimuonVector(genParticles);

  histogramsHandler->Fill("GenDimuon_pt", genDimuon.Pt());
  histogramsHandler->Fill("GenDimuon_pz", genDimuon.Pz());
  histogramsHandler->Fill("GenDimuon_eta", genDimuon.Eta());
  histogramsHandler->Fill("GenDimuon_phi", genDimuon.Phi());
  histogramsHandler->Fill("GenDimuon_mass", genDimuon.M());
  histogramsHandler->Fill("GenDimuon_vx", genDimuon.X());
  histogramsHandler->Fill("GenDimuon_vy", genDimuon.Y());
  histogramsHandler->Fill("GenDimuon_vz", genDimuon.Z());
}

void ShiftHistogramsFiller::FillRecoLevel(const shared_ptr<Event> event) {
  auto recoShiftMuons = event->GetCollection("LooseShiftMuon");

  for (size_t i = 0; i < recoShiftMuons->size(); i++) {
    auto muon1 = asNanoMuon(recoShiftMuons->at(i));
    for (size_t j = i + 1; j < recoShiftMuons->size(); j++) {
      auto muon2 = asNanoMuon(recoShiftMuons->at(j));
      float invMass = (muon1->GetFourVector() + muon2->GetFourVector()).M();
      histogramsHandler->Fill("LooseShiftMuon_minv", invMass);
    }
  }
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

    histogramsHandler->Fill("RecoVsGenMuon_vx", recoMuon->GetAs<float>("vx"), genMuon->GetAs<float>("vx"));
    histogramsHandler->Fill("RecoVsGenMuon_vy", recoMuon->GetAs<float>("vy"), genMuon->GetAs<float>("vy"));
    histogramsHandler->Fill("RecoVsGenMuon_vz", recoMuon->GetAs<float>("vz"), genMuon->GetAs<float>("vz"));
    histogramsHandler->Fill("RecoVsGenMuon_pt", recoMuon->GetAs<float>("pt"), genMuon->GetAs<float>("pt"));
    histogramsHandler->Fill("RecoVsGenMuon_pz", recoMuon->GetAs<float>("pz"), genMuon->GetAs<float>("pz"));
    histogramsHandler->Fill("RecoVsGenMuon_eta", recoMuon->GetAs<float>("eta"), genMuon->GetAs<float>("eta"));
    histogramsHandler->Fill("RecoVsGenMuon_phi", recoMuon->GetAs<float>("phi"), genMuon->GetAs<float>("phi"));
  }

  // dimuon
  auto genJPsiVec = GetGenJPsiDimuonVector(genParticles);
  if (genJPsiVec.Pt() == 0) return;

  auto recoShiftDimuons = event->GetCollection("ShiftDimuonVertex");
  for (size_t i = 0; i < recoShiftDimuons->size(); i++) {
    auto recoDimuon = recoShiftDimuons->at(i);

    histogramsHandler->Fill("RecoVsGenDimuon_vx", recoDimuon->GetAs<float>("vx"), genJPsiVec.X());
    histogramsHandler->Fill("RecoVsGenDimuon_vy", recoDimuon->GetAs<float>("vy"), genJPsiVec.Y());
    histogramsHandler->Fill("RecoVsGenDimuon_vz", recoDimuon->GetAs<float>("vz"), genJPsiVec.Z());
    histogramsHandler->Fill("RecoVsGenDimuon_pt", recoDimuon->GetAs<float>("pt"), genJPsiVec.Pt());
    histogramsHandler->Fill("RecoVsGenDimuon_pz", recoDimuon->GetAs<float>("pz"), genJPsiVec.Pz());
    histogramsHandler->Fill("RecoVsGenDimuon_eta", recoDimuon->GetAs<float>("eta"), genJPsiVec.Eta());
    histogramsHandler->Fill("RecoVsGenDimuon_phi", recoDimuon->GetAs<float>("phi"), genJPsiVec.Phi());
    histogramsHandler->Fill("RecoVsGenDimuon_minv", recoDimuon->GetAs<float>("mass"), genJPsiVec.M());
  }
}

void ShiftHistogramsFiller::FillResolutionPlots(const shared_ptr<Event> event) {
  // plot pt, pz, eta, phi, minv resolutions for reco vs gen muons and dimuons
  // Implementation for resolution plots
  auto genMuons = event->GetCollection("GenMuon");
  auto genParticles = event->GetCollection("GenPart");
  auto recoShiftMuons = event->GetCollection("ShiftMuon");
  auto recoShiftDimuons = event->GetCollection("ShiftDimuonVertex");

  // Fill single muon resolution plots
  for (size_t i = 0; i < recoShiftMuons->size(); i++) {
    auto recoMuon = recoShiftMuons->at(i);
    int genPartIdx = recoMuon->Get("genPartIdx");
    if (genPartIdx < 0 || genPartIdx >= genParticles->size()) continue;
    auto genMuon = asNanoGenParticle(genParticles->at(genPartIdx));

    histogramsHandler->Fill("MuonResolution_pt", (recoMuon->GetAs<float>("pt") - genMuon->GetAs<float>("pt")) / genMuon->GetAs<float>("pt"));
    histogramsHandler->Fill("MuonResolution_pz", (recoMuon->GetAs<float>("pz") - genMuon->GetAs<float>("pz")) / genMuon->GetAs<float>("pz"));
    histogramsHandler->Fill("MuonResolution_eta", (recoMuon->GetAs<float>("eta") - genMuon->GetAs<float>("eta")) / genMuon->GetAs<float>("eta"));
    histogramsHandler->Fill("MuonResolution_phi", (recoMuon->GetAs<float>("phi") - genMuon->GetAs<float>("phi")) / genMuon->GetAs<float>("phi"));
  }

  // Fill dimuon resolution plots
  auto genJPsiVec = GetGenJPsiDimuonVector(genParticles);

  for (size_t i = 0; i < recoShiftDimuons->size(); i++) {
    auto recoDimuon = recoShiftDimuons->at(i);

    histogramsHandler->Fill("DimuonResolution_pt", (recoDimuon->GetAs<float>("pt") - genJPsiVec.Pt()) / genJPsiVec.Pt());
    histogramsHandler->Fill("DimuonResolution_pz", (recoDimuon->GetAs<float>("pz") - genJPsiVec.Pz()) / genJPsiVec.Pz());
    histogramsHandler->Fill("DimuonResolution_eta", (recoDimuon->GetAs<float>("eta") - genJPsiVec.Eta()) / genJPsiVec.Eta());
    histogramsHandler->Fill("DimuonResolution_phi", (recoDimuon->GetAs<float>("phi") - genJPsiVec.Phi()) / genJPsiVec.Phi());
    histogramsHandler->Fill("DimuonResolution_minv", (recoDimuon->GetAs<float>("mass") - genJPsiVec.M()) / genJPsiVec.M());
  }
}

TLorentzVector ShiftHistogramsFiller::GetGenJPsiDimuonVector(const shared_ptr<PhysicsObjects> genParticles) {
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
    return TLorentzVector();  // return a zero vector
  }

  float muonMass = 0.1056583745;  // GeV/c^2
  TLorentzVector genJPsiVec = muon1->GetFourVector(muonMass) + muon2->GetFourVector(muonMass);

  // set the origin of the TLorentzVector to the average of the two muon vertices
  float vx = (muon1->GetAs<float>("vx") + muon2->GetAs<float>("vx")) / 2.0;
  float vy = (muon1->GetAs<float>("vy") + muon2->GetAs<float>("vy")) / 2.0;
  float vz = (muon1->GetAs<float>("vz") + muon2->GetAs<float>("vz")) / 2.0;
  genJPsiVec.SetXYZT(vx, vy, vz, genJPsiVec.T());

  return genJPsiVec;
}