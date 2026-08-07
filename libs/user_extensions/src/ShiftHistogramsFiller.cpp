#include "ShiftHistogramsFiller.hpp"

#include "ConfigManager.hpp"

using namespace std;

ShiftHistogramsFiller::ShiftHistogramsFiller(shared_ptr<HistogramsHandler> histogramsHandler_) : histogramsHandler(histogramsHandler_) {
  auto& config = ConfigManager::GetInstance();
  eventProcessor = make_unique<EventProcessor>();
}

ShiftHistogramsFiller::~ShiftHistogramsFiller() {}

bool ShiftHistogramsFiller::IsMotherJPsi(const shared_ptr<NanoGenParticle> particle, const shared_ptr<PhysicsObjects> genParticles) {
  Short_t motherIndex = particle->Get("genPartIdxMother");
  if (motherIndex < 0) return false;
  auto mother = asNanoGenParticle(genParticles->at(motherIndex));
  int motherPdgId = mother->GetPdgId();
  return (motherPdgId == 443);
}

void ShiftHistogramsFiller::FillGenLevel(const shared_ptr<Event> event) {
  auto genMuons = event->GetCollection("GenMuon");
  auto genParticles = event->GetCollection("GenPart");
  float muonMass = 0.1056583745;  // GeV/c^2

  for (size_t i = 0; i < genMuons->size(); i++) {
    auto muon1 = asNanoGenParticle(genMuons->at(i));
    // if (!IsMotherJPsi(muon1, genParticles)) continue;
    for (size_t j = i + 1; j < genMuons->size(); j++) {
      auto muon2 = asNanoGenParticle(genMuons->at(j));
      // if (!IsMotherJPsi(muon2, genParticles)) continue;

      float invMass = (muon1->GetFourVector(muonMass) + muon2->GetFourVector(muonMass)).M();
      histogramsHandler->Fill("GenMuon_minv", invMass);
    }
  }
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
    if (genPartIdx >= genParticles->size()) {
      warn() << "genPartIdx is out of bounds for genParticles collection, skipping." << endl;
      continue;
    }
    // check if the gen particle is a muon
    auto genParticle = asNanoGenParticle(genParticles->at(genPartIdx));
    if (abs(genParticle->GetPdgId()) != 13) {
      warn() << "Gen particle corresponding to reco muon is not a muon, skipping." << endl;
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
  auto genJPsis = event->GetCollection("GenJPsi");
  auto recoShiftDimuons = event->GetCollection("ShiftDimuonVertex");

  // check that there's just one gen JPsi
  if (genJPsis->size() != 1) {
    warn() << "Expected exactly one gen JPsi, but found " << genJPsis->size() << ". Skipping dimuon comparison." << endl;
    return;
  }

  auto genJPsi = genJPsis->at(0);
  auto genMuons = event->GetCollection("GenMuon");
  

  auto genJPsiVec = GetGenJPsiDimuonVector(genParticles);
  if (genJPsiVec.Pt() == 0) return;

  for (size_t i = 0; i < recoShiftDimuons->size(); i++) {
    auto recoDimuon = recoShiftDimuons->at(i);

    histogramsHandler->Fill("RecoVsGenJPsi_vx", recoDimuon->GetAs<float>("vx"), genJPsiVec.X());
    histogramsHandler->Fill("RecoVsGenJPsi_vy", recoDimuon->GetAs<float>("vy"), genJPsiVec.Y());
    histogramsHandler->Fill("RecoVsGenJPsi_vz", recoDimuon->GetAs<float>("vz"), genJPsiVec.Z());
    histogramsHandler->Fill("RecoVsGenJPsi_pt", recoDimuon->GetAs<float>("pt"), genJPsiVec.Pt());
    // histogramsHandler->Fill("RecoVsGenJPsi_pz", recoDimuon->GetAs<float>("pz"), genJPsiVec.Pz());
    histogramsHandler->Fill("RecoVsGenJPsi_eta", recoDimuon->GetAs<float>("eta"), genJPsiVec.Eta());
    histogramsHandler->Fill("RecoVsGenJPsi_phi", recoDimuon->GetAs<float>("phi"), genJPsiVec.Phi());
    histogramsHandler->Fill("RecoVsGenJPsi_minv", recoDimuon->GetAs<float>("mass"), genJPsiVec.M());
  }

}

TLorentzVector ShiftHistogramsFiller::GetGenJPsiDimuonVector(const shared_ptr<PhysicsObjects> genParticles) {
  shared_ptr<NanoGenParticle> muon1 = nullptr;
  shared_ptr<NanoGenParticle> muon2 = nullptr;

  for (size_t i = 0; i < genParticles->size(); i++) {
    auto particle = asNanoGenParticle(genParticles->at(i));
    if (IsMotherJPsi(particle, genParticles)) {
      if (!muon1) muon1 = particle;
      else muon2 = particle;
    }
    if (muon1 && muon2) break;  // found both muons, no need to continue
  }

  if (!muon1 || !muon2) {
    warn() << "Could not find both muons from JPsi decay." << endl;
    return TLorentzVector();  // return a zero vector
  }

  float muonMass = 0.1056583745;  // GeV/c^2
  TLorentzVector genJPsiVec = muon1->GetFourVector(muonMass) + muon2->GetFourVector(muonMass);

  return genJPsiVec;
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
    histogramsHandler->Fill("MuonResolution_eta", recoMuon->GetAs<float>("eta") - genMuon->GetAs<float>("eta"));
    histogramsHandler->Fill("MuonResolution_phi", recoMuon->GetAs<float>("phi") - genMuon->GetAs<float>("phi"));
  }

  // Fill dimuon resolution plots 
  auto genJPsiVec = GetGenJPsiDimuonVector(genParticles);

  for (size_t i = 0; i < recoShiftDimuons->size(); i++) {
    auto recoDimuon = recoShiftDimuons->at(i);

    histogramsHandler->Fill("DimuonResolution_pt", (recoDimuon->GetAs<float>("pt") - genJPsiVec.Pt()) / genJPsiVec.Pt());
    // histogramsHandler->Fill("DimuonResolution_pz", (recoDimuon->GetAs<float>("pz") - genJPsiVec.Pz()) / genJPsiVec.Pz());
    histogramsHandler->Fill("DimuonResolution_eta", recoDimuon->GetAs<float>("eta") - genJPsiVec.Eta());
    histogramsHandler->Fill("DimuonResolution_phi", recoDimuon->GetAs<float>("phi") - genJPsiVec.Phi());
    histogramsHandler->Fill("DimuonResolution_minv", (recoDimuon->GetAs<float>("mass") - genJPsiVec.M()) / genJPsiVec.M());
  }
}

void ShiftHistogramsFiller::Fill(const shared_ptr<Event> event) {
  FillGenLevel(event);
  FillRecoLevel(event);
  FillRecoVsGen2D(event);
  FillResolutionPlots(event);
}
