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
  auto recoPATmuons = event->GetCollection("LoosePATMuons");

  for (size_t i = 0; i < recoPATmuons->size(); i++) {
    auto muon1 = asNanoMuon(recoPATmuons->at(i));

    for (size_t j = i + 1; j < recoPATmuons->size(); j++) {
      auto muon2 = asNanoMuon(recoPATmuons->at(j));
      float invMass = (muon1->GetFourVector() + muon2->GetFourVector()).M();
      histogramsHandler->Fill("LoosePATMuon_minv", invMass);
    }
  }

  auto recoDSAmuons = event->GetCollection("LooseDSAMuons");

  for (size_t i = 0; i < recoDSAmuons->size(); i++) {
    auto muon1 = asNanoMuon(recoDSAmuons->at(i));

    for (size_t j = i + 1; j < recoDSAmuons->size(); j++) {
      auto muon2 = asNanoMuon(recoDSAmuons->at(j));
      float invMass = (muon1->GetFourVector() + muon2->GetFourVector()).M();
      histogramsHandler->Fill("LooseDSAMuon_minv", invMass);
    }
  }

  auto recoShiftMuons = event->GetCollection("LooseShiftMuons");

  for (size_t i = 0; i < recoShiftMuons->size(); i++) {
    auto muon1 = asNanoMuon(recoShiftMuons->at(i));

    for (size_t j = i + 1; j < recoShiftMuons->size(); j++) {
      auto muon2 = asNanoMuon(recoShiftMuons->at(j));
      float invMass = (muon1->GetFourVector() + muon2->GetFourVector()).M();
      histogramsHandler->Fill("LooseShiftMuons_minv", invMass);
    }
  }
}

void ShiftHistogramsFiller::FillRecoVsGen(const shared_ptr<Event> event) {
  auto genParticles = event->GetCollection("GenPart");
  auto recoShiftMuons = event->GetCollection("ShiftMuon");

  // first loop over reco muons, for each find the corresponding gen muon using genPartIdx
  for (size_t i = 0; i < recoShiftMuons->size(); i++) {
    auto recoMuon = asNanoMuon(recoShiftMuons->at(i));
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

    auto genMuon = asNanoGenParticle(genParticles->at(genPartIdx));

    histogramsHandler->Fill("RecoVsGenMuon_vx", recoMuon->GetAs<float>("vx"), genMuon->GetAs<float>("vx"));
    histogramsHandler->Fill("RecoVsGenMuon_vy", recoMuon->GetAs<float>("vy"), genMuon->GetAs<float>("vy"));
    histogramsHandler->Fill("RecoVsGenMuon_vz", recoMuon->GetAs<float>("vz"), genMuon->GetAs<float>("vz"));
    histogramsHandler->Fill("RecoVsGenMuon_pt", recoMuon->GetAs<float>("pt"), genMuon->GetAs<float>("pt"));
    histogramsHandler->Fill("RecoVsGenMuon_eta", recoMuon->GetAs<float>("eta"), genMuon->GetAs<float>("eta"));
    histogramsHandler->Fill("RecoVsGenMuon_phi", recoMuon->GetAs<float>("phi"), genMuon->GetAs<float>("phi"));
  }
}

void ShiftHistogramsFiller::Fill(const shared_ptr<Event> event) {
  FillGenLevel(event);
  FillRecoLevel(event);
  FillRecoVsGen(event);
}
