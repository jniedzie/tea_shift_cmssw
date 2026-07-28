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
  auto recoMuons = event->GetCollection("LoosePATMuons");
  
  for (size_t i = 0; i < recoMuons->size(); i++) {
    auto muon1 = asNanoMuon(recoMuons->at(i));

    for (size_t j = i + 1; j < recoMuons->size(); j++) {
      auto muon2 = asNanoMuon(recoMuons->at(j));
      float invMass = (muon1->GetFourVector() + muon2->GetFourVector()).M();
      histogramsHandler->Fill("Muon_minv", invMass);
    }
  }
}

void ShiftHistogramsFiller::Fill(const shared_ptr<Event> event) {
  FillGenLevel(event);
  FillRecoLevel(event);
}
