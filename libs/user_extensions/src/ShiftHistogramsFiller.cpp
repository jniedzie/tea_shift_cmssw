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

    histogramsHandler->Fill("GenMuon_x", muon1->GetAs<float>("vx"));
    histogramsHandler->Fill("GenMuon_y", muon1->GetAs<float>("vy"));
    histogramsHandler->Fill("GenMuon_z", muon1->GetAs<float>("vz"));
    histogramsHandler->Fill("GenMuon_logZ", log10(abs(muon1->GetAs<float>("vz"))));

    histogramsHandler->Fill("GenMuon_pt", muon1->GetAs<float>("pt"));
    histogramsHandler->Fill("GenMuon_eta", muon1->GetAs<float>("eta"));
    histogramsHandler->Fill("GenMuon_phi", muon1->GetAs<float>("phi"));

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

void ShiftHistogramsFiller::Fill(const shared_ptr<Event> event) {
  FillGenLevel(event);
  FillRecoLevel(event);
}
