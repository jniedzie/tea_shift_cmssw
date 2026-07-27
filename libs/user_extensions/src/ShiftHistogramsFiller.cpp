#include "ShiftHistogramsFiller.hpp"

#include "ConfigManager.hpp"
#include "ExtensionsHelpers.hpp"

using namespace std;

ShiftHistogramsFiller::ShiftHistogramsFiller(shared_ptr<HistogramsHandler> histogramsHandler_) : histogramsHandler(histogramsHandler_) {
  // Create a config manager
  auto &config = ConfigManager::GetInstance();

  // Try to read weights branch
  try {
    config.GetValue("weightsBranchName", weightsBranchName);
  } catch (const Exception& e) {
  }

  // Create an event processor
  eventProcessor = make_unique<EventProcessor>();
}

ShiftHistogramsFiller::~ShiftHistogramsFiller() {}

float ShiftHistogramsFiller::GetWeight(const std::shared_ptr<Event> event) {
  // Try to get event weight, otherwise set to 1.0
  float weight = 1.0;
  try {
    weight = event->Get(weightsBranchName);
  } catch (const Exception &e) {
  }
  return weight;
}

void ShiftHistogramsFiller::Fill(const std::shared_ptr<Event> event) {
  
  auto genMuons = event->GetCollection("GenMuon");
  float muonMass = 0.1056583745; // GeV/c^2

  // craete invariant mass for all combinations of gen muons
  for (size_t i = 0; i < genMuons->size(); i++) {
    auto muon1 = asNanoGenParticle(genMuons->at(i));

    for (size_t j = i + 1; j < genMuons->size(); j++) {
      auto muon2 = asNanoGenParticle(genMuons->at(j));
      auto invMass = (muon1->GetFourVector(muonMass) + muon2->GetFourVector(muonMass)).M();
      
      histogramsHandler->Fill("GenMuon_minv", invMass, GetWeight(event));
    }
  }

  // create invariant mass for all combinations of reco muons
  auto recoMuons = event->GetCollection("LoosePATMuons");
  for (size_t i = 0; i < recoMuons->size(); i++) {
    auto muon1 = asNanoMuon(recoMuons->at(i));

    for (size_t j = i + 1; j < recoMuons->size(); j++) {
      auto muon2 = asNanoMuon(recoMuons->at(j));
      auto invMass = (muon1->GetFourVector() + muon2->GetFourVector()).M();
      
      histogramsHandler->Fill("Muon_minv", invMass, GetWeight(event));
    }
  }
}
