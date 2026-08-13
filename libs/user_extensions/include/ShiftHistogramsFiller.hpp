#ifndef ShiftHistogramsFiller_hpp
#define ShiftHistogramsFiller_hpp

#include "Event.hpp"
#include "EventProcessor.hpp"
#include "Helpers.hpp"
#include "HistogramsHandler.hpp"
#include "ExtensionsHelpers.hpp"

class ShiftHistogramsFiller {
 public:
  ShiftHistogramsFiller(std::shared_ptr<HistogramsHandler> histogramsHandler_);
  ~ShiftHistogramsFiller();

  void Fill(const std::shared_ptr<Event> event);

 private:
  struct GenJPsiCandidate {
    int motherIdx;
    int muonMinusIdx;
    int muonPlusIdx;
    std::shared_ptr<NanoGenParticle> muonMinus;
    std::shared_ptr<NanoGenParticle> muonPlus;
    TLorentzVector momentum;
    TVector3 vertex;
  };

  std::shared_ptr<HistogramsHandler> histogramsHandler;
  std::unique_ptr<EventProcessor> eventProcessor;
  
  void FillGenLevel(const std::shared_ptr<Event> event);
  void FillRecoLevel(const std::shared_ptr<Event> event);
  void FillRecoVsGen2D(const std::shared_ptr<Event> event);
  void FillResolutionPlots(const std::shared_ptr<Event> event);
  void FillEfficiencies(const std::shared_ptr<Event> event);
  
  std::vector<GenJPsiCandidate> GetGenJPsiCandidates(const std::shared_ptr<PhysicsObjects> genParticles);
  std::pair<TLorentzVector, TVector3> GetGenJPsiDimuonVector(const std::shared_ptr<PhysicsObjects> genParticles);
};

#endif /* ShiftHistogramsFiller_hpp */
