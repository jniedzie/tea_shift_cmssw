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
  std::shared_ptr<HistogramsHandler> histogramsHandler;
  std::unique_ptr<EventProcessor> eventProcessor;
  
  void FillGenLevel(const std::shared_ptr<Event> event);
  void FillRecoLevel(const std::shared_ptr<Event> event);
  void FillRecoVsGen(const std::shared_ptr<Event> event);

  bool IsMotherJPsi(const std::shared_ptr<NanoGenParticle> particle, const std::shared_ptr<PhysicsObjects> genParticles);
};

#endif /* ShiftHistogramsFiller_hpp */
