#ifndef ShiftHistogramsFiller_hpp
#define ShiftHistogramsFiller_hpp

#include "Event.hpp"
#include "EventProcessor.hpp"
#include "Helpers.hpp"
#include "HistogramsHandler.hpp"
#include "ExtensionsHelpers.hpp"

class ShiftHistogramsFiller {
 public:
  ShiftHistogramsFiller(std::shared_ptr<HistogramsHandler> histogramsHandler_, TFile* inputFile = nullptr);
  ~ShiftHistogramsFiller();

  void Fill(const std::shared_ptr<Event> event);
  void SaveBdtMetadata();

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
  bool enableTruthDiagnostics = true;
  bool weightMcDiagnostics = false;
  bool enableBdtDiagnostics = false;
  bool bdtTruthDiagnostics = true;
  bool bdtScoreAvailable = false;
  std::string bdtWeightsBranch;
  std::string bdtInputMetadata;
  long long bdtEvents = 0, bdtPairs = 0;
  double bdtEventSumw = 0., bdtEventSumw2 = 0.;
  double bdtPairSumw = 0., bdtPairSumw2 = 0.;
  std::map<std::string, long long> bdtLabelPairs;
  std::map<std::string, long long> bdtNonfiniteMapCoordinates;
  void SetupBdtDiagnostics(TFile* inputFile);
  void FillBdtDiagnostics(const std::shared_ptr<Event>& event);
  std::string BdtTruthLabel(const std::shared_ptr<PhysicsObjects>& muons,
                           const std::shared_ptr<PhysicsObjects>& particles,
                           int first, int second) const;
  std::map<std::string, std::pair<std::string, std::string>> constrainedMuonVariables;
  void FillDiagnostic(const std::string& name, double value);
  int HitTruthIndex(const std::shared_ptr<PhysicsObject>& muon) const;
  int OriginTruthIndex(const std::shared_ptr<PhysicsObject>& muon) const;
  const GenJPsiCandidate* MatchDimuon(const std::shared_ptr<PhysicsObject>& dimuon,
      const std::shared_ptr<PhysicsObjects>& muons, const std::vector<GenJPsiCandidate>& candidates) const;
  
  void FillGenLevel(const std::shared_ptr<Event> event);
  void FillRecoLevel(const std::shared_ptr<Event> event);
  void FillConstrainedMuons(const std::shared_ptr<Event> event);
  void FillDimuonKinematics(const std::shared_ptr<Event> event);
  void FillRecoVsGen2D(const std::shared_ptr<Event> event);
  void FillResolutionPlots(const std::shared_ptr<Event> event);
  void FillEfficiencies(const std::shared_ptr<Event> event);
  void FillDetectorDiagnostics(const std::shared_ptr<Event> event);
  void FillDimuonOrigins(const std::shared_ptr<Event> event);
  
  std::vector<GenJPsiCandidate> GetGenJPsiCandidates(const std::shared_ptr<PhysicsObjects> genParticles);
};

#endif /* ShiftHistogramsFiller_hpp */
