#include "ArgsManager.hpp"
#include "ConfigManager.hpp"
#include "EventReader.hpp"
#include "ExtensionsHelpers.hpp"
#include "HistogramsFiller.hpp"
#include "HistogramsHandler.hpp"
#include "ShiftHistogramsFiller.hpp"
#include "NanoEventProcessor.hpp"

using namespace std;

int main(int argc, char** argv) {
  vector<string> requiredArgs = {"config"};
  vector<string> optionalArgs = {"input_path", "output_hists_path"};
  auto args = make_unique<ArgsManager>(argc, argv, requiredArgs, optionalArgs);
  ConfigManager::Initialize(args);

  auto eventReader = make_shared<EventReader>();
  auto cutFlowManager = make_shared<CutFlowManager>(eventReader);
  auto histogramsHandler = make_shared<HistogramsHandler>();
  auto histogramsFiller = make_unique<HistogramsFiller>(histogramsHandler);
  auto shiftHistogramsFiller = make_unique<ShiftHistogramsFiller>(histogramsHandler);
  auto nanoEventProcessor = make_unique<NanoEventProcessor>();

  bool enableTruthDiagnostics = true;
  ConfigManager::GetInstance().GetValue("enableTruthDiagnostics", enableTruthDiagnostics);

  cutFlowManager->RegisterCut("initial");

  for (int iEvent = 0; iEvent < eventReader->GetNevents(); iEvent++) {
    auto event = eventReader->GetEvent(iEvent);

    cutFlowManager->UpdateCutFlow("initial");

    map<string, float> weight = {
        {"default", enableTruthDiagnostics ? nanoEventProcessor->GetGenWeight(asNanoEvent(event)) : 1.f}};
    histogramsHandler->SetEventWeights(weight);

    histogramsFiller->FillDefaultVariables(event);
    shiftHistogramsFiller->Fill(event);
  }

  histogramsFiller->FillCutFlow(cutFlowManager);
  histogramsHandler->SaveHistograms();

  cutFlowManager->Print();

  auto& logger = Logger::GetInstance();
  logger.Print();

  return 0;
}
