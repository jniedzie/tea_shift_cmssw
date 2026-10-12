#include "ArgsManager.hpp"
#include "ConfigManager.hpp"
#include "EventReader.hpp"
#include "ExtensionsHelpers.hpp"
#include "HistogramsFiller.hpp"
#include "HistogramsHandler.hpp"
#include "ShiftHistogramsFiller.hpp"
#include "NanoEventProcessor.hpp"

using namespace std;

int RunHistogrammer(int argc, char** argv) {
  vector<string> requiredArgs = {"config"};
  vector<string> optionalArgs = {"input_path", "output_hists_path"};
  auto args = make_unique<ArgsManager>(argc, argv, requiredArgs, optionalArgs);
  ConfigManager::Initialize(args);

  bool runEventMixing = false;
  try {
    ConfigManager::GetInstance().GetValue("runEventMixing", runEventMixing);
  } catch (const Exception&) {}
  if (runEventMixing)
    return PyRun_SimpleString("run_mixing_stage(eventMixing)") == 0 ? 0 : 1;

  bool enableTruthDiagnostics = true;
  ConfigManager::GetInstance().GetValue("enableTruthDiagnostics", enableTruthDiagnostics);
  // Cutflow and custom fills must share the data unit-weight convention.
  if (!enableTruthDiagnostics && PyRun_SimpleString("weightsBranchName = ''") != 0) return 1;

  auto eventReader = make_shared<EventReader>();
  auto cutFlowManager = make_shared<CutFlowManager>(eventReader);
  auto histogramsHandler = make_shared<HistogramsHandler>();
  auto histogramsFiller = make_unique<HistogramsFiller>(histogramsHandler);
  auto shiftHistogramsFiller = make_unique<ShiftHistogramsFiller>(histogramsHandler, eventReader->GetInputFile());
  auto nanoEventProcessor = make_unique<NanoEventProcessor>();

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
  shiftHistogramsFiller->SaveBdtMetadata();

  cutFlowManager->Print();

  auto& logger = Logger::GetInstance();
  logger.Print();

  return 0;
}

int main(int argc, char** argv) {
  try {
    return RunHistogrammer(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "SHIFT histogramming failed: " << error.what() << std::endl;
    return 1;
  }
}
