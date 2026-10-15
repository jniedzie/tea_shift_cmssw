// Code style: human-code.
#include "ArgsManager.hpp"
#include "ConfigManager.hpp"
#include "EventReader.hpp"
#include "ExtensionsHelpers.hpp"
#include "HistogramsFiller.hpp"
#include "HistogramsHandler.hpp"
#include "ShiftHistogramsFiller.hpp"

#include <cmath>
#include <stdexcept>

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

  bool recoDataMode = false;
  try {
    ConfigManager::GetInstance().GetValue("recoDataMode", recoDataMode);
  } catch (const Exception&) {}
  // Turning off MC truth plots does not change the event normalization.
  if (recoDataMode && PyRun_SimpleString("weightsBranchName = ''") != 0) return 1;
  string weightsBranchName;
  ConfigManager::GetInstance().GetValue("weightsBranchName", weightsBranchName);

  auto eventReader = make_shared<EventReader>();
  auto cutFlowManager = make_shared<CutFlowManager>(eventReader);
  auto histogramsHandler = make_shared<HistogramsHandler>();
  auto histogramsFiller = make_unique<HistogramsFiller>(histogramsHandler);
  auto shiftHistogramsFiller = make_unique<ShiftHistogramsFiller>(histogramsHandler, eventReader->GetInputFile());

  cutFlowManager->RegisterCut("initial");

  for (int iEvent = 0; iEvent < eventReader->GetNevents(); iEvent++) {
    auto event = eventReader->GetEvent(iEvent);

    cutFlowManager->UpdateCutFlow("initial");

    double eventWeight = 1.0;
    if (!weightsBranchName.empty()) {
      try {
        eventWeight = event->GetAs<double>(weightsBranchName);
      } catch (const std::exception& error) {
        throw std::runtime_error("Cannot read configured event weight " + weightsBranchName + ": " + error.what());
      }
      if (!std::isfinite(eventWeight)) throw std::runtime_error("Configured event weight is not finite");
    }
    map<string, double> weights = {{"default", eventWeight}};
    histogramsHandler->SetEventWeights(weights);

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
