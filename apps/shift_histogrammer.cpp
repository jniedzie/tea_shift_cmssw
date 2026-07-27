#include "ArgsManager.hpp"
#include "ConfigManager.hpp"
#include "EventReader.hpp"
#include "ExtensionsHelpers.hpp"
#include "HistogramsFiller.hpp"
#include "HistogramsHandler.hpp"

#include "ShiftHistogramsFiller.hpp"

using namespace std;

int main(int argc, char **argv) {
  vector<string> requiredArgs = {"config"};
  vector<string> optionalArgs = {"input_path", "output_hists_path"};
  auto args = make_unique<ArgsManager>(argc, argv, requiredArgs, optionalArgs);
  ConfigManager::Initialize(args);

  auto eventReader = make_shared<EventReader>();
  auto histogramsHandler = make_shared<HistogramsHandler>();
  auto histogramsFiller = make_unique<HistogramsFiller>(histogramsHandler);
  auto shiftHistogramsFiller = make_unique<ShiftHistogramsFiller>(histogramsHandler);

  
  for (int iEvent = 0; iEvent < eventReader->GetNevents(); iEvent++) {
    auto event = eventReader->GetEvent(iEvent);

    histogramsFiller->FillDefaultVariables(event);
    shiftHistogramsFiller->Fill(event);
  }

  histogramsHandler->SaveHistograms();

  auto &logger = Logger::GetInstance();
  logger.Print();

  return 0;
}