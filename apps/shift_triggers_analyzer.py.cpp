#include "ConfigManager.hpp"
#include "CutFlowManager.hpp"
#include "EventReader.hpp"
#include "EventWriter.hpp"
#include "ExtensionsHelpers.hpp"
#include "UserExtensionsHelpers.hpp"
#include "HistogramsHandler.hpp"
#include "Profiler.hpp"
#include "HistogramsFiller.hpp"
#include "ArgsManager.hpp"
#include <iomanip>

using namespace std;

int main(int argc, char **argv) {
  vector<string> requiredArgs = {"config"};
  vector<string> optionalArgs = {"input_path"};

  auto args = make_unique<ArgsManager>(argc, argv, requiredArgs, optionalArgs);
  ConfigManager::Initialize(args);
  auto& config = ConfigManager::GetInstance();

  auto eventReader = make_shared<EventReader>();
  auto histogramsHandler = make_shared<HistogramsHandler>();

  auto hltBranches = eventReader->GetHLTbranchNames();
  auto l1Branches = eventReader->GetL1branchNames();

  map<string, int> passingTriggersCount = {};
  map<string, int> passingL1TriggersCount = {};

  for (int iEvent = 0; iEvent < eventReader->GetNevents(); iEvent++) {
    auto event = eventReader->GetEvent(iEvent);

    for (string hltBranch : hltBranches) {
      bool passesTrigger = event->Get(hltBranch);
      if (passesTrigger) {
        passingTriggersCount[hltBranch]++;
      }
    }

    for (string l1Branch : l1Branches) {
      bool passesL1Trigger = event->Get(l1Branch);
      if (passesL1Trigger) {
        passingL1TriggersCount[l1Branch]++;
      }
    }
  }

  vector<pair<string, double>> triggerEfficiencies;
  for (const auto& [trigger, count] : passingTriggersCount) {
    double efficiency = static_cast<double>(count) / eventReader->GetNevents();
    triggerEfficiencies.push_back({trigger, efficiency});
  }
  sort(triggerEfficiencies.begin(), triggerEfficiencies.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });

  for (const auto& [trigger, efficiency] : triggerEfficiencies) {
    cout << trigger << ": " << std::setprecision(2) << efficiency * 100 << "%" << endl;
  }

  vector<pair<string, double>> l1TriggerEfficiencies;
  for (const auto& [trigger, count] : passingL1TriggersCount) {
    double efficiency = static_cast<double>(count) / eventReader->GetNevents();
    l1TriggerEfficiencies.push_back({trigger, efficiency});
  }
  sort(l1TriggerEfficiencies.begin(), l1TriggerEfficiencies.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });

  for (const auto& [trigger, efficiency] : l1TriggerEfficiencies) {
    cout << trigger << ": " << std::setprecision(2) << efficiency * 100 << "%" << endl;
  }

  auto &logger = Logger::GetInstance();
  logger.Print();

  return 0;
}