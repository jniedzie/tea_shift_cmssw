// Code style: AI-code. AI-generated candidate-safe J/psi skimmer.
#include "ArgsManager.hpp"
#include "ConfigManager.hpp"
#include "EventReader.hpp"
#include "EventWriter.hpp"

#include <TFile.h>
#include <TKey.h>
#include <TObjString.h>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

namespace {
const string collectionName = "ShiftDimuonVertex";
const string countName = "nShiftDimuonVertex";
const string massName = "ShiftDimuonVertex_mass";
const string sealName = "ShiftJpsiSkimMetadata";
const vector<string> identityFields = {"run", "luminosityBlock", "event"};
const vector<string> floatFields = {"mass", "pt", "pz", "eta", "phi", "vx", "vy", "vz", "chi2", "normalizedChi2", "dca"};
const vector<string> intFields = {"topologyMin", "topologyMax", "isOS", "dcaValid"};
const vector<string> weightFields = {"genWeight", "shiftSamplingProbability", "shiftSamplingWeight",
                                     "shiftSamplingGenWeight", "shiftOriginalGenWeight", "shiftResponseGenWeight",
                                     "shiftRecoveryGenWeight", "shiftProposalGenWeight"};

void Require(bool condition, const string &message) {
  if (!condition) throw runtime_error(message);
}

string BranchName(const string &field) { return collectionName + "_" + field; }

// The writer opens its target with RECREATE. Give it a new owner-private
// sibling file, then publish only a checked and sealed result.
class PendingOutput {
 public:
  explicit PendingOutput(const string &requested) : finalPath(fs::absolute(requested)) {
    Require(!requested.empty() && requested.find("://") == string::npos, "A local output path is required");
    Require(!fs::exists(finalPath), "The skim output already exists");
    fs::create_directories(finalPath.parent_path());
    string pattern = (finalPath.parent_path() / ".shift_jpsi_pending_XXXXXX").string();
    vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int fd = mkstemp(writable.data());
    Require(fd >= 0, "Could not reserve a private output file");
    close(fd);
    temporaryPath = writable.data();
  }

  ~PendingOutput() {
    std::error_code error;
    fs::remove(temporaryPath, error);
  }

  const fs::path &Path() const { return temporaryPath; }

  void Publish() const {
    // A same-directory hard link fails if another writer already owns the name.
    Require(link(temporaryPath.c_str(), finalPath.c_str()) == 0,
            "Could not publish a new skim at the requested path");
  }

 private:
  fs::path finalPath;
  fs::path temporaryPath;
};

TLeaf *CheckedLeaf(TTree &tree, const string &name, const set<string> &types, const string &counter = "") {
  auto branch = tree.GetBranch(name.c_str());
  auto leaf = tree.GetLeaf(name.c_str());
  Require(branch && leaf && branch->GetListOfLeaves()->GetEntries() == 1 &&
              types.count(leaf->GetTypeName()) && leaf->GetLenStatic() == 1,
          "Missing or incompatible required input branch: " + name);
  const auto size = leaf->GetLeafCount();
  Require(counter.empty() ? size == nullptr : size && counter == size->GetName(),
          "Input branch size does not match its declared schema: " + name);
  return leaf;
}

map<string, string> CandidateFields() {
  map<string, string> fields;
  for (const auto &field : floatFields) fields.emplace(field, "Float_t");
  for (const auto &field : intFields) fields.emplace(field, "Int_t");
  return fields;
}

vector<string> KeptBranches() {
  auto branches = identityFields;
  branches.push_back(countName);
  for (const auto &field : floatFields) branches.push_back(BranchName(field));
  for (const auto &field : intFields) branches.push_back(BranchName(field));
  return branches;
}

void CheckConfig(ConfigManager &config) {
  vector<string> kept, removed, eventTrees;
  int entries = 0;
  config.GetVector("branchesToKeep", kept);
  config.GetVector("branchesToRemove", removed);
  config.GetVector("eventsTreeNames", eventTrees);
  config.GetValue("nEvents", entries);
  vector<AddedBranchParams> configured;
  config.GetAddedBranchesParams(configured);
  Require(kept == KeptBranches() && removed.empty() && configured.empty() &&
              eventTrees == vector<string>{"Events"} && entries == -1,
          "The skimmer requires the exact branch allowlist and the full input file");
}

pair<double, double> ReadWindow(ConfigManager &config) {
  string collection;
  vector<string> limits;
  config.GetValue("skimCollection", collection);
  config.GetVector("skimMassWindow", limits);
  Require(collection == collectionName && limits.size() == 2, "The shared J/psi definition is unavailable");
  size_t parsedMin = 0, parsedMax = 0;
  const double lower = stod(limits[0], &parsedMin);
  const double upper = stod(limits[1], &parsedMax);
  Require(parsedMin == limits[0].size() && parsedMax == limits[1].size() &&
              isfinite(lower) && isfinite(upper) && lower > 0.0 && lower < upper,
          "The shared J/psi mass limits are invalid");
  return {lower, upper};
}

vector<string> RetainedWeights(TTree &tree, bool isData) {
  vector<string> retained;
  for (const auto &name : weightFields) {
    if (!tree.GetBranch(name.c_str())) continue;
    Require(!isData, "A declared collision-data source has simulation weight branches");
    CheckedLeaf(tree, name, name == "genWeight" ? set<string>{"Float_t", "Double_t"} : set<string>{"Double_t"});
    retained.push_back(name);
  }
  return retained;
}

void ConfigureWeightClone(const vector<string> &weights) {
  string update = "branchesToKeep += (";
  for (const auto &name : weights) update += " '" + name + "',";
  update += ")";
  Require(PyRun_SimpleString(update.c_str()) == 0, "Could not set the exact writer allowlist");
}

template <typename T>
void CompactSelectedField(TLeaf &leaf, const vector<int> &indices) {
  auto values = static_cast<T *>(leaf.GetValuePointer());
  Require(values, "Could not locate a required candidate branch buffer");
  for (size_t target = 0; target < indices.size(); ++target) {
    const auto value = values[indices[target]];
    Require(isfinite(value), "A retained candidate field contains a nonfinite value: " + string(leaf.GetName()));
    values[target] = value;
  }
}

void ReadCandidateField(TLeaf &leaf, Long64_t entry, int count) {
  Require(leaf.GetBranch()->GetEntry(entry, 1) > 0 && leaf.GetLen() == count,
          "Could not read a required candidate branch: " + string(leaf.GetName()));
}

void SetOutputCount(TLeaf &leaf, size_t count) {
  auto address = leaf.GetValuePointer();
  Require(address != nullptr, "Could not locate TEA's cloned candidate count");
  if (string(leaf.GetTypeName()) == "UInt_t") {
    *static_cast<UInt_t *>(address) = static_cast<UInt_t>(count);
  } else {
    *static_cast<Int_t *>(address) = static_cast<Int_t>(count);
  }
}

void SealOutput(const fs::path &path, const vector<string> &weights, const string &inputKind,
                const string &weightBranch, double massMin, double massMax, Long64_t originalEvents,
                Long64_t selectedEvents, Long64_t selectedPairs) {
  unique_ptr<TFile> output(TFile::Open(path.c_str(), "UPDATE"));
  Require(output && !output->IsZombie() && !output->TestBit(TFile::kRecovered), "The skim ROOT output is unreadable");
  auto tree = output->Get<TTree>("Events");
  Require(tree && tree->GetEntries() == selectedEvents, "Skim output entries do not match the committed event count");

  set<string> expected(identityFields.begin(), identityFields.end());
  expected.insert(weights.begin(), weights.end());
  expected.insert(countName);
  CheckedLeaf(*tree, countName, {"Int_t", "UInt_t"});
  for (const auto &[field, type] : CandidateFields()) {
    const auto name = BranchName(field);
    CheckedLeaf(*tree, name, {type}, countName);
    expected.insert(name);
  }
  set<string> found;
  for (auto item : *tree->GetListOfBranches()) found.insert(item->GetName());
  Require(found == expected, "The written ROOT branch names differ from the approved skim schema");

  vector<string> sideTrees;
  for (auto key : *output->GetListOfKeys()) {
    const string name = key->GetName();
    if (name == "Events") continue;
    auto side = output->Get<TTree>(name.c_str());
    Require(side && side->GetEntries() == 0, "Unexpected payload on a non-event output tree");
    sideTrees.push_back(name);
  }
  for (const auto &name : sideTrees) output->Delete((name + ";*").c_str());

  ostringstream metadata;
  metadata << setprecision(numeric_limits<double>::max_digits10)
           << "{\"schema\":\"shift-jpsi-skim-v1\",\"input_kind\":\"" << inputKind
           << "\",\"mass_min_gev\":" << massMin << ",\"mass_max_gev\":" << massMax
           << ",\"endpoints\":\"inclusive\",\"original_events\":"
           << originalEvents << ",\"selected_events\":" << selectedEvents << ",\"selected_pairs\":" << selectedPairs
           << ",\"weight_branch\":\"" << weightBranch << "\"}";
  output->cd();
  Require(TObjString(metadata.str().c_str()).Write(sealName.c_str()) > 0, "Could not seal the skim metadata");
  output->Close();
}

int RunSkimmer(int argc, char **argv) {
  Require(argc % 2 == 1, "Every skimmer option requires a value");
  const vector<string> required = {"config", "input_path", "output_trees_path", "input_kind"};
  const vector<string> optional = {"weight_branch"};
  auto args = make_unique<ArgsManager>(argc, argv, required, optional);
  const string kind = args->GetString("input_kind").value();
  Require(kind == "collision-data" || kind == "mc", "Specify collision-data or mc as the input kind");
  const bool isData = kind == "collision-data";
  const auto requestedWeight = args->GetString("weight_branch");
  Require(!isData || !requestedWeight, "Collision data must have unit weights");
  PendingOutput output(args->GetString("output_trees_path").value());

  ConfigManager::Initialize(args);
  auto &config = ConfigManager::GetInstance();
  config.SetTreesOutputPath(output.Path().string());
  CheckConfig(config);
  const auto [massMin, massMax] = ReadWindow(config);

  // Read ROOT's exact candidate branch types. Candidate values are never logged.
  gErrorIgnoreLevel = kFatal;
  auto reader = make_shared<EventReader>();
  auto input = reader->GetInputFile();
  auto tree = input->Get<TTree>("Events");
  Require(tree, "The skim input has no Events tree");
  CheckedLeaf(*tree, "run", {"UInt_t"});
  CheckedLeaf(*tree, "luminosityBlock", {"UInt_t"});
  CheckedLeaf(*tree, "event", {"ULong64_t"});
  auto countLeaf = CheckedLeaf(*tree, countName, {"Int_t", "UInt_t"});
  auto massLeaf = CheckedLeaf(*tree, massName, {"Float_t"}, countName);

  vector<TLeaf *> floatLeaves, intLeaves;
  for (const auto &field : floatFields) {
    if (field != "mass") floatLeaves.push_back(CheckedLeaf(*tree, BranchName(field), {"Float_t"}, countName));
  }
  for (const auto &field : intFields) intLeaves.push_back(CheckedLeaf(*tree, BranchName(field), {"Int_t"}, countName));

  const auto weights = RetainedWeights(*tree, isData);
  string weightBranch;
  if (!isData) {
    const vector<string> defaults = {"shiftSamplingGenWeight", "shiftResponseGenWeight", "shiftRecoveryGenWeight",
                                     "shiftProposalGenWeight", "genWeight"};
    auto native = find_if(defaults.begin(), defaults.end(), [&](const string &name) { return tree->GetBranch(name.c_str()); });
    weightBranch = requestedWeight.value_or(native == defaults.end() ? "" : *native);
    Require(find(weights.begin(), weights.end(), weightBranch) != weights.end(), "The required MC weight is missing");
  }
  ConfigureWeightClone(weights);
  EventWriter writer(reader);

  // The TEA writer resets the source tree's branch mask after cloning. Only
  // identities and optional weights are active during TEA's accepted-event read.
  for (auto key : *input->GetListOfKeys()) {
    if (auto sourceTree = input->Get<TTree>(key->GetName())) sourceTree->SetBranchStatus("*", false);
  }
  for (const auto &name : identityFields) tree->SetBranchStatus(name.c_str(), true);
  for (const auto &name : weights) tree->SetBranchStatus(name.c_str(), true);
  tree->SetBranchStatus(countName.c_str(), true);

  const auto originalEvents = reader->GetNevents();
  Require(originalEvents == tree->GetEntries() && originalEvents <= numeric_limits<int>::max(),
          "The reader does not own the full input range");
  Long64_t selectedEvents = 0, selectedPairs = 0;
  for (Long64_t entry = 0; entry < originalEvents; ++entry) {
    Require(countLeaf->GetBranch()->GetEntry(entry, 1) > 0, "Could not read the original candidate count");
    const double rawCount = countLeaf->GetValue();
    Require(rawCount >= 0 && rawCount <= maxCollectionElements, "Original candidate count is outside the safe buffer");
    const int count = static_cast<int>(rawCount);
    if (count == 0) continue;
    ReadCandidateField(*massLeaf, entry, count);
    vector<int> indices;
    for (int index = 0; index < count; ++index) {
      const double mass = massLeaf->GetValue(index);
      Require(isfinite(mass), "An original canonical candidate mass is nonfinite");
      if (mass >= massMin && mass <= massMax) {
        indices.push_back(index);
      }
    }
    if (indices.empty()) continue;

    reader->GetEvent(static_cast<int>(entry));
    for (const auto &name : weights)
      Require(isfinite(tree->GetLeaf(name.c_str())->GetValue()), "A retained MC event weight is nonfinite");
    // The clone shares the original branch buffers. Compress in input order
    // and lower the count before the writer serializes the accepted rows.
    CompactSelectedField<Float_t>(*massLeaf, indices);
    for (auto leaf : floatLeaves) {
      ReadCandidateField(*leaf, entry, count);
      CompactSelectedField<Float_t>(*leaf, indices);
    }
    for (auto leaf : intLeaves) {
      ReadCandidateField(*leaf, entry, count);
      CompactSelectedField<Int_t>(*leaf, indices);
    }
    SetOutputCount(*countLeaf, indices.size());
    writer.AddCurrentEvent("Events");
    ++selectedEvents;
    selectedPairs += static_cast<Long64_t>(indices.size());
  }

  writer.Save();
  SealOutput(output.Path(), weights, kind, weightBranch, massMin, massMax, originalEvents, selectedEvents,
             selectedPairs);
  output.Publish();
  cout << "Verified " << selectedEvents << " skim events with " << selectedPairs << " approved candidates." << endl;
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  umask(0077);
  try {
    return RunSkimmer(argc, argv);
  } catch (const runtime_error &error) {
    cerr << "SHIFT skim failed: " << error.what() << endl;
    return 1;
  } catch (const exception &) {
    cerr << "SHIFT skim failed during the restricted input or output audit." << endl;
    return 1;
  }
}
