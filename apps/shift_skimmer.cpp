// Code style: AI-code. AI-generated candidate-safe J/psi skimmer.
#include "ArgsManager.hpp"
#include "ConfigManager.hpp"
#include "EventReader.hpp"
#include "EventWriter.hpp"

#include <TFile.h>
#include <TObjString.h>

#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

namespace {

struct SkimOutput {
  explicit SkimOutput(const string& destination) : path(fs::absolute(destination)) {
    if (fs::exists(path)) throw runtime_error("The skim output already exists");
    fs::create_directories(path.parent_path());
    string pattern = (path.parent_path() / ".shift_jpsi_pending_XXXXXX").string();
    int fd = mkstemp(pattern.data());
    if (fd < 0) throw runtime_error("Could not reserve a private skim output");
    close(fd);
    temporary = pattern;
  }

  ~SkimOutput() {
    error_code error;
    fs::remove(temporary, error);
  }

  void Publish() const {
    if (link(temporary.c_str(), path.c_str()) != 0)
      throw runtime_error("Could not publish a new skim at the requested path");
  }

  fs::path path;
  string temporary;
};

template <typename T>
void KeepRows(TLeaf& leaf, const vector<int>& rows) {
  auto* values = static_cast<T*>(leaf.GetValuePointer());
  for (size_t index = 0; index < rows.size(); ++index)
    values[index] = values[rows[index]];
}

vector<TLeaf*> CandidateLeaves(TTree& tree, TLeaf* count, const string& collection,
                               const vector<string>& fields, const string& type) {
  vector<TLeaf*> leaves;
  for (auto const& field : fields) {
    string name = collection + "_" + field;
    auto* leaf = tree.GetLeaf(name.c_str());
    if (!leaf || leaf->GetLeafCount() != count || leaf->GetTypeName() != type || leaf->GetLenStatic() != 1)
      throw runtime_error("Missing or incompatible candidate branch: " + name);
    leaves.push_back(leaf);
  }
  return leaves;
}

void WriteMetadata(const string& path, const string& kind, const string& weight, double lower, double upper,
                   Long64_t originalEvents, Long64_t selectedEvents, Long64_t selectedPairs) {
  unique_ptr<TFile> file(TFile::Open(path.c_str(), "UPDATE"));
  if (!file || file->IsZombie()) throw runtime_error("Could not reopen the skim output");

  // TEA clones side trees too. Only Events was filled and belongs in the skim.
  vector<string> sideTrees;
  for (auto key : *file->GetListOfKeys()) {
    string name = key->GetName();
    if (name != "Events") sideTrees.push_back(name);
  }
  for (auto const& name : sideTrees) file->Delete((name + ";*").c_str());

  ostringstream metadata;
  metadata << setprecision(numeric_limits<double>::max_digits10)
           << "{\"schema\":\"shift-jpsi-skim-v1\",\"input_kind\":\"" << kind
           << "\",\"mass_min_gev\":" << lower << ",\"mass_max_gev\":" << upper
           << ",\"endpoints\":\"inclusive\",\"original_events\":" << originalEvents
           << ",\"selected_events\":" << selectedEvents << ",\"selected_pairs\":" << selectedPairs
           << ",\"weight_branch\":\"" << weight << "\"}";
  file->cd();
  if (TObjString(metadata.str().c_str()).Write("ShiftJpsiSkimMetadata") <= 0)
    throw runtime_error("Could not write skim metadata");
}

int RunSkimmer(int argc, char** argv) {
  auto args = make_unique<ArgsManager>(argc, argv,
      vector<string>{"config", "input_path", "output_trees_path", "input_kind"}, vector<string>{"weight_branch"});
  string const kind = args->GetString("input_kind").value();
  auto const requestedWeight = args->GetString("weight_branch");
  bool const isData = kind == "collision-data";
  if ((!isData && kind != "mc") || (isData && requestedWeight))
    throw runtime_error("Specify collision-data without a weight, or mc with its event weight");

  SkimOutput output(args->GetString("output_trees_path").value());
  ConfigManager::Initialize(args);
  auto& config = ConfigManager::GetInstance();
  config.SetTreesOutputPath(output.temporary);
  string collection;
  vector<string> limits, floats, ints, branches, weightNames;
  config.GetValue("skimCollection", collection);
  config.GetVector("skimMassWindow", limits);
  config.GetVector("floatFields", floats);
  config.GetVector("intFields", ints);
  config.GetVector("branchesToKeep", branches);
  config.GetVector("mcWeightBranches", weightNames);
  double const lower = stod(limits.at(0));
  double const upper = stod(limits.at(1));

  gErrorIgnoreLevel = kFatal;
  auto reader = make_shared<EventReader>();
  auto* tree = reader->GetInputFile()->Get<TTree>("Events");
  if (!tree) throw runtime_error("The skim input has no Events tree");
  auto* count = tree->GetLeaf(("n" + collection).c_str());
  auto* mass = tree->GetLeaf((collection + "_mass").c_str());
  if (!count || !mass || count->GetLeafCount() || count->GetLenStatic() != 1 ||
      (string(count->GetTypeName()) != "Int_t" && string(count->GetTypeName()) != "UInt_t"))
    throw runtime_error("Missing or incompatible dimuon count or mass");
  auto const floatLeaves = CandidateLeaves(*tree, count, collection, floats, "Float_t");
  auto const intLeaves = CandidateLeaves(*tree, count, collection, ints, "Int_t");

  string weight;
  for (auto const& name : weightNames) {
    if (!tree->GetBranch(name.c_str())) continue;
    if (isData) throw runtime_error("The collision-data input has MC weights");
    if (requestedWeight && name == *requestedWeight) weight = name;
    if (!requestedWeight && weight.empty() &&
        (name == "genWeight" || (name.find("GenWeight") != string::npos && name != "shiftOriginalGenWeight")))
      weight = name;
  }
  if (!isData && weight.empty()) throw runtime_error("The requested MC weight is missing");

  EventWriter writer(reader);
  // The writer restores the input branch mask after cloning. Read the mass
  // first, then the other retained branches only for accepted events.
  tree->SetBranchStatus("*", false);
  for (auto const& name : branches)
    if (tree->GetBranch(name.c_str())) tree->SetBranchStatus(name.c_str(), true);
  Long64_t const original = tree->GetEntries();
  Long64_t selected = 0, pairs = 0;
  vector<int> rows;
  for (Long64_t entry = 0; entry < original; ++entry) {
    if (count->GetBranch()->GetEntry(entry) <= 0) throw runtime_error("Could not read the dimuon count");
    double const rawSize = count->GetValue();
    if (rawSize < 0 || rawSize > maxCollectionElements) throw runtime_error("Invalid dimuon buffer size");
    int const size = rawSize;
    if (size == 0) continue;
    if (mass->GetBranch()->GetEntry(entry) <= 0) throw runtime_error("Could not read the dimuon mass");

    rows.clear();
    for (int index = 0; index < size; ++index) {
      double const value = mass->GetValue(index);
      if (lower <= value && value <= upper) rows.push_back(index);
    }
    if (rows.empty()) continue;
    if (tree->GetEntry(entry) <= 0) throw runtime_error("Could not read the selected event");

    for (auto* leaf : floatLeaves) KeepRows<Float_t>(*leaf, rows);
    for (auto* leaf : intLeaves) KeepRows<Int_t>(*leaf, rows);
    if (count->IsUnsigned())
      *static_cast<UInt_t*>(count->GetValuePointer()) = rows.size();
    else
      *static_cast<Int_t*>(count->GetValuePointer()) = rows.size();
    writer.AddCurrentEvent("Events");
    ++selected;
    pairs += rows.size();
  }
  writer.Save();
  WriteMetadata(output.temporary, kind, weight, lower, upper, original, selected, pairs);
  output.Publish();
  cout << "Kept " << selected << " events with " << pairs << " J/psi pairs." << endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  umask(0077);
  try {
    return RunSkimmer(argc, argv);
  } catch (const exception& error) {
    cerr << "SHIFT skim failed: " << error.what() << endl;
    return 1;
  }
}
