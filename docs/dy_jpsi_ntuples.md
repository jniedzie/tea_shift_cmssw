 
# DY and J/ψ ntuples: quick start

These are the  DY and direct-J/ψ MC samples used by the SHIFT analysis as of **2026-10-23**. The fixed target is near z = +148 m, where the tertiary collimators are, with particles travelling toward CMS. Events are propagated through the LSS and CMS materials and B-fileds, detector simulation, and custom muon and dimuon reconstruction.
## Where to find the files

The ntuples are on EOS, replace `<sample>` with a directory from the tables below:
```text
/eos/cms/store/group/phys_exotica/xtracks/jeremi/shift_cmssw/<sample>/20261009-dimuon-bdt-merged-first-test-v1/nanoAOD_merged/ntuple_*.root
```


## Samples and cross sections

Cross sections are in pb, before detector and reconstruction selections.`D` is the sum of generator weights in the full generated sample, including those events that were not reconstructed. Some jobs are missing, but their events remain in the normalization denominator - this will need to be corrected later on.
### DY → μμ

DY includes γ*/Z exchange. The bins refer to the generated hard-process mass `mHat`, in GeV. In directory names, `p` means a decimal point and `inf` means there is no upper cut.
| mHat [GeV] | Sample directory | Files | σ [pb] | D |
|---|---|---:|---:|---:|
| 0.211317–0.5 | `dy_mHat0p211317to0p5GeV` | 25 | 326341.85535591666 | 10000 |
| 0.5–1 | `dy_mHat0p5to1GeV` | 25 | 132153.1881897009 | 10000 |
| 1–2 | `dy_mHat1to2GeV` | 25 | 25020.91681087829 | 10000 |
| 2–5 | `dy_mHat2to5GeV` | 25 | 5912.643177243027 | 10000 |
| 5–10 | `dy_mHat5to10GeV` | 25 | 458.0153183831834 | 10000 |
| 10–20 | `dy_mHat10to20GeV` | 25 | 38.46548547834682 | 10000 |
| ≥20 | `dy_mHat20toinfGeV` | 25 | 1.2761671115668445 | 10000 |

### Direct J/ψ → μμ

These bins refer to the hard-process transverse momentum `pThat`, in GeV.
This is different from the final J/ψ or muon pT. The three highest bins use
weighted generation and event sampling, which explains their small values of `D`.

| pThat [GeV] | Sample directory | Files | σ [pb] | D |
|---|---|---:|---:|---:|
| 0–1 | `jpsi_pThat0to1GeV` | 25 | 1503369.0870485106 | 10000 |
| 1–2 | `jpsi_pThat1to2GeV` | 25 | 342299.260533131 | 10000 |
| 2–5 | `jpsi_pThat2to5GeV` | 25 | 93225.77307339365 | 10000 |
| 5–10 | `jpsi_pThat5to10GeV` | 27 | 1881.374468818274 | 0.0452852508514405 |
| 10–20 | `jpsi_pThat10to20GeV` | 28 | 23.906403461436227 | 0.0009177813540578304 |
| ≥20 | `jpsi_pThat20toinfGeV` | 27 | 0.044064467556391526 | 0.0000032356386147520836 |

For expected yields, **lumi = 3 fb⁻¹ = 3000 pb⁻¹** is a reasonable first estimate. This is our assumed effective SHIFT exposure, set to 1% of the Run-3 luminosity. 

Each event then gets the weight
`lumi * (σ / D) * shiftSamplingGenWeight`, with lumi in pb⁻¹ and σ in pb.
Normalize each sample separately, then add the histograms. The framework already
reads `D` from the histogram metadata and uses it as `initial_weight_sum`,
including the small values in the weighted J/ψ bins. Their event weights are
also small, so the small denominator is expected.

The sampling correction is already included in `shiftSamplingGenWeight`;
keep its sign and do not apply it a second time. Use the full-sample `D` above,
even if you select only some events.

## Useful branches

The tree is called `Events`, with one entry per saved event. Momenta and masses are in GeV, positions in cm, and angles in radians.
The origin is at CMS, so the nominal target position is `z = 14800` cm.

The slash notation below stands for separate branches, e.g. `GenPart_vx/vy/vz`
means `GenPart_vx`, `GenPart_vy` and `GenPart_vz`.

| Branches | Description |
|---|---|
| `shiftSamplingGenWeight` | Generator weight divided by the probability of selecting the event for detector simulation. Use this weight for analysis. |
| `shiftOriginalGenWeight`, `shiftSamplingProbability`, `shiftSamplingWeight` | Original generator weight W, selection probability p, and 1/p. The analysis weight above is W/p. `genWeight` does not include this sampling correction. |
| `nGenPart`, `GenPart_pdgId`, `GenPart_status`, `GenPart_genPartIdxMother` | Generated particles, their PDG IDs, status and mother indices. Muons have `abs(pdgId)==13`; J/ψ has `pdgId==443`. A mother index of −1 means no stored mother. |
| `GenPart_pt/pz/eta/phi/mass`, `GenPart_vx/vy/vz` | Generated particle kinematics and production positions. |
| `nShiftMuon`, `ShiftMuon_pt/pz/eta/phi`, `ShiftMuon_charge` | Reconstructed muons, their kinematics and charge. |
| `ShiftMuon_vx/vy/vz`, `ShiftMuon_dxy`, `ShiftMuon_dz` | Track position near its closest approach to the target line, and impact parameters relative to the CMS origin. `dz` is not the production z. |
| `ShiftMuon_topology`, `ShiftMuon_recoAlgorithm` | Detector coverage and reconstruction algorithm; codes are listed below. |
| `ShiftMuon_nValidMuonHits`, `ShiftMuon_nMuonStations`, `ShiftMuon_normalizedChi2` | Hit counts, station counts and track-fit quality. |
| `ShiftMuon_genPartIdx` | Index of the matched generated particle, or −1 if unmatched. Use it for MC checks. |
| `nShiftDimuonVertex`, `ShiftDimuonVertex_muonIdx1/muonIdx2` | Reconstructed pairs and the indices of their two muons in the `ShiftMuon` arrays. |
| `ShiftDimuonVertex_mass/pt/pz/eta/phi`, `ShiftDimuonVertex_vx/vy/vz` | Reconstructed dimuon kinematics and common vertex. |
| `ShiftDimuonVertex_isOS`, `ShiftDimuonVertex_dcaValid`, `ShiftDimuonVertex_dca` | Opposite-sign flag and distance of closest approach. Use `dca` only when `dcaValid==1`. |
| `ShiftDimuonVertex_topologyMin/topologyMax` | Sorted topology codes of the two muons. Both-Both pairs have both codes equal to 2. |
| `ShiftMuon_constrained*`, `ShiftDimuonVertex_constrained*` | Alternative fits assuming a prompt source at the target. Use them only when the corresponding `constrainedValid==1`. |
| `ShiftDimuonVertex_refitted*`, `ShiftDimuonVertex_refitStatus` | Alternative joint vertex fit. Use its results only when `refitStatus==1`. |
| `ShiftDimuonVertex_bdtScore` | BDT score for each pair. All pairs are retained; no BDT cut has been applied. |

`ShiftMuon_topology` describes which parts of the detector recorded hits:
0 = near endcap only, 1 = near endcap and barrel, 2 = both endcaps,
3 = far endcap only, 4 = unclassified. Near and far are relative to the source. Both (code 2) is the one we should focus on.

`ShiftMuon_recoAlgorithm` identifies the reconstruction:
0 = displaced standalone (DSA), 1 = strict traversing, 2 = ordinary cosmic.
The older `quality` branch uses different codes. Typically, no need to use these flags, though - use topology instead.
