function extract_bpf_features()
% =========================================================================
% Project: Semantic Category Decoding in EEG Without Deep Models
% Description: Extracts 5,123 features per trial using Band-Pass Framing (BPF):
%   - 5 Canonical Bands: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz),
%                        Beta (13-30 Hz), Gamma (30-45 Hz)
%   - 12 Features/band/channel across 64 channels (3,840 features)
%   - N400 ERP mean amplitude over CPz/Pz/POz (1 feature)
%   - Left Temporo-Parietal (LTP) Alpha & Beta ERD (2 features)
%   - Spatiotemporal MVPA Bins across 64 channels (1,280 features)
% =========================================================================

clc; clear;

%% ---------- PATH & CONFIGURATION ----------
dataDir = uigetdir('', 'Select folder containing preprocessed CSV files');
if isequal(dataDir, 0), error('No folder selected.'); end

fs = 2048;                          % Sampling rate (Hz)
colRange = 2:65;                    % 64 EEG channels

% Canonical BPF bands
bpfBands = struct( ...
    'Delta', [0.5 4], ...
    'Theta', [4 8], ...
    'Alpha', [8 13], ...
    'Beta',  [13 30], ...
    'Gamma', [30 45]);

% ERP & ERD window definitions
erdBands      = struct('Alpha', [8 13], 'Beta', [13 30]);
baselineWin_s = [-0.20 0.00];       % Baseline window relative to cue
taskWin_s     = [0.30 0.80];        % ERD task window
n400Win_s     = [0.25 0.60];        % N400 ERP mean amplitude window
erpMVPAWin_s  = [0.20 0.70];        % Spatiotemporal ERP binning window
mvpaBins      = 20;                 % Number of temporal bins

% 64-channel 10-20 montage
channels = {'Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7', ...
            'CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz', ...
            'Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz', ...
            'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2'};

% Clusters
erpCluster = {'CPz','Pz','POz'};
LTPcluster = {'T7','FT7','TP7','P7','P5','P3','PO7','PO3','CP5','CP3'};

chIndex = containers.Map(channels, num2cell(1:numel(channels)));
erpIdx  = mapChanIdx(erpCluster, chIndex);
ltpIdx  = mapChanIdx(LTPcluster, chIndex);

%% ---------- PRECOMPUTE FILTERS ----------
bpfNames = fieldnames(bpfBands);
bpfFilt  = struct();
for b = 1:numel(bpfNames)
    fRange = bpfBands.(bpfNames{b});
    fRange(2) = min(fRange(2), (fs/2) - 1); % Guard Nyquist
    [bb, aa] = butter(2, fRange / (fs / 2));
    bpfFilt.(bpfNames{b}) = struct('b', bb, 'a', aa);
end

erdFilt  = struct();
erdNames = fieldnames(erdBands);
for e = 1:numel(erdNames)
    [bb, aa] = butter(2, erdBands.(erdNames{e}) / (fs / 2));
    erdFilt.(erdNames{e}) = struct('b', bb, 'a', aa);
end

%% ---------- PREPARE 5,123 FEATURE NAMES ----------
timeFeats = {'Mean', 'Std', 'Skewness', 'Kurtosis'};
freqFeats = {'BandPower_BPF', 'LogBandPower_BPF', 'Energy_BPF'};
newFeats  = {'Variance', 'Activity', 'Mobility', 'RelBandPower', 'SpectralEntropy'};

allFeatNames = {};
for ch = 1:numel(channels)
    for b = 1:numel(bpfNames)
        base = sprintf('%s_%s_', channels{ch}, bpfNames{b});
        allFeatNames = [allFeatNames, ...
            strcat(base, timeFeats), ...
            strcat(base, freqFeats), ...
            strcat(base, newFeats)]; %#ok<AGROW>
    end
end

% Append ERP, ERD, and MVPA feature names
allFeatNames{end+1} = 'ERP_N400_CPzPzPOz_MeanAmp';
allFeatNames{end+1} = 'ERD_Alpha_LTP_pct';
allFeatNames{end+1} = 'ERD_Beta_LTP_pct';
for b = 1:mvpaBins
    for ch = 1:numel(channels)
        allFeatNames{end+1} = sprintf('ERPST_bin%02d_%s', b, channels{ch}); %#ok<AGROW>
    end
end

%% ---------- PROCESS EACH FILE ----------
files = [dir(fullfile(dataDir, '*.csv')); dir(fullfile(dataDir, '*.CSV'))];
% Filter out already generated feature files
files = files(~contains({files.name}, '_feature.csv'));
if isempty(files), error('No preprocessed CSV files found in %s', dataDir); end

fprintf('Extracting BPF features from %d files...\n', numel(files));

for k = 1:numel(files)
    inPath = fullfile(dataDir, files(k).name);
    
    % Label inference: 0 = Animal, 1 = Tool
    lowerName = lower(files(k).name);
    if contains(lowerName, 'animal'),     label = 0;
    elseif contains(lowerName, 'tool'),   label = 1;
    else, label = -1;
    end

    M = readmatrix(inPath);
    Y = double(M(:, colRange))'; % [64 x N]
    [nCh, N] = size(Y);

    baseIdx = win2idx(baselineWin_s, fs, N);
    taskIdx = win2idx(taskWin_s,     fs, N);
    n400Idx = win2idx(n400Win_s,     fs, N);
    mvpaIdx = win2idx(erpMVPAWin_s,  fs, N);

    featVec = zeros(1, numel(allFeatNames));
    idx = 1;

    % ===== 1. Per-Channel x Per-Band BPF Feature Extraction =====
    for ch = 1:nCh
        sig = Y(ch, :);

        % Zero-phase filtering across 5 bands
        bandSig = cell(1, numel(bpfNames));
        for b = 1:numel(bpfNames)
            bName = bpfNames{b};
            bandSig{b} = filtfilt(bpfFilt.(bName).b, bpfFilt.(bName).a, sig);
        end

        % Compute total power across all 5 bands for relative power calculation
        bp = cellfun(@(x) mean(x.^2), bandSig);
        totPow = sum(bp) + eps;

        for b = 1:numel(bandSig)
            x = bandSig{b};

            % Time-domain statistics
            m  = mean(x);
            s  = std(x, 0);
            sk = skewness(x, 0);
            ku = kurtosis(x, 0);

            % Power & Energy
            bp_val = mean(x.^2);
            log_bp = log(bp_val + eps);
            energy = sum(x.^2);

            % Hjorth parameters & Entropy
            variance = var(x, 1);
            activity = variance;
            mobility = sqrt(var(diff(x), 1) / (variance + eps));
            rel_bp   = bp_val / totPow;

            % Normalized Spectral / Signal Entropy within band
            p = x.^2;
            p = p / (sum(p) + eps);
            spectralEntropy = -sum(p .* log2(p + eps));

            vals = [m, s, sk, ku, bp_val, log_bp, energy, ...
                    variance, activity, mobility, rel_bp, spectralEntropy];

            featVec(idx:idx+numel(vals)-1) = vals;
            idx = idx + numel(vals);
        end
    end

    % ===== 2. ERP N400 Mean Amplitude (CPz / Pz / POz) =====
    if ~isempty(n400Idx)
        erpSig = baselineCorrect(Y, baseIdx, fs);
        featVec(idx) = mean(mean(erpSig(erpIdx, n400Idx), 1));
    else
        featVec(idx) = NaN;
    end
    idx = idx + 1;

    % ===== 3. ERD Alpha & Beta over Left Temporo-Parietal Cluster =====
    [erdA, erdB] = computeERD(Y, erdFilt, ltpIdx, baseIdx, taskIdx);
    featVec(idx) = erdA; idx = idx + 1;
    featVec(idx) = erdB; idx = idx + 1;

    % ===== 4. Spatiotemporal ERP Pattern Bins (MVPA) =====
    if ~isempty(mvpaIdx)
        erpSig = baselineCorrect(Y, baseIdx, fs);
        binEdges = round(linspace(mvpaIdx(1), mvpaIdx(end)+1, mvpaBins+1));
        for b = 1:mvpaBins
            seg = erpSig(:, binEdges(b):binEdges(b+1)-1);
            featVec(idx:idx+64-1) = mean(seg, 2)';
            idx = idx + 64;
        end
    else
        featVec(idx:idx+(mvpaBins*64)-1) = NaN;
        idx = idx + (mvpaBins*64);
    end

    % ===== 5. Save Single Row Feature Table =====
    [~, base, ~] = fileparts(files(k).name);
    outTable = [table(string(files(k).name), label, 'VariableNames', {'File', 'Label'}), ...
                array2table(featVec, 'VariableNames', matlab.lang.makeValidName(allFeatNames))];
    
    outPath = fullfile(dataDir, sprintf('%s_bpf_feature.csv', base));
    writetable(outTable, outPath);
    fprintf('  [%3d/%3d] Saved: %s\n', k, numel(files), outPath);
end

fprintf('\nAll %d-dimensional BPF feature vectors extracted.\n', numel(allFeatNames));
end

%% ---------- HELPER FUNCTIONS ----------
function idxs = mapChanIdx(list, chIndex)
idxs = [];
for i = 1:numel(list)
    if isKey(chIndex, list{i}), idxs(end+1) = chIndex(list{i}); end %#ok<AGROW>
end
end

function ind = win2idx(win_s, fs, N)
s1 = max(0, win_s(1)); s2 = max(0, win_s(2));
i1 = min(N, max(1, round(s1*fs)+1));
i2 = min(N, max(i1, round(s2*fs)));
if i2 <= i1, ind = []; else, ind = i1:i2; end
end

function Ybc = baselineCorrect(Y, baseIdx, fs)
if isempty(baseIdx), baseIdx = 1:min(round(0.2*fs), size(Y,2)); end
Ybc = Y - mean(Y(:, baseIdx), 2);
end

function [erdAlphaPct, erdBetaPct] = computeERD(Y, erdFilt, ltpIdx, baseIdx, taskIdx)
if isempty(ltpIdx) || isempty(baseIdx) || isempty(taskIdx)
    erdAlphaPct = NaN; erdBetaPct = NaN; return;
end
Ya = filtfilt(erdFilt.Alpha.b, erdFilt.Alpha.a, Y')';
Yb = filtfilt(erdFilt.Beta.b,   erdFilt.Beta.a,   Y')';
Pa = abs(hilbert(Ya(ltpIdx,:).').').^2;
Pb = abs(hilbert(Yb(ltpIdx,:).').').^2;

baseA = mean(Pa(:, baseIdx), 2) + eps; taskA = mean(Pa(:, taskIdx), 2);
baseB = mean(Pb(:, baseIdx), 2) + eps; taskB = mean(Pb(:, taskIdx), 2);

erdAlphaPct = mean((taskA - baseA) ./ baseA * 100);
erdBetaPct  = mean((taskB - baseB) ./ baseB * 100);
end
