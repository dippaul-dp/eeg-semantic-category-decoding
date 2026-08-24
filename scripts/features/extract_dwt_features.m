function extract_dwt_features()
% =========================================================================
% Extracts multiresolution Discrete Wavelet Transform (MODWT db4) features,
% N400 ERP mean amplitude, Left Temporo-Parietal ERD, and spatiotemporal
% MVPA time bins across 64 channels (Total: 5,123 features per trial).
% =========================================================================

clc; clear;

dataDir = uigetdir('', 'Select folder containing preprocessed CSV files');
if isequal(dataDir, 0), error('No folder selected.'); end

fs = 2048;
colRange = 2:65;

% ERP/ERD Window Configurations
erdBands       = struct('Alpha', [8 13], 'Beta', [13 30]);
baselineWin_s  = [-0.20 0.00];
taskWin_s      = [0.30 0.80];
n400Win_s      = [0.25 0.60];
erpMVPAWin_s   = [0.20 0.70];
mvpaBins       = 20;

channels = {'Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7', ...
            'CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz', ...
            'Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz', ...
            'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2'};

erpCluster = {'CPz','Pz','POz'};
LTPcluster = {'T7','FT7','TP7','P7','P5','P3','PO7','PO3','CP5','CP3'};

chIndex = containers.Map(channels, num2cell(1:numel(channels)));
erpIdx  = mapChanIdx(erpCluster, chIndex);
ltpIdx  = mapChanIdx(LTPcluster, chIndex);

% Filter setup for ERD
erdFilt = struct();
erdNames = fieldnames(erdBands);
for e = 1:numel(erdNames)
    [bb, aa] = butter(2, erdBands.(erdNames{e}) / (fs / 2));
    erdFilt.(erdNames{e}) = struct('b', bb, 'a', aa);
end

% Feature Names Initialization
timeFeats = {'Mean','Std','Skewness','Kurtosis'};
freqFeats = {'BandPower_DWT','LogBandPower_DWT','Energy_DWT'};
newFeats  = {'Variance','Activity','Mobility','RelBandPower','WaveletEntropy'};
dwtBands  = {'Gamma','Beta','Alpha','Theta','Delta'};

allFeatNames = {};
for ch = 1:numel(channels)
    for b = 1:numel(dwtBands)
        base = sprintf('%s_%s_', channels{ch}, dwtBands{b});
        allFeatNames = [allFeatNames, strcat(base, timeFeats), strcat(base, freqFeats), strcat(base, newFeats)]; %#ok<AGROW>
    end
end

allFeatNames{end+1} = 'ERP_N400_CPzPzPOz_MeanAmp';
allFeatNames{end+1} = 'ERD_Alpha_LTP_pct';
allFeatNames{end+1} = 'ERD_Beta_LTP_pct';
for b = 1:mvpaBins
    for ch = 1:numel(channels)
        allFeatNames{end+1} = sprintf('ERPST_bin%02d_%s', b, channels{ch}); %#ok<AGROW>
    end
end

files = dir(fullfile(dataDir, '*.csv'));
fprintf('Extracting features from %d files...\n', numel(files));

for k = 1:numel(files)
    inPath = fullfile(dataDir, files(k).name);
    label = contains(lower(files(k).name), 'tool'); % 0 = Animal, 1 = Tool

    M = readmatrix(inPath);
    Y = double(M(:, colRange))'; % [64 x N]
    [nCh, N] = size(Y);

    baseIdx = win2idx(baselineWin_s, fs, N);
    taskIdx = win2idx(taskWin_s, fs, N);
    n400Idx = win2idx(n400Win_s, fs, N);
    mvpaIdx = win2idx(erpMVPAWin_s, fs, N);

    featVec = zeros(1, numel(allFeatNames));
    idx = 1;

    % DWT per-channel multiresolution decomposition
    for ch = 1:nCh
        sig = Y(ch, :);
        lvl = min(8, wmaxlev(numel(sig), 'db4'));
        [C, L] = wavedec(sig, lvl, 'db4');

        gamma = wrcoef('d', C, L, 'db4', 5);
        beta  = wrcoef('d', C, L, 'db4', 6);
        alpha = wrcoef('d', C, L, 'db4', 7);
        theta = wrcoef('d', C, L, 'db4', 8);
        delta = wrcoef('a', C, L, 'db4', min(lvl, 8));

        bandSig = {gamma, beta, alpha, theta, delta};
        bp = cellfun(@(x) mean(x.^2), bandSig);
        totPow = sum(bp) + eps;

        for b = 1:numel(bandSig)
            x = bandSig{b};
            m = mean(x); s = std(x, 0); sk = skewness(x, 0); ku = kurtosis(x, 0);
            bp_dwt = mean(x.^2); log_bp = log(bp_dwt + eps); energy = sum(x.^2);
            variance = var(x, 1); activity = variance;
            mobility = sqrt(var(diff(x), 1) / (variance + eps));
            rel_bp = bp_dwt / totPow;
            p = x.^2; p = p / (sum(p) + eps);
            waveletEntropy = -sum(p .* log2(p + eps));

            vals = [m, s, sk, ku, bp_dwt, log_bp, energy, variance, activity, mobility, rel_bp, waveletEntropy];
            featVec(idx:idx+numel(vals)-1) = vals;
            idx = idx + numel(vals);
        end
    end

    % ERP N400 Feature
    erpSig = baselineCorrect(Y, baseIdx, fs);
    featVec(idx) = mean(mean(erpSig(erpIdx, n400Idx), 1));
    idx = idx + 1;

    % ERD Alpha/Beta Features
    [erdA, erdB] = computeERD(Y, erdFilt, ltpIdx, baseIdx, taskIdx);
    featVec(idx) = erdA; idx = idx + 1;
    featVec(idx) = erdB; idx = idx + 1;

    % Spatiotemporal MVPA Bins
    binEdges = round(linspace(mvpaIdx(1), mvpaIdx(end)+1, mvpaBins+1));
    for b = 1:mvpaBins
        seg = erpSig(:, binEdges(b):binEdges(b+1)-1);
        featVec(idx:idx+64-1) = mean(seg, 2)';
        idx = idx + 64;
    end

    % Export single row CSV
    [~, base] = fileparts(files(k).name);
    outT = [table(string(files(k).name), label, 'VariableNames', {'File','Label'}), ...
            array2table(featVec, 'VariableNames', matlab.lang.makeValidName(allFeatNames))];
    writetable(outT, fullfile(dataDir, sprintf('%s_feature.csv', base)));
end
fprintf('All feature vectors extracted successfully.\n');
end

% Helper Functions
function idxs = mapChanIdx(list, chMap)
idxs = cellfun(@(c) chMap(c), list);
end

function ind = win2idx(win_s, fs, N)
s1 = max(0, win_s(1)); s2 = max(0, win_s(2));
i1 = min(N, max(1, round(s1*fs)+1));
i2 = min(N, max(i1, round(s2*fs)));
ind = i1:i2;
end

function Ybc = baselineCorrect(Y, baseIdx, fs)
if isempty(baseIdx), baseIdx = 1:min(round(0.2*fs), size(Y,2)); end
Ybc = Y - mean(Y(:, baseIdx), 2);
end

function [erdA, erdB] = computeERD(Y, erdFilt, ltpIdx, baseIdx, taskIdx)
Ya = filtfilt(erdFilt.Alpha.b, erdFilt.Alpha.a, Y')';
Yb = filtfilt(erdFilt.Beta.b, erdFilt.Beta.a, Y')';
Pa = abs(hilbert(Ya(ltpIdx,:)')').^2;
Pb = abs(hilbert(Yb(ltpIdx,:)')').^2;
baseA = mean(Pa(:, baseIdx), 2) + eps; taskA = mean(Pa(:, taskIdx), 2);
baseB = mean(Pb(:, baseIdx), 2) + eps; taskB = mean(Pb(:, taskIdx), 2);
erdA = mean((taskA - baseA) ./ baseA * 100);
erdB = mean((taskB - baseB) ./ baseB * 100);
end
