function extract_bpf_features(csvFilePath, fs)
% =========================================================================
% Computes 4th-Order zero-phase Butterworth Band-Pass Framing (BPF) across
% five canonical EEG bands for all 64 channels.
% =========================================================================

if nargin < 2, fs = 2048; end

dataRaw = importdata(csvFilePath);
signalMatrix = dataRaw.data(:, 2:65)'; % 64 Channels x N Samples
[nChannels, nSamples] = size(signalMatrix);

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};
out = struct();

for ch = 1:nChannels
    sig = signalMatrix(ch, :);

    % Bandpass filter designs
    [b1, a1] = butter(2, [0.5 4] / (fs / 2));
    out(ch).Delta = filtfilt(b1, a1, sig);

    [b2, a2] = butter(2, [4 8] / (fs / 2));
    out(ch).Theta = filtfilt(b2, a2, sig);

    [b3, a3] = butter(2, [8 13] / (fs / 2));
    out(ch).Alpha = filtfilt(b3, a3, sig);

    [b4, a4] = butter(2, [13 30] / (fs / 2));
    out(ch).Beta  = filtfilt(b4, a4, sig);

    [b5, a5] = butter(2, [30 min(45, (fs/2)-5)] / (fs / 2));
    out(ch).Gamma = filtfilt(b5, a5, sig);
end

fprintf('BPF decomposition completed for %d channels across 5 bands.\n', nChannels);
end
