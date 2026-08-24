function batch_preprocess()
% =========================================================================
% Project: Semantic Category Decoding in EEG Without Deep Models
% Pipeline: EEGLAB Preprocessing (0.5-45 Hz Bandpass -> CleanLine 50 Hz ->
%           CAR Rereference -> Bad Channel Interpolation -> Infomax ICA ->
%           ICLabel Artifact Removal -> CSV Export)
% =========================================================================

%% --- PATH CONFIGURATION ---
ROOT_DIR = uigetdir('', 'Select dataset directory containing subject/event folders');
if isequal(ROOT_DIR, 0), error('No directory selected.'); end

CED_FILE = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'config', 'CHANNEL_LOCS64_corrected_minimal.ced');
if ~isfile(CED_FILE)
    [f, p] = uigetfile({'*.ced;*.loc;*.locs', 'Channel location files (*.ced)'}, 'Select CED file');
    CED_FILE = fullfile(p, f);
end

%% --- PARAMETERS ---
Fs              = 2048;             % Sampling rate (Hz)
bandCutoffs     = [0.5 45];         % Bandpass range (Hz)
lineNoiseHz     = 50;               % Mains notch frequency (Hz)
doInterp        = true;             % Channel interpolation
fixedBadChans   = [28];             % Bad electrode index (Iz)
iclabelRules    = [NaN NaN; 0.9 1; 0.9 1; 0.9 1; NaN NaN; 0.9 1; NaN NaN];
skipIfCsvExists = true;

%% --- INITIALIZE EEGLAB ---
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; %#ok<ASGLU>

expectedTasks = { ...
    'animal_auditory_imagery_task', 'animal_image', 'animal_silent_naming_task', ...
    'animal_tactile_imagery_task', 'animal_visual_imagery_task', ...
    'tool_auditory_imagery_task', 'tool_image', 'tool_silent_naming_task', ...
    'tool_tactile_imagery_task', 'tool_visual_imagery_task'};

eventDirs = dir(fullfile(ROOT_DIR, '*_event*'));
eventDirs = eventDirs([eventDirs.isdir]);

for e = 1:numel(eventDirs)
    evtPath = fullfile(ROOT_DIR, eventDirs(e).name);
    fprintf('\n=== Processing Event Folder: %s ===\n', eventDirs(e).name);

    for t = 1:numel(expectedTasks)
        taskPath = fullfile(evtPath, expectedTasks{t});
        if ~isfolder(taskPath), continue; end

        mats = dir(fullfile(taskPath, '*.mat'));
        fprintf('  -> Task: %s (%d trials)\n', expectedTasks{t}, numel(mats));

        for i = 1:numel(mats)
            matFile = fullfile(taskPath, mats(i).name);
            [~, baseName, ~] = fileparts(matFile);
            outCsv = fullfile(taskPath, [baseName '.csv']);

            if skipIfCsvExists && isfile(outCsv)
                continue;
            end

            try
                ALLEEG = []; CURRENTSET = 0; %#ok<NASGU>

                % 1. Import
                EEG = pop_importdata('dataformat','matlab','nbchan',0, ...
                      'data', matFile, 'srate', Fs, 'pnts',0, 'xmin',0, ...
                      'chanlocs', CED_FILE);
                [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'gui', 'off');

                % 2. CleanLine (50 Hz)
                EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',1:EEG.nbchan,'computepower',1, ...
                      'linefreqs',lineNoiseHz,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0);
                [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'gui', 'off');

                % 3. Bandpass (0.5 - 45 Hz)
                EEG = pop_eegfiltnew(EEG, 'locutoff', bandCutoffs(1), 'hicutoff', bandCutoffs(2));
                [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 2, 'gui', 'off');

                % 4. Interpolate bad channels
                if doInterp && ~isempty(fixedBadChans)
                    EEG = pop_interp(EEG, fixedBadChans, 'invdist');
                    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 3, 'gui', 'off');
                end

                % 5. Common Average Re-referencing (CAR)
                EEG = pop_reref(EEG, []);
                [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 4, 'gui', 'off');

                % 6. Extended Infomax ICA
                EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');
                [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

                % 7. ICLabel Flagging & Artifact Rejection
                EEG = pop_iclabel(EEG, 'default');
                [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
                EEG = pop_icflag(EEG, iclabelRules);
                [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

                comp2rm = find(EEG.reject.gcompreject);
                EEG = pop_subcomp(EEG, comp2rm, 0);
                [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 5, 'gui', 'off');

                % 8. CSV Export
                pop_export(EEG, outCsv, 'transpose', 'on', 'separator', ',', 'precision', 4);
            catch ME
                fprintf(2, '    [ERROR] Failed %s: %s\n', mats(i).name, ME.message);
            end
        end
    end
end
fprintf('\nPreprocessing complete.\n');
end
