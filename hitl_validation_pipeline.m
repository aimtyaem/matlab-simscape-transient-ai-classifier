function run_hitl_validation_pipeline()
    outDir = fullfile(pwd, 'output');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    ts = datestr(datetime('now','TimeZone','UTC'), 'yyyymmdd_HHMMSS');
    matFile = fullfile(outDir, ['sensor_archive_' ts '.mat']);
    manualJsonl = fullfile(outDir, 'fine_tuning_dataset.jsonl');
    autoJsonl = fullfile(outDir, 'auto_labeled_dataset.jsonl');

    try
        % Run the Simscape simulation script – it creates:
        %   time, pressure, flowRate, deltaP, payload (JSON string)
        run('simulate_and_extract_features.mlx');

        % Verify required variables exist
        requiredVars = {'time','pressure','flowRate','deltaP','payload'};
        for i = 1:numel(requiredVars)
            if ~exist(requiredVars{i}, 'var')
                error('Required variable "%s" not found after simulation.', requiredVars{i});
            end
        end

        % Build a consistent 'sensors' structure from available data
        dt = median(diff(time));
        if isempty(dt) || dt == 0
            dt = 0.001;  % fallback
        end
        sampleRate = 1/dt;

        sensors.pressure    = pressure(:);
        sensors.flow        = flowRate(:);
        % Dummy fields for temperature and vibration – replace with real sensors if available
        sensors.temperature = 20 * ones(size(time(:)));   % deg C
        sensors.vibration   = zeros(size(time(:)));       % g
        sensors.sampleRate  = sampleRate;

        % Build enhanced payload with derived features and anomaly flags
        [payloadStruct, payloadJSON, derived_features, anomaly_flags] = ...
            build_payload_with_sensor_features(payload, time, sensors);

        % Archive the raw sensor data and features
        save_sensor_archive(matFile, time, sensors, deltaP, payloadStruct, derived_features, anomaly_flags);

        % AI labelling with escalation
        aiResult = query_with_escalation(payloadJSON);

        threshold = 0.80;
        userCorrectionRequired = aiResult.confidence_score < threshold || any(struct2array(anomaly_flags));

        if userCorrectionRequired
            fprintf('\n[WARNING] Review required. Confidence: %.2f\n', aiResult.confidence_score);
            correctedLabel = show_hitl_portal(time, sensors, deltaP);

            ft_record = build_finetune_record(payloadStruct, correctedLabel, time, sensors, derived_features, anomaly_flags, aiResult);
            append_jsonl_record(manualJsonl, ft_record);

            fprintf('Manual record appended to %s\n', manualJsonl);
        else
            fprintf('\nAI auto-labeled simulation: "%s" (confidence: %.2f)\n', ...
                aiResult.regime_classification, aiResult.confidence_score);

            auto_record = struct( ...
                'predicted_label', aiResult.regime_classification, ...
                'confidence', aiResult.confidence_score, ...
                'payload', payloadStruct, ...
                'derived_features', derived_features, ...
                'anomaly_flags', anomaly_flags, ...
                'meta', struct('timestamp_utc', char(datetime('now','TimeZone','UTC')), 'source', 'simulated', 'mat_file', matFile) ...
            );

            append_jsonl_record(autoJsonl, auto_record);
            fprintf('Auto-labeled record appended to %s\n', autoJsonl);
        end

    catch ME
        fprintf(2, '\n[ERROR] Pipeline failed: %s\n', ME.message);
        rethrow(ME);
    end
end

%% ---- Helper Functions ----

function [payloadStruct, payloadJSON, derived_features, anomaly_flags] = build_payload_with_sensor_features(payload, time, sensors)
    % Unpack payload if it is a JSON string
    if isstruct(payload)
        payloadStruct = payload;
    else
        try
            payloadStruct = jsondecode(payload);
        catch
            payloadStruct = struct('raw_payload', payload);
        end
    end

    pressureVec = sensors.pressure(:);
    tempVec     = sensors.temperature(:);
    flowVec     = sensors.flow(:);
    vibVec      = sensors.vibration(:);

    derived_features = struct( ...
        'pressure_mean',              mean(pressureVec), ...
        'pressure_std',               std(pressureVec), ...
        'pressure_peak',              max(pressureVec), ...
        'pressure_min',               min(pressureVec), ...
        'vibration_rms',              rms(vibVec), ...
        'vibration_peak',             max(abs(vibVec)), ...
        'temperature_mean',           mean(tempVec), ...
        'temperature_gradient_mean',  mean(gradient(tempVec, time(:))), ...
        'flow_mean',                  mean(flowVec), ...
        'flow_std',                   std(flowVec) ...
    );

    anomaly_flags = struct( ...
        'pressure',    any(isoutlier(pressureVec)), ...
        'temperature', any(isoutlier(tempVec)), ...
        'flow',        any(isoutlier(flowVec)), ...
        'vibration',   any(isoutlier(vibVec)) ...
    );

    payloadStruct.sensors_summary = struct( ...
        'names',            {{'pressure','temperature','flow','vibration'}}, ...
        'sample_rate_Hz',   sensors.sampleRate, ...
        'num_samples',      numel(time), ...
        'units',            struct('pressure','Pa','temperature','C','flow','m^3/s','vibration','g'), ...
        'derived_features', derived_features, ...
        'anomaly_flags',    anomaly_flags ...
    );

    payloadJSON = jsonencode(payloadStruct);
end

function finalResult = query_with_escalation(jsonPayload)
    fableResult = query_ai_labeler(jsonPayload, 'claude-fable-5');
    fprintf('Fable 5 classification: "%s" (confidence: %.2f)\n', ...
        fableResult.regime_classification, fableResult.confidence_score);

    if fableResult.confidence_score < 0.80
        fprintf('Confidence below 80%%. Escalating to Claude Opus...\n');
        finalResult = query_ai_labeler(jsonPayload, 'claude-opus-4-20250514');
    else
        finalResult = fableResult;
    end
end

function correctedLabel = show_hitl_portal(time, sensors, deltaP)
    figure('Name', 'HITL Signal Verification Portal');
    tiledlayout(2,2, 'Padding','compact', 'TileSpacing','compact');

    nexttile;
    plot(time, sensors.pressure, 'LineWidth', 2, 'Color', '#D95319');
    grid on; title(sprintf('Pressure (\\DeltaP = %.1f Pa)', deltaP));
    xlabel('Time (s)'); ylabel('Pressure (Pa)');

    nexttile;
    plot(time, sensors.temperature, 'LineWidth', 2, 'Color', '#0072BD');
    grid on; title('Temperature');
    xlabel('Time (s)'); ylabel('Temperature (C)');

    nexttile;
    plot(time, sensors.flow, 'LineWidth', 2, 'Color', '#77AC30');
    grid on; title('Flow');
    xlabel('Time (s)'); ylabel('Flow (m^3/s)');

    nexttile;
    plot(time, sensors.vibration, 'LineWidth', 2, 'Color', '#A2142F');
    grid on; title('Vibration');
    xlabel('Time (s)'); ylabel('Acceleration (g)');

    disp('Select the correct physical regime:');
    disp('1: Safe Operation');
    disp('2: Water Hammer Surge');
    disp('3: Cavitation Risk');
    choice = input('Enter index (1-3): ');

    regimes = {'Safe Operation', 'Water Hammer Surge', 'Cavitation Risk'};
    if ~ismember(choice, 1:numel(regimes))
        error('Invalid selection.');
    end

    correctedLabel = regimes{choice};
end

function ft_record = build_finetune_record(payloadStruct, correctedLabel, time, sensors, derived_features, anomaly_flags, aiResult)
    maxSamples = 500;
    nSamples = numel(time);
    idx = 1:nSamples;
    if nSamples > maxSamples
        idx = round(linspace(1, nSamples, maxSamples));
    end

    sensor_record = struct( ...
        'time',              time(idx)', ...
        'pressure',          sensors.pressure(idx)', ...
        'temperature',       sensors.temperature(idx)', ...
        'flow',              sensors.flow(idx)', ...
        'vibration',         sensors.vibration(idx)', ...
        'sample_rate_Hz',    sensors.sampleRate, ...
        'derived_features',  derived_features, ...
        'anomaly_flags',     anomaly_flags ...
    );

    ft_record = struct( ...
        'messages', {{ ...
            struct('role', 'system', 'content', 'You are an expert engineering data labeler.'), ...
            struct('role', 'user', 'content', payloadStruct), ...
            struct('role', 'assistant', 'content', struct('regime_classification', correctedLabel)) ...
        }}, ...
        'sensors', sensor_record, ...
        'aiResult', aiResult, ...
        'meta', struct('timestamp_utc', char(datetime('now','TimeZone','UTC')), 'source', 'simulated') ...
    );
end

function append_jsonl_record(filename, recordStruct)
    fid = fopen(filename, 'a');
    if fid == -1
        error('Cannot open %s for writing.', filename);
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(recordStruct));
end

function save_sensor_archive(filename, time, sensors, deltaP, payloadStruct, derived_features, anomaly_flags)
    archive = struct();
    archive.time = time;
    archive.sensors = sensors;
    archive.deltaP = deltaP;
    archive.payloadStruct = payloadStruct;
    archive.derived_features = derived_features;
    archive.anomaly_flags = anomaly_flags;
    archive.created_at_utc = char(datetime('now','TimeZone','UTC'));

    save(filename, '-struct', 'archive');
end
