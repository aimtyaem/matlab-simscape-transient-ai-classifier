# simscape-llm-hitl-labeler
An automated, physics-grounded pipeline for Simscape transient fluid simulation classification. Pairs MATLAB programmatic solvers with a Large Reasoning Model (LRM) labeling agent and a Human-in-the-Loop (HITL) validation interface to generate verified machine learning datasets for physics-aligned model fine-tuning.
## Project Overview
In engineering design-space exploration, analyzing thousands of parametric simulation sweeps to identify physical anomalies—such as cavitation, thermal choking, or water hammer surges—is a manually-intensive task.
This repository provides an automated, multi-agent AI framework integrated directly into MATLAB. The system runs transient Simscape pipeline simulations, extracts critical telemetry, and utilizes a generative AI model to automatically classify the flow regimes. If the AI model's confidence falls below a set safety threshold, the pipeline automatically triggers a Human-in-the-Loop (HITL) graphical interface. The expert's corrections are captured and logged to continuously fine-tune and align the AI's physical intuition.
## Mathematical & Physical Foundations
The simulation specifically targets transient fluid inertia (water hammer surges) in a hydraulic pipeline. When a downstream valve closes abruptly, the sudden deceleration of the fluid column creates a massive pressure spike.
### Code-Style Representation
In the MATLAB pipeline, this transient surge is calculated programmatically using the following variables:
• ΔP: The inertial pressure rise
• ρ: The fluid density
• L: The conduit length
• A: The cross-sectional area
• dQ/dt: The rate of change of the volumetric flow rate

```matlab
% Hydraulic momentum relation calculated programmatically:
delta_P = rho * (L / A) * (dQ_dt);

```
### Formal Physical Model
The underlying physical phenomenon is governed by the hydraulic momentum equation:
## AI Agent Taxonomy: Large Reasoning Models (LRMs)
To ensure strict compliance with physical conservation laws, this framework rejects vanilla autoregressive transformers or Small Language Models (SLMs) in favor of **Large Reasoning Models (LRMs)**.
As mapped in the agentic architecture taxonomy, LRMs meet retrieval-augmented generation (RAG) by integrating deep internal reasoning paths with external physical constraints. Unlike standard models that rely on superficial text-matching, frontier LRMs—such as **Claude Fable 5** and **Claude Opus 4.7/4.8**—employ adaptive reasoning and "max effort" thinking protocols. This allows the agent to systematically verify that the transient inputs (ΔP, dQ/dt) are mathematically consistent with the fluid density and conduit geometry before outputting a classification label, virtually eliminating unphysical code and parameter hallucinations.
## Key Architectural Features
 1. **Programmatic Simscape Interface:** Automatically configures pipeline dimensions and valve boundary parameters, runs the solver, and extracts high-frequency pressure and flow rate sensors.
 2. **Telemetry Compression & Feature Extraction:** Compresses raw time-series data into high-level physical features (peak pressure, deceleration rate, and surge time) to reduce LLM token usage and prevent context window saturation.
 3. **LRM Labeling Agent:** Calls an Anthropic messages endpoint targeting the state-of-the-art claude-fable-5 or claude-opus-4.8 model with a structured JSON payload to classify the run as a "Safe Operation", "Water Hammer Surge", or "Cavitation Risk".
 4. **Deterministic Confidence Gate:** Monitors the LRM's reported confidence score. If it falls below the safety threshold of 0.80, the automated process pauses and initiates the HITL interface.
 5. **Fine-Tuning Dataset Generator:** Saves human-corrected labels into a training-ready JSON Lines (.jsonl) dataset format to support Supervised Fine-Tuning (SFT).
## Getting Started
### 1. Feature Extraction & Simulation Run
Execute the simulation and extract transient telemetry features:
```matlab
% programmatic_sim_runner.m
% Execute Simscape models and extract key transient features for AI labeling

modelName = 'SimscapeWaterHammer';
load_system(modelName);

% Set parametric sweep variable: Valve Closing Time (seconds)
valveCloseTime = 0.05; 
set_param([modelName '/ControlValve'], 'Value', num2str(valveCloseTime));

% Run Simscape Simulation
simOut = sim(modelName, 'StopTime', '2.0');

% Extract sensory telemetry
time = simOut.tout;
pressure = simOut.logsout.get('pressure_sensor').Values.Data; % Pa
flowRate = simOut.logsout.get('flow_sensor').Values.Data;      % m^3/s

% Calculate mathematical transient features
peakPressure = max(pressure);
initialPressure = pressure(1);
deltaP = peakPressure - initialPressure;

% Calculate dQ/dt (deceleration rate)
dQ = flowRate(end) - flowRate(1);
dt = valveCloseTime;
dQ_dt = dQ / dt;

% Compile structural JSON metadata payload
simulationMetadata = struct(...
    'conduit_length_m', 10.0,...
    'fluid_density_kg_m3', 1000.0,...
    'valve_close_time_s', valveCloseTime,...
    'delta_pressure_Pa', deltaP,...
    'max_pressure_Pa', peakPressure,...
    'deceleration_rate_m3_s2', dQ_dt...
);

payload = jsonencode(simulationMetadata);
disp('Feature extraction payload ready:');
disp(payload);

```
### 2. Query the LRM Labeler
Transmit the structured telemetry payload to the LRM agent using the Anthropic Messages API:
```matlab
% queryAILabeler.m
function aiLabel = queryAILabeler(jsonPayload)
    apiUrl = 'https://api.anthropic.com/v1/messages';
    apiKey = getenv('ANTHROPIC_API_KEY'); 
    
    headers =;

    systemPrompt =;

    requestBody = struct(...
        'model', 'claude-fable-5',... % Targets the leading 2026 LRM model
        'max_tokens', 1024,...
        'system', systemPrompt,...
        'messages', {{...
            struct('role', 'user', 'content', jsonPayload)...
        }}...
    );

    options = weboptions('HeaderFields', headers, 'MediaType', 'application/json', 'Timeout', 30);
    response = webwrite(apiUrl, requestBody, options);
    
    % Decodes structured text from Anthropic''s response envelope
    aiLabel = jsondecode(response.content{1}.text);
end

```
### 3. Human-in-the-Loop Validation Pipeline
Execute the automated gate and record manual human corrections:
```matlab
% hitl_validation_pipeline.m

% Run simulation and gather features
run('programmatic_sim_runner.m');

% Send feature payload to AI Labeling Agent
aiResult = queryAILabeler(payload);

% Confidence check
threshold = 0.80;
userCorrectionRequired = false;

if aiResult.confidence_score < threshold
    fprintf('\n AI Confidence Low (%.2f). Launching Human review...\n', aiResult.confidence_score);
    userCorrectionRequired = true;
else
    fprintf('\nAI auto-labeled simulation: "%s" (Confidence: %.2f)\n',...
        aiResult.regime_classification, aiResult.confidence_score);
end

if userCorrectionRequired
    % Plot transient curve for human validation
    figure('Name', 'HITL Signal Verification Portal');
    plot(time, pressure, 'LineWidth', 2, 'Color', '#D95319');
    grid on;
    title();
    xlabel('Time (s)');
    ylabel('Pressure (Pa)');
    
    % Request manual label override
    disp('Select the correct physical regime:');
    disp('1: Safe Operation');
    disp('2: Water Hammer Surge');
    disp('3: Cavitation Risk');
    choice = input('Enter index (1-3): ');
    
    regimes = {'Safe Operation', 'Water Hammer Surge', 'Cavitation Risk'};
    correctedLabel = regimes{choice};
    
    % Structure SFT training log record
    ft_record = struct(...
        'messages', {{...
            struct('role', 'system', 'content', 'You are an expert engineering data labeler.'),...
            struct('role', 'user', 'content', payload),...
            struct('role', 'assistant', 'content', sprintf('{"regime_classification": "%s"}', correctedLabel))...
        }}...
    );
    
    % Append to JSON Lines fine-tuning dataset
    fid = fopen('fine_tuning_dataset.jsonl', 'a');
    fprintf(fid, '%s\n', jsonencode(ft_record));
    fclose(fid);
    
    fprintf('Data point manually validated and logged to fine_tuning_dataset.jsonl\n');
end

```
## Training Dataset Schema (.jsonl)
The human-in-the-loop overrides generate training pairs formatted for Supervised Fine-Tuning:
```json
{"messages": [{"role": "system", "content": "You are an expert engineering data labeler."}, {"role": "user", "content": "{\"conduit_length_m\":10,\"fluid_density_kg_m3\":1000,\"valve_close_time_s\":0.05,\"delta_pressure_Pa\":1420500,\"max_pressure_Pa\":1520500,\"deceleration_rate_m3_s2\":-0.08}"}, {"role": "assistant", "content": "{\"regime_classification\": \"Water Hammer Surge\"}"}]}

```
