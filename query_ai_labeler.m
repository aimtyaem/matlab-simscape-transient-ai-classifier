function aiLabel = query_ai_labeler(jsonPayload, modelName)
% QUERY_AI_LABELER  Call Anthropic Claude API to classify a transient regime.
%   aiLabel = query_ai_labeler(jsonPayload) uses the default model
%   'claude-fable-5' (Claude Fable 5 – fast, cost‑effective LRM).
%   aiLabel = query_ai_labeler(jsonPayload, modelName) lets you specify
%   a different model, e.g., 'claude-opus-4-20250514'.
%
%   Returns a struct with fields:
%       regime_classification   (string)
%       safety_margin_pct        (float)
%       confidence_score         (float 0..1)

    arguments
        jsonPayload char
        modelName char = 'claude-fable-5'   % default: fast Claude Fable 5
    end

    % Anthropic API configuration
    apiUrl = 'https://api.anthropic.com/v1/messages';
    apiKey = getenv('ANTHROPIC_API_KEY');   % Set in environment

    headers = [
        httpHeader('Content-Type', 'application/json')
        httpHeader('x-api-key', apiKey)
        httpHeader('anthropic-version', '2023-06-01')
    ];

    % System prompt: strict expert labeling, JSON‑only output
    systemPrompt = [ ...
        'You are an expert AI Data Labeling agent specialized in Computational Fluid Dynamics ' ...
        'and transient hydraulic simulations. You receive a JSON payload with physical parameters ' ...
        '(conduit length, fluid density, valve close time, delta pressure, max pressure, deceleration rate). ' ...
        'You must output ONLY a valid JSON object, with no other text. The JSON must contain: ' ...
        '"regime_classification": string ("Safe Operation", "Water Hammer Surge", or "Cavitation Risk"), ' ...
        '"safety_margin_pct": float, ' ...
        '"confidence_score": float (between 0.0 and 1.0).' ...
    ];

    % Build request
    userMessage = struct('role', 'user', 'content', jsonPayload);
    requestBody = struct(...
        'model', modelName, ...
        'max_tokens', 256, ...
        'system', systemPrompt, ...
        'messages', {{userMessage}} ...
    );

    options = weboptions('HeaderFields', headers, ...
                         'MediaType', 'application/json', ...
                         'Timeout', 30);

    try
        response = webwrite(apiUrl, requestBody, options);
        % Claude returns content as a cell of structs with 'text' field
        contentText = response.content{1}.text;
        aiLabel = jsondecode(contentText);
    catch ME
        warning('Claude API call failed. Falling back to manual review.');
        aiLabel = struct('regime_classification', 'Unknown', ...
                         'safety_margin_pct', 0.0, ...
                         'confidence_score', 0.0);
    end
end