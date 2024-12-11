const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
document.head.appendChild(script);

const apiUrl = '/api/tags';
const chatUrl = '/api/chat';
const modelSelect = document.getElementById('model');
const questionInput = document.getElementById('question');
const streamToggle = document.getElementById('streamToggle');
const responseDiv = document.getElementById('response');
const metricsDiv = document.getElementById('metrics');
const timerDiv = document.getElementById('timer');


// Allow input box to expand
document.getElementById('question').addEventListener('input', function () {
    this.style.height = 'auto'; // Reset height
    this.style.height = `${this.scrollHeight}px`; // Adjust to content
});

// Fetch models from /api/tags and populate dropdown
async function loadModels() {
    try {
        const response = await fetch(apiUrl);
        const data = await response.json();
        const models = data.models;

        modelSelect.innerHTML = '<option value="" disabled selected>Select a model</option>';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        });
    } catch (error) {
        modelSelect.innerHTML = '<option value="" disabled>Error loading models</option>';
        console.error('Error fetching models:', error);
    }
}


// Submit question and handle responses
async function submitQuestion() {
    const model = modelSelect.value;
    const question = questionInput.value;
    const streamEnabled = streamToggle.checked;

    if (!model || !question) {
        alert('Please select a model and enter a question.');
        return;
    }

    responseDiv.textContent = '';
    metricsDiv.textContent = '';
    timerDiv.textContent = '';
    const startTime = performance.now();

    const requestData = {
        model: model,
        stream: streamEnabled,
        messages: [
            {
                role: 'user',
                content: question
            }
        ]
    };

    if (streamEnabled) {
        await fetchStreamedResponse(requestData, startTime);
    } else {
        await fetchNonStreamedResponse(requestData, startTime);
    }
}

// Fetch non-streamed response
async function fetchNonStreamedResponse(requestData, startTime) {
    try {
        const response = await fetch(chatUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();
        const endTime = performance.now();
        const elapsed = ((endTime - startTime) / 1000).toFixed(2);

        // Convert durations from nanoseconds to seconds
        const promptEvalDurationSec = (result.prompt_eval_duration / 1e9).toFixed(3) || 'N/A';
        const evalDurationSec = (result.eval_duration / 1e9).toFixed(3) || 'N/A';
        const loadDurationSec = (result.load_duration / 1e9).toFixed(3) || 'N/A';
        const totalDurationSec = (result.total_duration / 1e9).toFixed(3) || 'N/A';

        // Ensure proper formatting of the response content
        const formattedContent = formatContent(result.message.content || 'No meaningful response received.');

        // Display response
        responseDiv.innerHTML = `
            <div>${formattedContent}</div>
        `;
        metricsDiv.innerHTML = `
            <ul>
                <li>Response Time: ${elapsed} seconds</li>
                <li>Prompt Eval Count: ${result.prompt_eval_count || 'N/A'}</li>
                <li>Prompt Eval Duration: ${promptEvalDurationSec} seconds</li>
                <li>Eval Count: ${result.eval_count || 'N/A'}</li>
                <li>Eval Duration: ${evalDurationSec} seconds</li>
                <li>Eval Speed: ${result.eval_count ? (result.eval_count / evalDurationSec).toFixed(3) : 'N/A'} tokens/sec</li>
                <li>Load Duration: ${loadDurationSec} seconds</li>
                <li>Total Duration: ${totalDurationSec} seconds</li>
            </ul>
        `;
    } catch (error) {
        responseDiv.textContent = 'Error during chat.';
        timerDiv.textContent = '';
        console.error('Error:', error);
    }
}

function formatContent(content) {
    // Check if the response includes Markdown
    const isMarkdown = content.includes('```') || content.includes('*') || content.includes('_');

    if (isMarkdown) {
        // Use a Markdown parser for formatted responses
        return marked.parse(content); // Requires marked.js or another library
    }

    // For plain text, replace line breaks with <br> for proper rendering
    return content.replace(/\n/g, '<br>');
}


async function fetchStreamedResponse(requestData, startTime) {
    try {
        const response = await fetch(chatUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let resultText = '';
        let finalMetrics = {};

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });

            // Parse each JSON line and append content
            const lines = chunk.split('\n').filter(line => line.trim());
            lines.forEach(line => {
                try {
                    const json = JSON.parse(line);
                    if (json.message && json.message.content) {
                        resultText += json.message.content;
                        responseDiv.textContent = resultText;
                    }
                    if (json.done === true) {
                        finalMetrics = {
                            prompt_eval_count: json.prompt_eval_count || 'N/A',
                            prompt_eval_duration: (json.prompt_eval_duration / 1e9).toFixed(3) || 'N/A',
                            eval_count: json.eval_count || 'N/A',
                            eval_duration: (json.eval_duration / 1e9).toFixed(3) || 'N/A',
                            load_duration: (json.load_duration / 1e9).toFixed(3) || 'N/A',
                            total_duration: (json.total_duration / 1e9).toFixed(3) || 'N/A',
                        };
                    }
                } catch (error) {
                    console.error('Error parsing streamed chunk:', error);
                }
            });
        }

        const endTime = performance.now();
        const elapsed = ((endTime - startTime) / 1000).toFixed(2);

        const formattedContent = formatContent(resultText);

        // Display final response with metrics
        responseDiv.innerHTML = `
             ${formattedContent}</p>
        `;

        metricsDiv.innerHTML = `
            <ul>
                <li>Response Time: ${elapsed} seconds</li>
                <li>Prompt Eval Count: ${finalMetrics.prompt_eval_count}</li>
                <li>Prompt Eval Duration: ${finalMetrics.prompt_eval_duration} seconds</li>
                <li>Eval Count: ${finalMetrics.eval_count}</li>
                <li>Eval Duration: ${finalMetrics.eval_duration} seconds</li>
                <li>Eval Speed: ${finalMetrics.eval_count ? (finalMetrics.eval_count / finalMetrics.eval_duration).toFixed(3) : 'N/A'} tokens/sec</li>
                <li>Load Duration: ${finalMetrics.load_duration} seconds</li>
                <li>Total Duration: ${finalMetrics.total_duration} seconds</li>
            </ul>
        `;
    } catch (error) {
        responseDiv.textContent = 'Error during streamed chat.';
        timerDiv.textContent = '';
        console.error('Error:', error);
    }
}

// Load models on page load
document.addEventListener('DOMContentLoaded', loadModels);
