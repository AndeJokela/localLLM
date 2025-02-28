// DOM elements
const chatContainer = document.getElementById('chat-container');
const promptInput = document.getElementById('prompt-input');
const sendButton = document.getElementById('send-button');
const newChatButton = document.getElementById('new-chat-button');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const maxTokensInput = document.getElementById('max-tokens');
const contextWindowSlider = document.getElementById('context-window');
const contextWindowValue = document.getElementById('context-window-value');

// Client-side chat history
let chatHistory = [];
let sessionId = localStorage.getItem('sessionId') || generateSessionId();

function generateSessionId() {
    const newId = Math.random().toString(36).substring(2, 15);
    localStorage.setItem('sessionId', newId);
    return newId;
}

// Load existing chat history if available
function loadChatHistory() {
    fetch(`/history/${sessionId}`)
        .then(response => response.json())
        .then(data => {
            if (data.history && data.history.length > 0) {
                // Set chat history directly from server
                chatHistory = data.history;
                renderChatHistory();
            } else {
                // Add welcome message if no history
                chatHistory = [];
                addMessage("How can I help you?", 'assistant');
            }
        })
        .catch(error => console.error('Error loading chat history:', error));
}

function renderChatHistory() {
    chatContainer.innerHTML = '';
    chatHistory.forEach(item => {
        // Display the message but don't save to history again
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', item.role);
        messageDiv.textContent = item.content;
        chatContainer.appendChild(messageDiv);
    });
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addMessage(text, sender, saveToHistory = true) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    if (saveToHistory) {
        chatHistory.push({
            role: sender,
            content: text
        });
    }
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('loading');
    loadingDiv.id = 'loading-indicator';
    loadingDiv.textContent = 'Generating response...';
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function hideLoading() {
    const loadingDiv = document.getElementById('loading-indicator');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

async function sendMessage() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;
    
    // Add user message to chat
    addMessage(prompt, 'user');
    promptInput.value = '';
    
    // Show loading indicator
    showLoading();
    
    // Get generation parameters
    const temperature = parseFloat(temperatureSlider.value);
    const maxTokens = parseInt(maxTokensInput.value);
    const contextWindow = parseInt(contextWindowSlider.value);
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                temperature: temperature,
                max_tokens: maxTokens,
                context_window: contextWindow,
                session_id: sessionId,
                history: chatHistory
            })
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.error) {
            addMessage(`Error: ${data.error}`, 'assistant');
        } else {
            addMessage(data.response, 'assistant');
        }
    } catch (error) {
        hideLoading();
        addMessage(`Network error: ${error.message}`, 'assistant');
    }
}

function startNewChat() {
    // Clear the chat container
    chatContainer.innerHTML = '';
    
    // Generate a new session ID
    sessionId = generateSessionId();
    
    // Clear chat history
    chatHistory = [];
    
    // Reset to server
    fetch('/new_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId
        })
    });
    
    // Add welcome message
    addMessage("How can I help you?", 'assistant');
}

// Event listeners
temperatureSlider.addEventListener('input', () => {
    temperatureValue.textContent = temperatureSlider.value;
});

contextWindowSlider.addEventListener('input', () => {
    contextWindowValue.textContent = `${contextWindowSlider.value} messages`;
});

sendButton.addEventListener('click', sendMessage);
newChatButton.addEventListener('click', startNewChat);

promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Load chat history when page loads
window.addEventListener('load', () => {
    loadChatHistory();
});