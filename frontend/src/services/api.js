// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_RAG_API_KEY || '';

function getHeaders() {
    const headers = {
        'Content-Type': 'application/json',
    };

    if (API_KEY) {
        headers['X-API-Key'] = API_KEY;
    }

    return headers;
}

// Health check
export async function healthCheck() {
    const response = await fetch(`${API_BASE_URL}/`, {
        method: 'GET',
        headers: getHeaders(),
    });

    if (!response.ok) {
        throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
}

// List all indexed repositories
export async function listIndexes() {
    const response = await fetch(`${API_BASE_URL}/indexes`, {
        method: 'GET',
        headers: getHeaders(),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Failed to list indexes');
    }

    return response.json();
}

// Index a repository
export async function indexRepository(repoUrl, force = false) {
    const response = await fetch(`${API_BASE_URL}/index`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ repo_url: repoUrl, force }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Failed to index repository');
    }

    return response.json();
}

// Ask a question (single query without history)
export async function askQuestion(question, repos, topK = 8) {
    const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({
            question,
            repos: repos || null,
            top_k: topK,
            use_history: true,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Failed to get answer');
    }

    return response.json();
}

// Chat with context (uses /chat endpoint)
export async function chat(question, repos, topK = 8) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({
            question,
            repos,
            top_k: topK,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Failed to get response');
    }

    return response.json();
}

// Delete an index (uses DELETE method to /index)
export async function deleteIndex(repoUrl) {
    const response = await fetch(`${API_BASE_URL}/index`, {
        method: 'DELETE',
        headers: getHeaders(),
        body: JSON.stringify({ repo_url: repoUrl }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Failed to delete index');
    }

    return response.json();
}

// Get repository stats
export async function getStats(repoUrl) {
    const url = new URL(`${API_BASE_URL}/stats`);
    url.searchParams.append('repo_url', repoUrl);

    const response = await fetch(url.toString(), {
        method: 'GET',
        headers: getHeaders(),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Failed to get stats');
    }

    return response.json();
}
