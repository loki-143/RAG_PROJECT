import { useState, useCallback } from 'react';
import { useToast } from '@/hooks/use-toast';
import * as api from '@/services/api';
import { useAppStore } from '@/stores/appStore';

export function useIndexRepository() {
    const [isIndexing, setIsIndexing] = useState(false);
    const { toast } = useToast();
    const { addRepository, updateRepoStatus, setIndexingProgress } = useAppStore();

    const indexRepo = useCallback(async (repoUrl) => {
        if (!repoUrl.trim()) return;

        // Extract repo name from URL
        const urlParts = repoUrl.split('/');
        const repoName = urlParts.slice(-2).join('/').replace('.git', '');

        const newRepo = {
            id: crypto.randomUUID(),
            name: repoName || 'unknown/repo',
            url: repoUrl,
            status: 'indexing',
            isActive: false,
            chunkCount: 0,
        };

        addRepository(newRepo);
        setIsIndexing(true);
        setIndexingProgress({
            repoId: newRepo.id,
            progress: 10,
            chunksProcessed: 0,
            status: 'processing',
        });

        try {
            const result = await api.indexRepository(repoUrl);

            // Backend returns { status: "indexed", meta: {...} }
            if (result.status === 'indexed') {
                const chunkCount = result.meta?.chunk_count || result.meta?.total_chunks || 0;
                updateRepoStatus(newRepo.id, 'indexed', chunkCount);
                toast({
                    title: 'Repository indexed',
                    description: `${repoName} has been indexed successfully.`,
                });
            } else {
                throw new Error(result.message || 'Indexing failed');
            }
        } catch (error) {
            updateRepoStatus(newRepo.id, 'error');
            toast({
                title: 'Indexing failed',
                description: error instanceof Error ? error.message : 'Unknown error occurred',
                variant: 'destructive',
            });
        } finally {
            setIsIndexing(false);
            setIndexingProgress(null);
        }
    }, [addRepository, updateRepoStatus, setIndexingProgress, toast]);

    return { indexRepo, isIndexing };
}

export function useDeleteRepository() {
    const { toast } = useToast();
    const { repositories, removeRepository } = useAppStore();

    const deleteRepo = useCallback(async (repoId) => {
        const repo = repositories.find(r => r.id === repoId);
        if (!repo) return;

        try {
            await api.deleteIndex(repo.url);
            removeRepository(repoId);
            toast({
                title: 'Repository deleted',
                description: `${repo.name} has been removed.`,
            });
        } catch (error) {
            toast({
                title: 'Delete failed',
                description: error instanceof Error ? error.message : 'Unknown error occurred',
                variant: 'destructive',
            });
        }
    }, [repositories, removeRepository, toast]);

    return { deleteRepo };
}

export function useChat() {
    const [isLoading, setIsLoading] = useState(false);
    const { toast } = useToast();
    const { repositories, activeRepoIds, addMessage, setIsLoading: setStoreLoading } = useAppStore();

    const sendMessage = useCallback(async (content) => {
        const activeRepos = repositories
            .filter(r => activeRepoIds.includes(r.id) && r.status === 'indexed')
            .map(r => r.url);

        if (activeRepos.length === 0) {
            toast({
                title: 'No active repositories',
                description: 'Please select at least one indexed repository.',
                variant: 'destructive',
            });
            return;
        }

        // Add user message
        const userMessage = {
            id: crypto.randomUUID(),
            role: 'user',
            content,
            timestamp: new Date(),
        };
        addMessage(userMessage);
        setIsLoading(true);
        setStoreLoading(true);

        try {
            // Call chat API with question and repos
            const result = await api.chat(content, activeRepos);

            // Transform citations to our format
            // Backend returns { answer: "...", citations: [...] }
            const citations = result.citations?.map((c, index) => ({
                id: `${Date.now()}-${index}`,
                filePath: c.file_path || c.filePath || '',
                lineStart: c.line_start || c.lineStart || 1,
                lineEnd: c.line_end || c.lineEnd || 1,
                content: c.content || '',
                language: c.language || detectLanguage(c.file_path || c.filePath || ''),
            })) || [];

            const assistantMessage = {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: result.answer || result.response || 'No response received.',
                citations: citations.length > 0 ? citations : undefined,
                timestamp: new Date(),
            };
            addMessage(assistantMessage);
        } catch (error) {
            toast({
                title: 'Failed to get response',
                description: error instanceof Error ? error.message : 'Unknown error occurred',
                variant: 'destructive',
            });

            // Add error message to chat
            addMessage({
                id: crypto.randomUUID(),
                role: 'assistant',
                content: 'Sorry, I encountered an error while processing your question. Please try again.',
                timestamp: new Date(),
            });
        } finally {
            setIsLoading(false);
            setStoreLoading(false);
        }
    }, [repositories, activeRepoIds, addMessage, setStoreLoading, toast]);

    return { sendMessage, isLoading };
}

// Helper to detect language from file path
function detectLanguage(filePath) {
    if (!filePath) return 'plaintext';
    const ext = filePath.split('.').pop()?.toLowerCase();
    const languageMap = {
        ts: 'typescript',
        tsx: 'typescript',
        js: 'javascript',
        jsx: 'javascript',
        py: 'python',
        rb: 'ruby',
        go: 'go',
        rs: 'rust',
        java: 'java',
        cpp: 'cpp',
        c: 'c',
        cs: 'csharp',
        php: 'php',
        swift: 'swift',
        kt: 'kotlin',
        scala: 'scala',
        vue: 'vue',
        svelte: 'svelte',
        html: 'html',
        css: 'css',
        scss: 'scss',
        json: 'json',
        yaml: 'yaml',
        yml: 'yaml',
        md: 'markdown',
        sql: 'sql',
        sh: 'bash',
        bash: 'bash',
        zsh: 'bash',
    };
    return languageMap[ext || ''] || 'plaintext';
}
