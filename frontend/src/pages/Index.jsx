import { useEffect } from 'react';
import { AppSidebar } from '@/components/layout/AppSidebar';
import { ChatArea } from '@/components/chat/ChatArea';
import { useAppStore } from '@/stores/appStore';
import * as api from '@/services/api';

const Index = () => {
    const { repositories, setRepositories } = useAppStore();

    // Load indexed repositories from backend on mount
    useEffect(() => {
        const loadRepositories = async () => {
            try {
                const result = await api.listIndexes();
                console.log('Loaded indexes from backend:', result);

                if (result.indexes && result.indexes.length > 0) {
                    // Backend returns array of objects: { repo_url, repo_hash, chunk_count, ... }
                    const repos = result.indexes.map((indexData, index) => ({
                        id: `loaded-${index}-${Date.now()}`,
                        name: extractRepoName(indexData.repo_url || indexData),
                        url: indexData.repo_url || indexData,
                        status: 'indexed',
                        isActive: false,
                        chunkCount: indexData.chunk_count || 0,
                        indexedAt: indexData.indexed_at || null,
                    }));
                    setRepositories(repos);
                    console.log('Loaded repositories:', repos);
                }
            } catch (error) {
                console.error('Failed to load repositories:', error);
            }
        };

        // Always try to load on mount
        loadRepositories();
    }, [setRepositories]);

    return (
        <div className="flex h-screen w-full bg-background overflow-hidden">
            <AppSidebar />
            <main className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
                <ChatArea />
            </main>
        </div>
    );
};

// Helper to extract repo name from URL
function extractRepoName(url) {
    const parts = url.replace(/\.git$/, '').split('/');
    return parts.slice(-2).join('/') || 'unknown/repo';
}

export default Index;

