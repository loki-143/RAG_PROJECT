import { create } from 'zustand';

export const useAppStore = create((set) => ({
    repositories: [],
    activeRepoIds: [],
    indexingProgress: null,
    messages: [],
    isLoading: false,
    theme: 'dark',

    addRepository: (repo) =>
        set((state) => ({
            repositories: [...state.repositories, repo],
            activeRepoIds: [...state.activeRepoIds, repo.id],
        })),

    setRepositories: (repos) =>
        set({ repositories: repos }),

    removeRepository: (id) =>
        set((state) => ({
            repositories: state.repositories.filter((r) => r.id !== id),
            activeRepoIds: state.activeRepoIds.filter((rid) => rid !== id),
        })),

    toggleRepoActive: (id) =>
        set((state) => ({
            activeRepoIds: state.activeRepoIds.includes(id)
                ? state.activeRepoIds.filter((rid) => rid !== id)
                : [...state.activeRepoIds, id],
        })),

    setIndexingProgress: (progress) =>
        set({ indexingProgress: progress }),

    updateRepoStatus: (id, status, chunkCount) =>
        set((state) => ({
            repositories: state.repositories.map((r) =>
                r.id === id
                    ? { ...r, status, chunkCount: chunkCount ?? r.chunkCount, indexedAt: status === 'indexed' ? new Date() : r.indexedAt }
                    : r
            ),
        })),

    addMessage: (message) =>
        set((state) => ({
            messages: [...state.messages, message],
        })),

    clearMessages: () =>
        set({ messages: [] }),

    setIsLoading: (loading) =>
        set({ isLoading: loading }),

    toggleTheme: () =>
        set((state) => {
            const newTheme = state.theme === 'dark' ? 'light' : 'dark';
            document.documentElement.classList.toggle('dark', newTheme === 'dark');
            return { theme: newTheme };
        }),
}));
