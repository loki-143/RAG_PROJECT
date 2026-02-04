import { useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { ChatHeader } from './ChatHeader';
import { EmptyState } from './EmptyState';
import { useAppStore } from '@/stores/appStore';
import { useChat } from '@/hooks/useApi';

export function ChatArea() {
    const messagesEndRef = useRef(null);
    const {
        repositories,
        activeRepoIds,
        messages,
        isLoading,
        indexingProgress,
    } = useAppStore();

    const { sendMessage } = useChat();

    const hasActiveRepos = activeRepoIds.length > 0;
    const hasIndexedRepos = repositories.some(r => r.status === 'indexed');
    const isIndexing = indexingProgress !== null;

    // Scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async (content) => {
        await sendMessage(content);
    };

    // Show empty state if no indexed repos
    if (!hasIndexedRepos && !isIndexing) {
        return (
            <div className="h-full flex flex-col bg-background overflow-hidden">
                <EmptyState />
            </div>
        );
    }

    // Show indexing state
    if (isIndexing && !hasActiveRepos) {
        return (
            <div className="h-full flex flex-col bg-background overflow-hidden">
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center animate-fade-in">
                        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-warning/10 flex items-center justify-center animate-pulse">
                            <div className="w-8 h-8 rounded-full border-2 border-warning border-t-transparent animate-spin" />
                        </div>
                        <h3 className="text-lg font-medium text-foreground mb-2">Indexing repository...</h3>
                        <p className="text-sm text-muted-foreground">
                            Please wait while we process your codebase
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    // Show prompt to select repos
    if (!hasActiveRepos && hasIndexedRepos) {
        return (
            <div className="h-full flex flex-col bg-background overflow-hidden">
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center animate-fade-in">
                        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                            <span className="text-2xl">📂</span>
                        </div>
                        <h3 className="text-lg font-medium text-foreground mb-2">Select a repository</h3>
                        <p className="text-sm text-muted-foreground">
                            Check a repository in the sidebar to start chatting
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col bg-background overflow-hidden">
            <ChatHeader />

            {/* Messages area */}
            <div className="flex-1 overflow-y-auto px-6 scrollbar-thin">
                {messages.length === 0 ? (
                    <div className="h-full flex items-center justify-center">
                        <div className="text-center animate-fade-in">
                            <p className="text-muted-foreground">
                                Ask a question about your codebase
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="max-w-3xl mx-auto py-4">
                        {messages.map((message) => (
                            <ChatMessage key={message.id} message={message} />
                        ))}
                        {isLoading && (
                            <div className="flex gap-3 py-4 animate-fade-in">
                                <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                                    <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                                </div>
                                <div className="flex items-center">
                                    <div className="flex gap-1">
                                        <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '0ms' }} />
                                        <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '150ms' }} />
                                        <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '300ms' }} />
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            <ChatInput
                onSend={handleSend}
                disabled={!hasActiveRepos || isIndexing}
                isLoading={isLoading}
            />
        </div>
    );
}
