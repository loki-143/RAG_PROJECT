import { useState } from 'react';
import PropTypes from 'prop-types';
import { GitBranch, Plus, Trash2, Moon, Sun, MessageSquare, Github, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { useAppStore } from '@/stores/appStore';
import { useIndexRepository, useDeleteRepository } from '@/hooks/useApi';
import { cn } from '@/lib/utils';

export function AppSidebar() {
    const [repoUrl, setRepoUrl] = useState('');
    const {
        repositories,
        activeRepoIds,
        indexingProgress,
        theme,
        toggleRepoActive,
        clearMessages,
        toggleTheme,
    } = useAppStore();

    const { indexRepo, isIndexing: isApiIndexing } = useIndexRepository();
    const { deleteRepo } = useDeleteRepository();

    const handleIndexRepo = async () => {
        if (!repoUrl.trim()) return;
        setRepoUrl('');
        await indexRepo(repoUrl);
    };

    const isIndexing = indexingProgress !== null || isApiIndexing;

    return (
        <aside className="flex flex-col h-full w-72 min-w-[288px] flex-shrink-0 border-r border-border bg-sidebar overflow-hidden">
            {/* Branding */}
            <div className="p-4 border-b border-border">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center glow-primary">
                        <GitBranch className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                        <h1 className="font-semibold text-foreground tracking-tight">CodeLens AI</h1>
                        <p className="text-xs text-muted-foreground">Chat with your codebase</p>
                    </div>
                </div>
            </div>

            {/* Repository Panel */}
            <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
                <div className="space-y-2">
                    <h2 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        Indexed Repositories
                    </h2>

                    {repositories.length === 0 ? (
                        <p className="text-sm text-muted-foreground py-4 text-center">
                            No repositories indexed yet
                        </p>
                    ) : (
                        <div className="space-y-2">
                            {repositories.map((repo) => (
                                <div
                                    key={repo.id}
                                    className={cn(
                                        "group p-3 rounded-lg border transition-all duration-200",
                                        "bg-card hover:bg-surface-hover",
                                        activeRepoIds.includes(repo.id)
                                            ? "border-primary/50 glow-primary"
                                            : "border-border"
                                    )}
                                >
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="flex items-center gap-2 flex-1 min-w-0">
                                            <Checkbox
                                                id={repo.id}
                                                checked={activeRepoIds.includes(repo.id)}
                                                onCheckedChange={() => toggleRepoActive(repo.id)}
                                                disabled={repo.status !== 'indexed'}
                                                className="mt-0.5"
                                            />
                                            <div className="flex-1 min-w-0">
                                                <label
                                                    htmlFor={repo.id}
                                                    className="text-sm font-medium truncate block cursor-pointer text-foreground"
                                                >
                                                    {repo.name}
                                                </label>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <StatusBadge status={repo.status} />
                                                    {repo.status === 'indexed' && repo.chunkCount && (
                                                        <span className="text-xs text-muted-foreground">
                                                            {repo.chunkCount} chunks
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive"
                                            onClick={() => deleteRepo(repo.id)}
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </Button>
                                    </div>

                                    {indexingProgress?.repoId === repo.id && (
                                        <div className="mt-3 space-y-1.5">
                                            <Progress value={indexingProgress.progress} className="h-1.5" />
                                            <p className="text-xs text-muted-foreground">
                                                {indexingProgress.chunksProcessed} chunks processed
                                            </p>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Add Repository */}
                <div className="space-y-2 pt-2">
                    <h2 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        Add Repository
                    </h2>
                    <div className="space-y-2">
                        <Input
                            placeholder="https://github.com/user/repo"
                            value={repoUrl}
                            onChange={(e) => setRepoUrl(e.target.value)}
                            disabled={isIndexing}
                            className="bg-background border-border text-sm"
                            onKeyDown={(e) => e.key === 'Enter' && handleIndexRepo()}
                        />
                        <Button
                            onClick={handleIndexRepo}
                            disabled={!repoUrl.trim() || isIndexing}
                            className="w-full"
                            size="sm"
                        >
                            <Plus className="w-4 h-4 mr-2" />
                            Index Repository
                        </Button>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-border space-y-2">
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearMessages}
                    className="w-full justify-start text-muted-foreground hover:text-foreground"
                >
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Clear Chat
                </Button>
                <div className="flex items-center justify-between">
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={toggleTheme}
                        className="text-muted-foreground hover:text-foreground"
                    >
                        {theme === 'dark' ? (
                            <Sun className="w-4 h-4" />
                        ) : (
                            <Moon className="w-4 h-4" />
                        )}
                    </Button>
                    <a
                        href="https://github.com"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                        <Github className="w-4 h-4" />
                        <ExternalLink className="w-3 h-3" />
                    </a>
                </div>
            </div>
        </aside>
    );
}

function StatusBadge({ status }) {
    const styles = {
        indexed: 'bg-success/10 text-success border-success/20',
        indexing: 'bg-warning/10 text-warning border-warning/20 animate-pulse',
        error: 'bg-destructive/10 text-destructive border-destructive/20',
    };

    const labels = {
        indexed: 'Indexed',
        indexing: 'Indexing…',
        error: 'Error',
    };

    return (
        <span className={cn(
            "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium border",
            styles[status]
        )}>
            {status === 'indexed' && <span className="w-1.5 h-1.5 rounded-full bg-success mr-1" />}
            {status === 'indexing' && <span className="w-1.5 h-1.5 rounded-full bg-warning mr-1" />}
            {status === 'error' && <span className="w-1.5 h-1.5 rounded-full bg-destructive mr-1" />}
            {labels[status]}
        </span>
    );
}

StatusBadge.propTypes = {
    status: PropTypes.oneOf(['indexed', 'indexing', 'error']).isRequired,
};
