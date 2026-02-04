import { MessageSquare, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAppStore } from '@/stores/appStore';

export function ChatHeader() {
    const { repositories, activeRepoIds } = useAppStore();

    const activeRepos = repositories.filter(r => activeRepoIds.includes(r.id));

    if (activeRepos.length === 0) {
        return null;
    }

    return (
        <header className="border-b border-border bg-card/50 backdrop-blur-sm px-6 py-3">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                        <MessageSquare className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                        <p className="text-sm text-muted-foreground">Chatting with</p>
                        <div className="flex items-center gap-2 flex-wrap">
                            {activeRepos.map((repo, idx) => (
                                <span key={repo.id}>
                                    <span className="text-sm font-medium text-foreground">
                                        {repo.name}
                                    </span>
                                    {idx < activeRepos.length - 1 && (
                                        <span className="text-muted-foreground">, </span>
                                    )}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>

                <Button variant="ghost" size="icon" className="text-muted-foreground">
                    <Settings className="w-4 h-4" />
                </Button>
            </div>
        </header>
    );
}
