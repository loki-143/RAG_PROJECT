import PropTypes from 'prop-types';
import { GitBranch, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function EmptyState({ onAddRepo }) {
    return (
        <div className="flex-1 flex items-center justify-center p-8">
            <div className="max-w-md text-center animate-fade-in">
                {/* Abstract illustration */}
                <div className="relative mx-auto w-32 h-32 mb-8">
                    <div className="absolute inset-0 rounded-full bg-primary/5 animate-pulse-glow" />
                    <div className="absolute inset-4 rounded-full bg-primary/10" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="relative">
                            <GitBranch className="w-12 h-12 text-primary" />
                            <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-success flex items-center justify-center">
                                <span className="text-[8px] font-bold text-success-foreground">AI</span>
                            </div>
                        </div>
                    </div>
                    {/* Decorative elements */}
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-4 bg-gradient-to-b from-primary/0 to-primary/30" />
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-4 bg-gradient-to-t from-primary/0 to-primary/30" />
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-4 h-px bg-gradient-to-r from-primary/0 to-primary/30" />
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-px bg-gradient-to-l from-primary/0 to-primary/30" />
                </div>

                <h2 className="text-2xl font-semibold text-foreground mb-3">
                    Chat with your codebase
                </h2>
                <p className="text-muted-foreground mb-6 leading-relaxed">
                    Index a GitHub repository to start asking questions about your code.
                    Get intelligent answers with source-level citations.
                </p>

                <div className="flex flex-col items-center gap-3">
                    <Button
                        size="lg"
                        onClick={onAddRepo}
                        className="glow-primary-hover transition-all duration-300"
                    >
                        Index a Repository
                        <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                    <p className="text-xs text-muted-foreground">
                        Paste a GitHub URL in the sidebar to get started
                    </p>
                </div>

                {/* Feature hints */}
                <div className="mt-12 grid grid-cols-3 gap-4 text-xs text-muted-foreground">
                    <div className="flex flex-col items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                            <span className="text-lg">🔍</span>
                        </div>
                        <span>Smart Search</span>
                    </div>
                    <div className="flex flex-col items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                            <span className="text-lg">📝</span>
                        </div>
                        <span>Citations</span>
                    </div>
                    <div className="flex flex-col items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                            <span className="text-lg">⚡</span>
                        </div>
                        <span>Fast RAG</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

EmptyState.propTypes = {
    onAddRepo: PropTypes.func,
};
