import { useState } from 'react';
import PropTypes from 'prop-types';
import { FileCode, ChevronDown, ChevronUp } from 'lucide-react';
import { CodeBlock } from './CodeBlock';
import { cn } from '@/lib/utils';

export function Citation({ citation, index }) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className="animate-fade-in" style={{ animationDelay: `${index * 50}ms` }}>
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className={cn(
                    "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-all duration-200",
                    "bg-secondary/50 hover:bg-secondary border border-transparent",
                    isExpanded && "border-primary/30 glow-primary"
                )}
            >
                <FileCode className="w-4 h-4 text-primary flex-shrink-0" />
                <span className="flex-1 text-sm font-mono truncate text-foreground">
                    {citation.filePath}
                </span>
                <span className="text-xs text-muted-foreground flex-shrink-0">
                    lines {citation.lineStart}–{citation.lineEnd}
                </span>
                {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                ) : (
                    <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                )}
            </button>

            {isExpanded && (
                <div className="mt-2 animate-fade-in">
                    <CodeBlock
                        code={citation.content}
                        language={citation.language}
                        fileName={citation.filePath}
                        lineStart={citation.lineStart}
                        lineEnd={citation.lineEnd}
                    />
                </div>
            )}
        </div>
    );
}

Citation.propTypes = {
    citation: PropTypes.shape({
        id: PropTypes.string.isRequired,
        filePath: PropTypes.string.isRequired,
        lineStart: PropTypes.number.isRequired,
        lineEnd: PropTypes.number.isRequired,
        content: PropTypes.string.isRequired,
        language: PropTypes.string.isRequired,
    }).isRequired,
    index: PropTypes.number.isRequired,
};

export function CitationsList({ citations }) {
    if (!citations || citations.length === 0) return null;

    return (
        <div className="mt-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                <span>Sources</span>
                <span className="px-1.5 py-0.5 rounded bg-secondary text-foreground">
                    {citations.length}
                </span>
            </div>
            <div className="space-y-2">
                {citations.map((citation, idx) => (
                    <Citation key={citation.id} citation={citation} index={idx} />
                ))}
            </div>
        </div>
    );
}

CitationsList.propTypes = {
    citations: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string.isRequired,
        filePath: PropTypes.string.isRequired,
        lineStart: PropTypes.number.isRequired,
        lineEnd: PropTypes.number.isRequired,
        content: PropTypes.string.isRequired,
        language: PropTypes.string.isRequired,
    })),
};
