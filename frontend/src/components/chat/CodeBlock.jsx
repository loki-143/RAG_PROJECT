import { useState } from 'react';
import PropTypes from 'prop-types';
import { Copy, Check, FileCode } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function CodeBlock({
    code,
    language = 'text',
    fileName,
    lineStart,
    lineEnd,
    className
}) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        await navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const lines = code.split('\n');
    const startLine = lineStart || 1;

    return (
        <div className={cn("code-block group", className)}>
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2 bg-secondary/50 border-b border-code-border">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <FileCode className="w-3.5 h-3.5" />
                    {fileName && (
                        <>
                            <span className="font-mono text-foreground">{fileName}</span>
                            {lineStart && lineEnd && (
                                <span className="text-muted-foreground">
                                    (lines {lineStart}–{lineEnd})
                                </span>
                            )}
                        </>
                    )}
                    {!fileName && language && (
                        <span className="font-mono">{language}</span>
                    )}
                </div>
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={handleCopy}
                >
                    {copied ? (
                        <Check className="w-3.5 h-3.5 text-success" />
                    ) : (
                        <Copy className="w-3.5 h-3.5 text-muted-foreground" />
                    )}
                </Button>
            </div>

            {/* Code content */}
            <div className="overflow-x-auto">
                <pre className="p-4 text-sm leading-relaxed">
                    <code className="font-mono text-code-text">
                        {lines.map((line, idx) => (
                            <div key={idx} className="flex">
                                <span className="select-none w-10 text-right pr-4 text-muted-foreground/50 text-xs">
                                    {startLine + idx}
                                </span>
                                <span className="flex-1">{line || ' '}</span>
                            </div>
                        ))}
                    </code>
                </pre>
            </div>
        </div>
    );
}

CodeBlock.propTypes = {
    code: PropTypes.string.isRequired,
    language: PropTypes.string,
    fileName: PropTypes.string,
    lineStart: PropTypes.number,
    lineEnd: PropTypes.number,
    className: PropTypes.string,
};

CodeBlock.defaultProps = {
    language: 'text',
};
