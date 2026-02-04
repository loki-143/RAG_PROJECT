import PropTypes from 'prop-types';
import { User, Bot } from 'lucide-react';
import { CitationsList } from './Citation';
import { CodeBlock } from './CodeBlock';
import { cn } from '@/lib/utils';

export function ChatMessage({ message }) {
    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';

    if (isSystem) {
        return (
            <div className="flex justify-center py-4 animate-fade-in">
                <div className="px-4 py-2 rounded-lg bg-secondary/50 text-sm text-muted-foreground">
                    {message.content}
                </div>
            </div>
        );
    }

    return (
        <div
            className={cn(
                "flex gap-3 py-4 animate-fade-in",
                isUser ? "flex-row-reverse" : "flex-row"
            )}
        >
            {/* Avatar */}
            <div
                className={cn(
                    "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center",
                    isUser ? "bg-primary/10" : "bg-secondary"
                )}
            >
                {isUser ? (
                    <User className="w-4 h-4 text-primary" />
                ) : (
                    <Bot className="w-4 h-4 text-foreground" />
                )}
            </div>

            {/* Message content */}
            <div
                className={cn(
                    "flex-1 max-w-[85%] space-y-2",
                    isUser && "flex flex-col items-end"
                )}
            >
                <div
                    className={cn(
                        "rounded-2xl px-4 py-3",
                        isUser
                            ? "bg-chat-user text-foreground rounded-tr-md"
                            : "bg-chat-assistant border border-border rounded-tl-md"
                    )}
                >
                    <MessageContent content={message.content} />
                </div>

                {!isUser && message.citations && message.citations.length > 0 && (
                    <CitationsList citations={message.citations} />
                )}
            </div>
        </div>
    );
}

ChatMessage.propTypes = {
    message: PropTypes.shape({
        id: PropTypes.string.isRequired,
        role: PropTypes.oneOf(['user', 'assistant', 'system']).isRequired,
        content: PropTypes.string.isRequired,
        citations: PropTypes.array,
        timestamp: PropTypes.instanceOf(Date),
    }).isRequired,
};

function MessageContent({ content }) {
    // Parse content for code blocks and markdown
    const parts = parseContent(content);

    return (
        <div className="space-y-3 text-sm leading-relaxed">
            {parts.map((part, idx) => {
                if (part.type === 'code') {
                    return (
                        <CodeBlock
                            key={idx}
                            code={part.content}
                            language={part.language}
                            className="my-2"
                        />
                    );
                }
                return (
                    <div key={idx} className="prose prose-sm dark:prose-invert max-w-none">
                        {renderMarkdown(part.content)}
                    </div>
                );
            })}
        </div>
    );
}

MessageContent.propTypes = {
    content: PropTypes.string.isRequired,
};

function parseContent(content) {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
        // Add text before code block
        if (match.index > lastIndex) {
            const text = content.slice(lastIndex, match.index).trim();
            if (text) {
                parts.push({ type: 'text', content: text });
            }
        }

        // Add code block
        parts.push({
            type: 'code',
            content: match[2].trim(),
            language: match[1] || 'text',
        });

        lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < content.length) {
        const text = content.slice(lastIndex).trim();
        if (text) {
            parts.push({ type: 'text', content: text });
        }
    }

    return parts.length > 0 ? parts : [{ type: 'text', content }];
}

function renderMarkdown(text) {
    // Simple markdown rendering
    const lines = text.split('\n');

    return lines.map((line, idx) => {
        // Headers
        if (line.startsWith('### ')) {
            return <h3 key={idx} className="font-semibold text-foreground mt-3 mb-1">{line.slice(4)}</h3>;
        }
        if (line.startsWith('## ')) {
            return <h2 key={idx} className="font-semibold text-foreground mt-3 mb-1">{line.slice(3)}</h2>;
        }
        if (line.startsWith('# ')) {
            return <h1 key={idx} className="font-bold text-foreground mt-3 mb-1">{line.slice(2)}</h1>;
        }

        // Bullet points
        if (line.startsWith('- ') || line.startsWith('* ')) {
            return (
                <div key={idx} className="flex gap-2">
                    <span className="text-primary">•</span>
                    <span>{renderInlineStyles(line.slice(2))}</span>
                </div>
            );
        }

        // Empty lines
        if (!line.trim()) {
            return <div key={idx} className="h-2" />;
        }

        return <p key={idx}>{renderInlineStyles(line)}</p>;
    });
}

function renderInlineStyles(text) {
    // Bold
    const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);

    return parts.map((part, idx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={idx} className="font-semibold">{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith('`') && part.endsWith('`')) {
            return (
                <code key={idx} className="px-1.5 py-0.5 rounded bg-secondary font-mono text-xs">
                    {part.slice(1, -1)}
                </code>
            );
        }
        return part;
    });
}
