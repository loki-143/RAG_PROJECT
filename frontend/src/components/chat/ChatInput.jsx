import { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function ChatInput({ onSend, disabled, isLoading }) {
    const [message, setMessage] = useState('');
    const textareaRef = useRef(null);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [message]);

    const handleSubmit = () => {
        if (message.trim() && !disabled && !isLoading) {
            onSend(message.trim());
            setMessage('');
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="border-t border-border bg-background/80 backdrop-blur-sm p-4">
            <div
                className={cn(
                    "flex items-end gap-2 rounded-xl border bg-card p-2 transition-all duration-200",
                    disabled
                        ? "border-border opacity-50"
                        : "border-border hover:border-primary/50 focus-within:border-primary focus-within:glow-primary"
                )}
            >
                <textarea
                    ref={textareaRef}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={disabled ? "Index a repository to start chatting..." : "Ask how this code works…"}
                    disabled={disabled}
                    rows={1}
                    className={cn(
                        "flex-1 resize-none bg-transparent px-3 py-2 text-sm",
                        "placeholder:text-muted-foreground",
                        "focus:outline-none",
                        "scrollbar-thin",
                        "min-h-[40px] max-h-[200px]"
                    )}
                />
                <Button
                    onClick={handleSubmit}
                    disabled={!message.trim() || disabled || isLoading}
                    size="icon"
                    className="h-9 w-9 flex-shrink-0"
                >
                    {isLoading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                        <Send className="w-4 h-4" />
                    )}
                </Button>
            </div>
            <p className="text-[10px] text-muted-foreground mt-2 text-center">
                Press Enter to send, Shift+Enter for new line
            </p>
        </div>
    );
}

ChatInput.propTypes = {
    onSend: PropTypes.func.isRequired,
    disabled: PropTypes.bool,
    isLoading: PropTypes.bool,
};

ChatInput.defaultProps = {
    disabled: false,
    isLoading: false,
};
