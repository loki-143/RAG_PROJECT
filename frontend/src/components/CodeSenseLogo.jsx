import PropTypes from 'prop-types';

/**
 * CodeSense Logo Component
 * 
 * A modern, minimalist logo combining:
 * - Code brackets < > representing code
 * - Brain/neural network pattern representing AI
 * - Gradient colors for modern tech feel
 */
export function CodeSenseLogo({ size = 32, className = '' }) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 100 100"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className={className}
        >
            {/* Gradient Definitions */}
            <defs>
                <linearGradient id="codeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
                <linearGradient id="brainGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#8b5cf6" />
                    <stop offset="100%" stopColor="#ec4899" />
                </linearGradient>
            </defs>

            {/* Left Bracket < */}
            <path
                d="M 35 25 L 20 50 L 35 75"
                stroke="url(#codeGradient)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
            />

            {/* Right Bracket > */}
            <path
                d="M 65 25 L 80 50 L 65 75"
                stroke="url(#codeGradient)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
            />

            {/* Neural Network Pattern (Brain) */}
            {/* Center node */}
            <circle cx="50" cy="50" r="4" fill="url(#brainGradient)" />
            
            {/* Top nodes */}
            <circle cx="45" cy="35" r="3" fill="url(#brainGradient)" opacity="0.8" />
            <circle cx="55" cy="35" r="3" fill="url(#brainGradient)" opacity="0.8" />
            
            {/* Bottom nodes */}
            <circle cx="45" cy="65" r="3" fill="url(#brainGradient)" opacity="0.8" />
            <circle cx="55" cy="65" r="3" fill="url(#brainGradient)" opacity="0.8" />
            
            {/* Connections */}
            <line x1="50" y1="50" x2="45" y2="35" stroke="url(#brainGradient)" strokeWidth="1.5" opacity="0.6" />
            <line x1="50" y1="50" x2="55" y2="35" stroke="url(#brainGradient)" strokeWidth="1.5" opacity="0.6" />
            <line x1="50" y1="50" x2="45" y2="65" stroke="url(#brainGradient)" strokeWidth="1.5" opacity="0.6" />
            <line x1="50" y1="50" x2="55" y2="65" stroke="url(#brainGradient)" strokeWidth="1.5" opacity="0.6" />
            
            {/* Cross connections */}
            <line x1="45" y1="35" x2="55" y2="65" stroke="url(#brainGradient)" strokeWidth="1" opacity="0.3" />
            <line x1="55" y1="35" x2="45" y2="65" stroke="url(#brainGradient)" strokeWidth="1" opacity="0.3" />
        </svg>
    );
}

CodeSenseLogo.propTypes = {
    size: PropTypes.number,
    className: PropTypes.string,
};

/**
 * CodeSense Logo with Text
 */
export function CodeSenseLogoWithText({ size = 'md', className = '' }) {
    const sizes = {
        sm: { logo: 24, text: 'text-sm' },
        md: { logo: 32, text: 'text-base' },
        lg: { logo: 40, text: 'text-lg' },
    };

    const { logo, text } = sizes[size] || sizes.md;

    return (
        <div className={`flex items-center gap-3 ${className}`}>
            <div className="relative">
                <CodeSenseLogo size={logo} />
                {/* Glow effect */}
                <div className="absolute inset-0 blur-xl opacity-30 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full" />
            </div>
            <div>
                <h1 className={`font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent ${text}`}>
                    CodeSense
                </h1>
                <p className="text-xs text-muted-foreground">AI Code Assistant</p>
            </div>
        </div>
    );
}

CodeSenseLogoWithText.propTypes = {
    size: PropTypes.oneOf(['sm', 'md', 'lg']),
    className: PropTypes.string,
};
