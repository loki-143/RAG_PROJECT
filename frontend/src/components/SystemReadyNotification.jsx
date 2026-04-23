import { useEffect, useState } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, CheckCircle2 } from 'lucide-react';

// Use Nginx reverse proxy - no need for env variable
const API_URL = '/api';

export const SystemReadyNotification = () => {
    const [systemState, setSystemState] = useState({
        ready: false,
        checking: true,
        error: null,
    });

    useEffect(() => {
        let mounted = true;
        let checkInterval;

        const checkSystemReady = async () => {
            try {
                const response = await fetch(`${API_URL}/health`);
                const data = await response.json();

                if (mounted) {
                    if (data.ready) {
                        setSystemState({
                            ready: true,
                            checking: false,
                            error: null,
                        });
                        // Stop checking once ready
                        if (checkInterval) {
                            clearInterval(checkInterval);
                        }
                    } else {
                        setSystemState({
                            ready: false,
                            checking: true,
                            error: null,
                        });
                    }
                }
            } catch (error) {
                if (mounted) {
                    setSystemState({
                        ready: false,
                        checking: true,
                        error: 'Unable to connect to backend',
                    });
                }
            }
        };

        // Initial check
        checkSystemReady();

        // Poll every 2 seconds until ready
        checkInterval = setInterval(checkSystemReady, 2000);

        return () => {
            mounted = false;
            if (checkInterval) {
                clearInterval(checkInterval);
            }
        };
    }, []);

    // Don't show anything if system is ready
    if (systemState.ready) {
        return null;
    }

    // Show error state
    if (systemState.error) {
        return (
            <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-md px-4">
                <Alert variant="destructive">
                    <AlertDescription className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        {systemState.error}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    // Show loading state
    return (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-md px-4">
            <Alert className="bg-blue-50 border-blue-200 dark:bg-blue-950 dark:border-blue-800">
                <AlertDescription className="flex items-center gap-2 text-blue-900 dark:text-blue-100">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading embedding model... Please wait
                </AlertDescription>
            </Alert>
        </div>
    );
};
