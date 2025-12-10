// SongGeneration Studio - Custom Hooks

// ============ Hover Hook ============
var useHover = () => {
    const [isHovered, setIsHovered] = useState(false);
    const handlers = {
        onMouseEnter: () => setIsHovered(true),
        onMouseLeave: () => setIsHovered(false),
    };
    return [isHovered, handlers];
};

// ============ Models Hook ============
var useModels = () => {
    const [models, setModels] = useState([]);
    const [allModels, setAllModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('songgeneration_base');
    const [hasReadyModel, setHasReadyModel] = useState(false);
    const [recommendedModel, setRecommendedModel] = useState(null);
    const [isInitializing, setIsInitializing] = useState(true);
    const [downloadPolling, setDownloadPolling] = useState(false);
    const [autoDownloadStarting, setAutoDownloadStarting] = useState(false);
    const autoDownloadTriggeredRef = useRef(false);

    const loadModels = useCallback(async (triggerAutoDownload = false) => {
        try {
            const r = await fetch('/api/models');
            if (!r.ok) throw new Error(`Failed to load models: ${r.status}`);
            const d = await r.json();
            
            const all = d.models || [];
            const ready = d.ready_models || all.filter(m => m.status === 'ready');

            setAllModels(all);
            setModels(ready);
            setHasReadyModel(d.has_ready_model);
            setRecommendedModel(d.recommended);
            if (d.default) setSelectedModel(d.default);

            const hasDownloading = all.some(m => m.status === 'downloading');
            setDownloadPolling(hasDownloading);

            // Auto-download recommended model on first launch
            if (triggerAutoDownload && !d.has_ready_model && d.recommended && 
                !autoDownloadTriggeredRef.current && !hasDownloading) {
                autoDownloadTriggeredRef.current = true;
                setAutoDownloadStarting(true);
                try {
                    const downloadRes = await fetch(`/api/models/${d.recommended}/download`, { method: 'POST' });
                    if (downloadRes.ok) {
                        setDownloadPolling(true);
                        loadModels();
                    }
                } catch (e) {
                    console.error('[AUTO-DOWNLOAD] Error:', e);
                    autoDownloadTriggeredRef.current = false;
                }
                setAutoDownloadStarting(false);
            }
            setIsInitializing(false);
        } catch (e) {
            console.error('[MODELS] Error:', e);
            setIsInitializing(false);
        }
    }, []);

    const startDownload = useCallback(async (modelId) => {
        try {
            const r = await fetch(`/api/models/${modelId}/download`, { method: 'POST' });
            if (r.ok) {
                setDownloadPolling(true);
                loadModels();
            }
        } catch (e) { console.error(e); }
    }, [loadModels]);

    const cancelDownload = useCallback(async (modelId) => {
        try {
            await fetch(`/api/models/${modelId}/download`, { method: 'DELETE' });
            setAllModels(prev => {
                const updated = prev.map(m => 
                    m.id === modelId ? { ...m, status: 'not_downloaded', progress: 0 } : m
                );
                if (!updated.some(m => m.status === 'downloading')) {
                    setDownloadPolling(false);
                }
                return updated;
            });
            await loadModels();
        } catch (e) { console.error(e); }
    }, [loadModels]);

    const deleteModelHandler = useCallback(async (modelId) => {
        if (!confirm(`Delete model ${modelId}? You'll need to download it again to use it.`)) return;
        const r = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
        if (r.ok) loadModels();
    }, [loadModels]);

    // Poll for download progress
    useEffect(() => {
        if (!downloadPolling) return;
        const interval = setInterval(loadModels, 5000);
        return () => clearInterval(interval);
    }, [downloadPolling, loadModels]);

    return {
        models, allModels, selectedModel, setSelectedModel,
        hasReadyModel, recommendedModel, isInitializing,
        downloadPolling, autoDownloadStarting,
        loadModels, startDownload, cancelDownload, deleteModel: deleteModelHandler,
        setAllModels, setModels, setHasReadyModel, setDownloadPolling,
    };
};

// ============ Audio Player Hook ============
var useAudioPlayer = (library) => {
    const [playingId, setPlayingId] = useState(null);
    const [playingItem, setPlayingItem] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const audioRef = useRef(null);
    const playNextRef = useRef(null);

    const seek = useCallback((time) => {
        if (audioRef.current) {
            audioRef.current.currentTime = time;
            setProgress(time);
        }
    }, []);

    const setVolumeHandler = useCallback((vol) => {
        setVolume(vol);
        if (audioRef.current) audioRef.current.volume = vol;
    }, []);

    const play = useCallback((item) => {
        if (!item.output_file && (!item.output_files || item.output_files.length === 0)) return;

        // Toggle play/pause for same item
        if (playingId === item.id && audioRef.current) {
            if (audioRef.current.paused) {
                audioRef.current.play().then(() => setIsPlaying(true));
            } else {
                audioRef.current.pause();
                setIsPlaying(false);
            }
            return;
        }

        // Stop current audio
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.src = '';
            audioRef.current = null;
        }

        setPlayingId(item.id);
        setPlayingItem(item);
        setProgress(0);
        setDuration(0);
        setIsPlaying(true);

        const audio = new Audio();
        audio.preload = 'auto';
        audio.volume = volume;
        audio.onended = () => {
            setIsPlaying(false);
            setProgress(0);
            // Use ref to get current playNext function
            if (playNextRef.current) playNextRef.current();
        };
        audio.onerror = () => setIsPlaying(false);
        audio.ontimeupdate = () => setProgress(audio.currentTime);
        audio.onloadedmetadata = () => setDuration(audio.duration);
        audio.src = `/api/audio/${item.id}/0`;
        audioRef.current = audio;
        audio.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
    }, [playingId, volume]);

    const playNext = useCallback(() => {
        if (!playingItem) return;
        const songs = library.filter(l => l.status === 'completed' && (l.output_file || l.output_files?.length > 0));
        const idx = songs.findIndex(l => l.id === playingItem.id);
        if (idx >= 0 && idx < songs.length - 1) play(songs[idx + 1]);
    }, [playingItem, library, play]);

    const playPrev = useCallback(() => {
        if (!playingItem) return;
        const songs = library.filter(l => l.status === 'completed' && (l.output_file || l.output_files?.length > 0));
        const idx = songs.findIndex(l => l.id === playingItem.id);
        if (idx > 0) play(songs[idx - 1]);
    }, [playingItem, library, play]);

    // Keep playNextRef updated with current playNext function
    useEffect(() => {
        playNextRef.current = playNext;
    }, [playNext]);

    // Update playingItem when library changes
    useEffect(() => {
        if (playingItem) {
            const updated = library.find(item => item.id === playingItem.id);
            if (updated) setPlayingItem(updated);
        }
    }, [library, playingItem]);

    return {
        playingId, playingItem, isPlaying, progress, duration, volume,
        play, seek, setVolume: setVolumeHandler, playNext, playPrev,
    };
};

// ============ Time Estimation Hook ============
var useTimeEstimation = (timingStats) => {
    return useCallback((model, sectionsList, hasReference = false) => {
        const numSections = sectionsList.length;
        const totalLyrics = sectionsList.reduce((acc, s) => acc + (s.lyrics || '').length, 0);
        const hasLyrics = totalLyrics > 0;

        // Try historical data first
        if (timingStats?.has_history && timingStats.models?.[model]) {
            const modelStats = timingStats.models[model];
            const sectionKey = String(numSections);
            
            if (modelStats.by_sections?.[sectionKey]) {
                let estimate = modelStats.by_sections[sectionKey];
                if (hasLyrics && modelStats.avg_with_lyrics && modelStats.avg_without_lyrics) {
                    const ratio = modelStats.avg_with_lyrics / modelStats.avg_without_lyrics;
                    if (!hasLyrics) estimate = Math.round(estimate / ratio);
                }
                if (hasReference && modelStats.avg_with_reference && modelStats.avg_without_reference) {
                    estimate = Math.round(estimate * (modelStats.avg_with_reference / modelStats.avg_without_reference));
                }
                return estimate;
            }

            let baseEstimate = hasLyrics ? modelStats.avg_with_lyrics : modelStats.avg_without_lyrics;
            if (!baseEstimate) baseEstimate = modelStats.avg_time;
            if (baseEstimate) {
                const multiplier = 1 + ((numSections - 5) * 0.08);
                let estimate = Math.round(baseEstimate * multiplier);
                if (hasReference && modelStats.avg_with_reference && modelStats.avg_without_reference) {
                    estimate = Math.round(estimate * (modelStats.avg_with_reference / modelStats.avg_without_reference));
                }
                return Math.max(60, estimate);
            }
        }

        // Fallback: static estimation
        let baseTime = MODEL_BASE_TIMES[model] || 240;
        const lyricsAdjust = hasLyrics ? Math.floor(totalLyrics / 500) * 30 : -30;
        const sectionsAdjust = Math.max(0, numSections - 3) * 15;
        const durationSections = sectionsList.filter(s =>
            s.type.includes('intro') || s.type.includes('outro') || s.type.includes('inst')
        ).length;
        const durationAdjust = durationSections * 20;
        const refAdjust = hasReference ? 60 : 0;

        return Math.max(60, baseTime + lyricsAdjust + sectionsAdjust + durationAdjust + refAdjust);
    }, [timingStats]);
};

