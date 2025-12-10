// SongGeneration Studio - Icon Components

var PlayIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 12 12" fill={color}>
        <path d="M2 1.5v9l8-4.5-8-4.5z" />
    </svg>
);

var PauseIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 12 12" fill={color}>
        <rect x="2" y="1" width="3" height="10" rx="1" />
        <rect x="7" y="1" width="3" height="10" rx="1" />
    </svg>
);

var CloseIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M18 6L6 18M6 6l12 12"/>
    </svg>
);

var MusicNoteIcon = ({ size = 24, color = '#666' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M9 18V5l12-2v13"/>
        <circle cx="6" cy="18" r="3"/>
        <circle cx="18" cy="16" r="3"/>
    </svg>
);

var ExpandIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
    </svg>
);

var ChevronIcon = ({ size = 16, color = '#888', rotated = false }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transition: 'transform 0.2s', transform: rotated ? 'rotate(180deg)' : 'rotate(0deg)' }}>
        <path d="M6 9l6 6 6-6"/>
    </svg>
);

var VolumeIcon = ({ size = 16, color = '#666' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
    </svg>
);

var VolumeFullIcon = ({ size = 18, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M11 5L6 9H2v6h4l5 4V5z"/>
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>
    </svg>
);

var VolumeMuteIcon = ({ size = 18, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M11 5L6 9H2v6h4l5 4V5z"/>
        <line x1="23" y1="9" x2="17" y2="15"/>
        <line x1="17" y1="9" x2="23" y2="15"/>
    </svg>
);

var PlusIcon = ({ size = 16, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M12 5v14M5 12h14"/>
    </svg>
);

var TrashIcon = ({ size = 10, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
    </svg>
);

var EditIcon = ({ size = 14, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
    </svg>
);

var SkipBackIcon = ({ size = 20, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
    </svg>
);

var SkipForwardIcon = ({ size = 20, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>
    </svg>
);

var PlayLargeIcon = ({ size = 24, color = '#fff', style = {} }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color} style={style}>
        <path d="M8 5v14l11-7z"/>
    </svg>
);

var PauseLargeIcon = ({ size = 24, color = '#fff' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <rect x="6" y="4" width="4" height="16"/>
        <rect x="14" y="4" width="4" height="16"/>
    </svg>
);

var SpinnerIcon = ({ size = 24 }) => (
    <div style={{ width: size, height: size, border: '2px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
);

var SettingsIcon = ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3"/>
        <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
    </svg>
);

var CheckIcon = ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
        <polyline points="20 6 9 17 4 12"/>
    </svg>
);

var DragIcon = ({ size = 12, color = 'currentColor', style = {} }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color} style={style}>
        <circle cx="9" cy="6" r="2"/>
        <circle cx="15" cy="6" r="2"/>
        <circle cx="9" cy="12" r="2"/>
        <circle cx="15" cy="12" r="2"/>
        <circle cx="9" cy="18" r="2"/>
        <circle cx="15" cy="18" r="2"/>
    </svg>
);
