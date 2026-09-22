/**
 * Interactive Client Script for Traffic Signal Violation Detection System.
 * Manages live telemetry polling, interactive stop-line dragging,
 * signal remote controls, and evidence modal inspection.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const videoStream = document.getElementById('main-video-stream');
    const canvas = document.getElementById('interactive-canvas');
    const ctx = canvas.getContext('2d');
    const canvasHelpBadge = document.getElementById('canvas-help-badge');

    // KPI Elements
    const statTotalVehicles = document.getElementById('stat-total-vehicles');
    const statActiveVehicles = document.getElementById('stat-active-vehicles');
    const statSignalViolations = document.getElementById('stat-signal-violations');
    const statSpeedViolations = document.getElementById('stat-speed-violations');
    const statFps = document.getElementById('stat-fps');

    // Video & Remote Controls
    const selectVideo = document.getElementById('select-video-source');
    const btnTogglePause = document.getElementById('btn-toggle-pause');
    const pauseIcon = document.getElementById('pause-icon');
    const btnResetPipeline = document.getElementById('btn-reset-pipeline');
    const btnExportCsv = document.getElementById('btn-export-csv');

    // Signal Buttons
    const btnSigAuto = document.getElementById('btn-sig-auto');
    const btnSigGreen = document.getElementById('btn-sig-green');
    const btnSigYellow = document.getElementById('btn-sig-yellow');
    const btnSigRed = document.getElementById('btn-sig-red');

    // Parameter Sliders
    const sliderSpeedLimit = document.getElementById('slider-speed-limit');
    const valSpeedLimit = document.getElementById('val-speed-limit');
    const sliderConfidence = document.getElementById('slider-confidence');
    const valConfidence = document.getElementById('val-confidence');
    const sliderCalibration = document.getElementById('slider-calibration');
    const valCalibration = document.getElementById('val-calibration');

    // Incident Log
    const incidentsContainer = document.getElementById('incidents-container');
    const emptyIncidentsPlaceholder = document.getElementById('empty-incidents-placeholder');
    const badgeViolationCount = document.getElementById('badge-violation-count');

    // Modal Elements
    const evidenceModal = document.getElementById('evidence-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const modalEvidenceImg = document.getElementById('modal-evidence-img');
    const modalDetailId = document.getElementById('modal-detail-id');
    const modalDetailClass = document.getElementById('modal-detail-class');
    const modalDetailSpeed = document.getElementById('modal-detail-speed');
    const modalDetailLight = document.getElementById('modal-detail-light');
    const modalDetailViolation = document.getElementById('modal-detail-violation');
    const btnDownloadEvidence = document.getElementById('btn-download-evidence');

    // Canvas State
    let isDrawingLine = false;
    let dragStart = null;
    let dragEnd = null;
    let activeStopLine = null; // Video coords: [[x1, y1], [x2, y2]]
    let videoDim = { w: 1280, h: 720 };
    let violationsCache = [];

    // -------------------------------------------------------------
    // Canvas Sizing & Stop Line Drawing
    // -------------------------------------------------------------
    function resizeCanvas() {
        const rect = videoStream.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
            canvas.width = rect.width;
            canvas.height = rect.height;
            drawOverlay();
        }
    }

    window.addEventListener('resize', resizeCanvas);
    videoStream.addEventListener('load', resizeCanvas);

    function toCanvasCoords(vx, vy) {
        const scaleX = canvas.width / (videoDim.w || 1280);
        const scaleY = canvas.height / (videoDim.h || 720);
        return { x: vx * scaleX, y: vy * scaleY };
    }

    function toVideoCoords(cx, cy) {
        const scaleX = (videoDim.w || 1280) / canvas.width;
        const scaleY = (videoDim.h || 720) / canvas.height;
        return { x: Math.round(cx * scaleX), y: Math.round(cy * scaleY) };
    }

    function drawOverlay() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Render current dragging line
        if (isDrawingLine && dragStart && dragEnd) {
            ctx.beginPath();
            ctx.setLineDash([6, 6]);
            ctx.moveTo(dragStart.x, dragStart.y);
            ctx.lineTo(dragEnd.x, dragEnd.y);
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 3;
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw handle points
            drawHandle(dragStart.x, dragStart.y);
            drawHandle(dragEnd.x, dragEnd.y);
        }
    }

    function drawHandle(x, y) {
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#06b6d4';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Mouse events for canvas dragging
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        dragStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        dragEnd = { ...dragStart };
        isDrawingLine = true;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawingLine) return;
        const rect = canvas.getBoundingClientRect();
        dragEnd = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        drawOverlay();
    });

    canvas.addEventListener('mouseup', () => {
        if (isDrawingLine && dragStart && dragEnd) {
            isDrawingLine = false;
            const dist = Math.hypot(dragEnd.x - dragStart.x, dragEnd.y - dragStart.y);
            if (dist > 20) {
                const p1 = toVideoCoords(dragStart.x, dragStart.y);
                const p2 = toVideoCoords(dragEnd.x, dragEnd.y);
                sendStopLine(p1.x, p1.y, p2.x, p2.y);
                canvasHelpBadge.innerText = `Stop line updated: (${p1.x}, ${p1.y}) -> (${p2.x}, ${p2.y})`;
                setTimeout(() => {
                    canvasHelpBadge.innerText = "Click & Drag to Reposition Stop Line";
                }, 3000);
            }
            dragStart = null;
            dragEnd = null;
            drawOverlay();
        }
    });

    function sendStopLine(x1, y1, x2, y2) {
        fetch('/api/set_stop_line', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x1, y1, x2, y2 })
        }).catch(err => console.error('Error saving stop line:', err));
    }

    // -------------------------------------------------------------
    // Traffic Signal Controls
    // -------------------------------------------------------------
    function updateSignalButtons(mode, state) {
        [btnSigAuto, btnSigGreen, btnSigYellow, btnSigRed].forEach(b => b.classList.remove('active'));

        if (mode === 'AUTO_CYCLE') {
            btnSigAuto.classList.add('active');
        } else {
            if (state === 'GREEN') btnSigGreen.classList.add('active');
            else if (state === 'YELLOW') btnSigYellow.classList.add('active');
            else if (state === 'RED') btnSigRed.classList.add('active');
        }
    }

    btnSigAuto.addEventListener('click', () => {
        fetch('/api/set_signal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'AUTO_CYCLE' })
        }).then(() => updateSignalButtons('AUTO_CYCLE', 'GREEN'));
    });

    btnSigGreen.addEventListener('click', () => {
        fetch('/api/set_signal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'MANUAL', state: 'GREEN' })
        }).then(() => updateSignalButtons('MANUAL', 'GREEN'));
    });

    btnSigYellow.addEventListener('click', () => {
        fetch('/api/set_signal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'MANUAL', state: 'YELLOW' })
        }).then(() => updateSignalButtons('MANUAL', 'YELLOW'));
    });

    btnSigRed.addEventListener('click', () => {
        fetch('/api/set_signal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'MANUAL', state: 'RED' })
        }).then(() => updateSignalButtons('MANUAL', 'RED'));
    });

    // -------------------------------------------------------------
    // Playback and Parameter Handlers
    // -------------------------------------------------------------
    btnTogglePause.addEventListener('click', () => {
        fetch('/api/toggle_pause', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                pauseIcon.innerHTML = data.is_paused ? '&#9658; Resume' : '&#10074;&#10074; Pause';
            });
    });

    btnResetPipeline.addEventListener('click', () => {
        fetch('/api/reset', { method: 'POST' })
            .then(() => {
                fetchStats();
                fetchViolations();
            });
    });

    btnExportCsv.addEventListener('click', () => {
        window.location.href = '/api/export_csv';
    });

    sliderSpeedLimit.addEventListener('input', (e) => {
        const val = e.target.value;
        valSpeedLimit.textContent = `${val} km/h`;
        fetch('/api/set_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed_limit: parseFloat(val) })
        });
    });

    sliderConfidence.addEventListener('input', (e) => {
        const val = e.target.value;
        valConfidence.textContent = val;
        fetch('/api/set_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conf_thresh: parseFloat(val) })
        });
    });

    sliderCalibration.addEventListener('input', (e) => {
        const val = e.target.value;
        valCalibration.textContent = val;
        fetch('/api/set_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ meters_per_pixel: parseFloat(val) })
        });
    });

    // -------------------------------------------------------------
    // Video Source Management
    // -------------------------------------------------------------
    function loadVideoList() {
        fetch('/api/get_videos')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    selectVideo.innerHTML = '';
                    data.videos.forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v;
                        opt.textContent = v;
                        if (v === data.current_video) opt.selected = true;
                        selectVideo.appendChild(opt);
                    });
                }
            });
    }

    selectVideo.addEventListener('change', (e) => {
        const vid = e.target.value;
        fetch('/api/select_video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video: vid })
        }).then(() => {
            // Force stream reload
            videoStream.src = '/video_feed?t=' + Date.now();
        });
    });

    // -------------------------------------------------------------
    // Telemetry Polling
    // -------------------------------------------------------------
    function fetchStats() {
        fetch('/api/stats')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    const s = data.stats;
                    statTotalVehicles.textContent = s.total_vehicles || 0;
                    statActiveVehicles.textContent = s.active_vehicles || 0;
                    statSignalViolations.textContent = s.signal_violations || 0;
                    statSpeedViolations.textContent = s.speed_violations || 0;
                    statFps.textContent = (s.fps_actual || 0).toFixed(1);

                    if (s.frame_w && s.frame_h) {
                        videoDim = { w: s.frame_w, h: s.frame_h };
                    }

                    if (data.stop_line) {
                        activeStopLine = data.stop_line;
                    }

                    // Keep remote buttons in sync
                    updateSignalButtons(data.signal_mode, data.signal_state);
                }
            })
            .catch(err => console.error('Stats error:', err));
    }

    function fetchViolations() {
        fetch('/api/violations')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    violationsCache = data.violations;
                    renderIncidents(data.violations);
                }
            })
            .catch(err => console.error('Violations error:', err));
    }

    function renderIncidents(viols) {
        badgeViolationCount.textContent = viols.length;

        if (viols.length === 0) {
            emptyIncidentsPlaceholder.style.display = 'flex';
            // Clear existing incident cards
            const cards = incidentsContainer.querySelectorAll('.incident-card');
            cards.forEach(c => c.remove());
            return;
        }

        emptyIncidentsPlaceholder.style.display = 'none';

        // Check if cards already match to avoid unnecessary DOM thrash
        const existingIds = Array.from(incidentsContainer.querySelectorAll('.incident-card')).map(c => c.dataset.violId);
        const incomingIds = viols.map(v => String(v.violation_id));

        if (JSON.stringify(existingIds) === JSON.stringify(incomingIds)) {
            return; // No new changes
        }

        // Re-render
        incidentsContainer.innerHTML = '';
        viols.forEach(v => {
            const card = document.createElement('div');
            card.className = 'incident-card';
            card.dataset.violId = v.violation_id;

            card.innerHTML = `
                <div class="incident-thumb-wrap">
                    <img class="incident-thumb" src="/violations/${v.snapshot_file}" alt="Vehicle #${v.track_id}" loading="lazy">
                </div>
                <div class="incident-details">
                    <div class="incident-row-top">
                        <span class="incident-id">#${v.track_id} ${v.class_name.toUpperCase()}</span>
                        <span class="incident-time">${v.timestamp.split(' ')[1] || v.timestamp}</span>
                    </div>
                    <div class="incident-row-mid">
                        <span class="incident-badge-offense">${v.violation_type}</span>
                        <span class="incident-speed-tag">${v.speed_kmh} km/h</span>
                    </div>
                    <div class="incident-row-bot">
                        <span>Signal: ${v.signal_state} &bull; Frame: ${v.frame_idx}</span>
                    </div>
                </div>
            `;

            card.addEventListener('click', () => openModal(v));
            incidentsContainer.appendChild(card);
        });
    }

    // -------------------------------------------------------------
    // Evidence Modal
    // -------------------------------------------------------------
    function openModal(v) {
        modalEvidenceImg.src = `/violations/${v.snapshot_file}`;
        modalDetailId.textContent = `#${v.track_id}`;
        modalDetailClass.textContent = v.class_name.toUpperCase();
        modalDetailSpeed.textContent = `${v.speed_kmh} km/h`;
        modalDetailLight.textContent = v.signal_state;
        modalDetailViolation.textContent = v.violation_type;
        btnDownloadEvidence.href = `/violations/${v.snapshot_file}`;

        evidenceModal.classList.add('open');
    }

    function closeModal() {
        evidenceModal.classList.remove('open');
    }

    btnCloseModal.addEventListener('click', closeModal);
    evidenceModal.addEventListener('click', (e) => {
        if (e.target === evidenceModal) closeModal();
    });

    // -------------------------------------------------------------
    // Initialization & Polling (Clean slate on page load/refresh)
    // -------------------------------------------------------------
    fetch('/api/reset', { method: 'POST' }).finally(() => {
        loadVideoList();
        fetchStats();
        fetchViolations();
    });

    setInterval(fetchStats, 1200);
    setInterval(fetchViolations, 1800);
});
