// ============================================================
// TremorPause AI — app.js  v6
// 4-position sequential protocol, 40s per position
// Inline report generation (port of reportv15.py)
// ============================================================

// ---------- CONSTANTS ----------
const WINDOW_SIZE   = 128;
const STEP_SIZE     = 32;
const CALIB_SAMPLES = 50;
const DEFAULT_FT    = 9.32;
let freqThreshold   = DEFAULT_FT;

const POSITIONS = [
    { key: 'Still', label: 'Position 1 — Still',  desc: 'Hand flat on table, forearm rested.' },
    { key: 'Hover', label: 'Position 2 — Hover',  desc: 'Forearm hovering above knee, elbow unlocked.' },
    { key: 'Spoon', label: 'Position 3 — Spoon',  desc: 'Hold spoon to mouth, elbow out.' },
    { key: 'Point', label: 'Position 4 — Point',  desc: 'Arm pointed outward above shoulder level.' }
];

const RECORDING_SECONDS = 40;
const SKIP_INITIAL      = 6;   // skip first N windows (calibration settling), scaled from 15→40s

const DEVICEMAP = {
    left: {
        service: '12345678-1234-5678-1234-56789abcdef0',
        char:    'abcdef01-1234-5678-1234-56789abcdef0',
        motor:   'abcdef02-1234-5678-1234-56789abcdef0'
    },
    right: {
        service: '12345678-1234-5678-1234-56789abcdef0',
        char:    'abcdef01-1234-5678-1234-56789abcdef0',
        motor:   null
    }
};

// Active mode: 'researcher' or 'participant'
let activeMode = 'researcher';

function switchTab(mode) {
    activeMode = mode;
    document.getElementById('tab-researcher').classList.toggle('tab-active', mode === 'researcher');
    document.getElementById('tab-participant').classList.toggle('tab-active', mode === 'participant');
    document.getElementById('pane-researcher').style.display = mode === 'researcher' ? 'block' : 'none';
    document.getElementById('pane-participant').style.display = mode === 'participant' ? 'block' : 'none';
}

const deviceNames = { left: 'Left-TremorPause', right: 'Right-TremorPause' };

let model       = null;
let isRecording = false;

// Session state — tracks all 4 positions
let session = {
    participantId: '',
    diagnosis:     'TREMOR',
    currentPos:    0,           // 0-3
    started:       false,
    complete:      false,
    // positionData[i] = array of window objects for position i
    positionData:  [[], [], [], []]
};

function makeSideState() {
    return {
        device: null, gattServer: null, service: null,
        imuChar: null, motorChar: null,
        rawBuffer: [], timestamps: [],
        calibBuf: [], bias: [0, 0, 0],
        calibrated: false, connected: false,
        peakSeverity: 0, sampleCount: 0
    };
}
const sideState = { left: makeSideState(), right: makeSideState() };

// ============================================================
// MODEL LOADING
// ============================================================
function loadModel() {
    setModelStatus('Loading…', '#ff9500');
    fetch('TremorModel.json')
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then(data => {
            model = data;
            setModelStatus(`✓ ${Array.isArray(data) ? data.length : '?'} trees`, '#34c759');
        })
        .catch(e => setModelStatus(`✗ ${e.message}`, '#ff3b30'));
}

// ============================================================
// FFT
// ============================================================
function computeFFTMagnitudes(signal) {
    const N  = signal.length;
    const re = new Float64Array(signal);
    const im = new Float64Array(N);
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { [re[i], re[j]] = [re[j], re[i]]; }
    }
    for (let len = 2; len <= N; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for (let i = 0; i < N; i += len) {
            let cRe = 1, cIm = 0;
            for (let k = 0; k < (len >> 1); k++) {
                const uRe = re[i+k], uIm = im[i+k];
                const vRe = re[i+k+(len>>1)]*cRe - im[i+k+(len>>1)]*cIm;
                const vIm = re[i+k+(len>>1)]*cIm + im[i+k+(len>>1)]*cRe;
                re[i+k]          = uRe+vRe; im[i+k]          = uIm+vIm;
                re[i+k+(len>>1)] = uRe-vRe; im[i+k+(len>>1)] = uIm-vIm;
                const nr = cRe*wRe - cIm*wIm; cIm = cRe*wIm + cIm*wRe; cRe = nr;
            }
        }
    }
    const half = N >> 1;
    const mag  = new Float64Array(half);
    for (let i = 0; i < half; i++) mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);
    return mag;
}

// ============================================================
// FEATURE EXTRACTION
// ============================================================
function extractFeatures(rawWindow, bias, timestamps) {
    const n = rawWindow.length;
    const timeDiffSec = (timestamps[n-1] - timestamps[0]) / 1000.0;
    const actualFs    = timeDiffSec > 0 ? n / timeDiffSec : 50.0;
    const data = rawWindow.map(([ax, ay, az]) => [ax-bias[0], ay-bias[1], az-bias[2]]);
    const mag  = data.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z));
    const meanVal = mag.reduce((s, v) => s + v, 0) / n;
    const stdVal  = Math.sqrt(mag.reduce((s, v) => s + (v-meanVal)**2, 0) / n);
    const maxVal  = Math.max(...mag);
    const energy  = mag.reduce((s, v) => s + v*v, 0);
    const sig    = mag.map(v => v - meanVal);
    const fftMag = computeFFTMagnitudes(sig);
    let domFreq = 0, maxAmp = -Infinity;
    for (let i = 0; i < fftMag.length; i++) {
        if (fftMag[i] > maxAmp) { maxAmp = fftMag[i]; domFreq = i * actualFs / n; }
    }
    let vibRate = 0;
    for (let i = 1; i < sig.length; i++) {
        if (Math.sign(sig[i]) !== Math.sign(sig[i-1])) vibRate++;
    }
    return { featureArray: [meanVal, stdVal, maxVal, energy, domFreq, vibRate], domFreq, actualFs };
}

// ============================================================
// ML INFERENCE
// ============================================================
const TREMOR_IDX = 1;

function predictProba(featureArray) {
    if (!model || !model.length) return 0;
    let total = 0;
    for (const nodes of model) {
        let id = 0;
        while (nodes[id].feature !== -2) {
            const { feature, threshold, left, right } = nodes[id];
            id = featureArray[feature] <= threshold ? left : right;
        }
        const counts = nodes[id].value[0];
        const sum    = counts.reduce((a, b) => a + b, 0);
        total += sum > 0 ? counts[TREMOR_IDX] / sum : 0;
    }
    return total / model.length;
}

function sigmoidGate(hz, ft) {
    return 1.0 / (1.0 + Math.exp(-1.0 * (hz - ft)));
}

// reportv15.py uses -2.5 steepness for report display
function sigmoidReport(hz, ft) {
    return 1.0 / (1.0 + Math.exp(-2.5 * (hz - ft)));
}

// ============================================================
// IMU HANDLER
// ============================================================
function handleIMU(event, side) {
    const raw   = new TextDecoder().decode(event.target.value).trim();
    const parts = raw.split(',').map(Number);
    if (parts.length < 3 || parts.some(isNaN)) return;

    const s   = sideState[side];
    const now = Date.now();
    const xyz = [parts[0], parts[1], parts[2]];

    s.sampleCount++;
    if (s.sampleCount <= 3) addDiagLog(side, `Raw[${s.sampleCount}]: [${xyz.map(v=>v.toFixed(3)).join(', ')}] dps`);

    if (!s.calibrated) {
        s.calibBuf.push(xyz);
        const pct = Math.round((s.calibBuf.length / CALIB_SAMPLES) * 100);
        setConnStatus(side, `Calibrating… ${pct}%`, '#ff9500');
        if (s.calibBuf.length >= CALIB_SAMPLES) {
            s.bias = [0, 1, 2].map(i => s.calibBuf.reduce((sum, r) => sum + r[i], 0) / CALIB_SAMPLES);
            s.calibrated = true; s.calibBuf = [];
            setConnStatus(side, 'Connected ✓', '#34c759');
            setCalibBadge(side, true);
            addDiagLog(side, `✓ Bias=[${s.bias.map(v=>v.toFixed(3)).join(', ')}]`);
            updateDebugBias(side, s.bias);
        }
        return;
    }

    s.rawBuffer.push(xyz);
    s.timestamps.push(now);

    if (s.rawBuffer.length >= WINDOW_SIZE) {
        const winXYZ  = s.rawBuffer.slice(0, WINDOW_SIZE);
        const winTime = s.timestamps.slice(0, WINDOW_SIZE);

        const { featureArray, domFreq, actualFs } = extractFeatures(winXYZ, s.bias, winTime);
        const rawProb    = predictProba(featureArray);
        const freqWeight = sigmoidGate(domFreq, freqThreshold);
        const severity   = rawProb * freqWeight * 100;

        if (severity > s.peakSeverity) s.peakSeverity = severity;
        updateDebugPanel(side, featureArray, rawProb, freqWeight, severity, actualFs);
        updateSideUI(side, severity, domFreq, actualFs);

        if (isRecording) {
            // Route to participant session if in participant mode
            const isParticipantMode = (activeMode === 'participant');
            const posIdx = isParticipantMode ? pSession.currentPos : session.currentPos;
            const targetData = isParticipantMode ? pSession.positionData : session.positionData;
            targetData[posIdx].push({
                timestamp:      now,
                side,
                severity:       parseFloat(severity.toFixed(2)),
                mean_mag:       parseFloat(featureArray[0].toFixed(4)),
                std_mag:        parseFloat(featureArray[1].toFixed(4)),
                max_mag:        parseFloat(featureArray[2].toFixed(4)),
                energy:         parseFloat(featureArray[3].toFixed(2)),
                dom_freq_hz:    parseFloat(featureArray[4].toFixed(2)),
                vibration_rate: featureArray[5],
                raw_prob:       parseFloat(rawProb.toFixed(4)),
                freq_weight:    parseFloat(freqWeight.toFixed(4)),
                actual_fs:      parseFloat(actualFs.toFixed(1))
            });
        } else {
            sendMotorFeedback(side, severity);
        }

        s.rawBuffer  = s.rawBuffer.slice(STEP_SIZE);
        s.timestamps = s.timestamps.slice(STEP_SIZE);
    }
}

// ============================================================
// BLUETOOTH
// ============================================================
async function connectBluetooth(side, forceAll = false) {
    const cfg = DEVICEMAP[side];
    const btn = document.getElementById(`${side}-conn-btn`);
    try {
        btn.disabled = true;
        setConnStatus(side, 'Opening scanner…', '#ff9500');
        const name = deviceNames[side];
        let opts;
        if (forceAll) {
            opts = { acceptAllDevices: true, optionalServices: [cfg.service] };
            addDiagLog(side, 'Scan: ALL DEVICES');
        } else if (name) {
            opts = { filters: [{ name }], optionalServices: [cfg.service] };
            addDiagLog(side, `Scan: name="${name}"`);
        } else {
            opts = { filters: [{ services: [cfg.service] }], optionalServices: [cfg.service] };
            addDiagLog(side, 'Scan: service UUID');
        }
        const device  = await navigator.bluetooth.requestDevice(opts);
        addDiagLog(side, `Found: "${device.name || '(unnamed)'}"`);
        setConnStatus(side, 'Connecting…', '#ff9500');
        const server  = await device.gatt.connect();
        const service = await server.getPrimaryService(cfg.service);
        addDiagLog(side, 'Service OK');
        const imuChar = await service.getCharacteristic(cfg.char);
        addDiagLog(side, 'IMU char OK');
        let motorChar = null;
        if (cfg.motor) {
            try { motorChar = await service.getCharacteristic(cfg.motor); addDiagLog(side, 'Motor char OK'); }
            catch (e) { addDiagLog(side, 'Motor char: not found'); }
        }
        const s = sideState[side];
        Object.assign(s, { device, gattServer: server, service, imuChar, motorChar, connected: true });
        setConnStatus(side, 'Calibrating…', '#ff9500');
        btn.textContent = 'Disconnect'; btn.disabled = false;
        btn.onclick = () => disconnectBluetooth(side);
        await imuChar.startNotifications();
        imuChar.addEventListener('characteristicvaluechanged', e => handleIMU(e, side));
        addDiagLog(side, '✓ Notifications started');
        device.addEventListener('gattserverdisconnected', () => onDisconnect(side));
    } catch (err) {
        const msg = err.message || String(err);
        addDiagLog(side, `ERROR: ${msg}`);
        setConnStatus(side, `Failed: ${msg.slice(0, 50)}`, '#ff3b30');
        btn.disabled = false; btn.textContent = 'Connect';
        btn.onclick = () => connectBluetooth(side);
    }
}

async function disconnectBluetooth(side) {
    const s = sideState[side];
    if (s.device?.gatt?.connected) try { await s.device.gatt.disconnect(); } catch(_) {}
    onDisconnect(side);
}

function onDisconnect(side) {
    Object.assign(sideState[side], makeSideState());
    setConnStatus(side, 'Disconnected', '#ff3b30');
    setCalibBadge(side, false); resetSideUI(side);
    addDiagLog(side, 'Disconnected');
    const btn = document.getElementById(`${side}-conn-btn`);
    if (btn) { btn.textContent = 'Connect'; btn.onclick = () => connectBluetooth(side); btn.disabled = false; }
}

async function sendMotorFeedback(side, severity) {
    const mc = sideState[side].motorChar;
    if (!mc) return;
    try { await mc.writeValueWithoutResponse(new Uint8Array([Math.min(Math.floor(severity), 100)])); }
    catch (_) {}
}

// ============================================================
// SESSION PROTOCOL — 4-POSITION SEQUENTIAL RECORDING
// ============================================================
let assessmentTimer = null;

function startSession() {
    const pid  = document.getElementById('participant-id').value.trim();
    const diag = document.getElementById('diagnosis-select').value;
    if (!pid) { showToast('Enter a Participant ID first'); return; }

    session = {
        participantId: pid,
        diagnosis:     diag,
        currentPos:    0,
        started:       true,
        complete:      false,
        positionData:  [[], [], [], []]
    };

    document.getElementById('session-setup').style.display    = 'none';
    document.getElementById('session-recording').style.display = 'block';
    document.getElementById('report-section').style.display   = 'none';
    updateProtocolUI();
}

function updateProtocolUI() {
    const pos   = POSITIONS[session.currentPos];
    const total = POSITIONS.length;

    // Progress dots
    let dots = '';
    for (let i = 0; i < total; i++) {
        const done    = i < session.currentPos;
        const active  = i === session.currentPos;
        const color   = done ? '#34c759' : active ? '#00c2ff' : '#252f3a';
        const border  = active ? '2px solid #00c2ff' : '2px solid transparent';
        dots += `<div style="width:12px;height:12px;border-radius:50%;background:${color};border:${border};"></div>`;
    }
    document.getElementById('pos-dots').innerHTML = dots;

    document.getElementById('pos-label').textContent   = pos.label;
    document.getElementById('pos-desc').textContent    = pos.desc;
    document.getElementById('pos-counter').textContent = `${session.currentPos + 1} of ${total}`;

    const btn = document.getElementById('record-btn');
    btn.textContent = `Start 40s Recording`;
    btn.disabled    = false;
    document.getElementById('countdown').textContent = '';
}

function startPositionRecording() {
    if (isRecording) return;
    isRecording = true;
    const btn = document.getElementById('record-btn');
    const cd  = document.getElementById('countdown');
    btn.disabled = true;

    let rem = RECORDING_SECONDS;
    cd.textContent = `${rem}s remaining`; cd.style.color = '#ff9500';

    assessmentTimer = setInterval(() => {
        rem--;
        cd.textContent = rem > 0 ? `${rem}s remaining` : 'Finishing…';
        if (rem <= 0) {
            clearInterval(assessmentTimer);
            isRecording = false;
            finishPosition();
        }
    }, 1000);
}

function finishPosition() {
    const posIdx = session.currentPos;
    const count  = session.positionData[posIdx].length;
    const cd     = document.getElementById('countdown');
    cd.textContent = `✓ ${POSITIONS[posIdx].key} — ${count} windows captured`;
    cd.style.color = '#34c759';

    session.currentPos++;

    if (session.currentPos >= POSITIONS.length) {
        // All 4 positions done
        session.complete = true;
        document.getElementById('record-btn').textContent = 'All positions complete';
        document.getElementById('record-btn').disabled    = true;
        document.getElementById('report-section').style.display = 'block';
        document.getElementById('report-summary').textContent =
            `Session complete for ${session.participantId} · ${session.diagnosis} · ` +
            POSITIONS.map((p, i) => `${p.key}: ${session.positionData[i].length} windows`).join(' · ');
    } else {
        // Next position
        setTimeout(() => updateProtocolUI(), 800);
    }
}

function resetSession() {
    isRecording = false;
    clearInterval(assessmentTimer);
    document.getElementById('session-setup').style.display    = 'block';
    document.getElementById('session-recording').style.display = 'none';
    document.getElementById('report-section').style.display   = 'none';
}

// ============================================================
// CSV DOWNLOAD (one file per position, matching training format)
// ============================================================
function downloadAllCSVs() {
    if (!session.complete) return;
    const pid  = session.participantId;
    const diag = session.diagnosis;

    POSITIONS.forEach((pos, i) => {
        const rows = session.positionData[i];
        if (!rows.length) return;
        const header = 'mean_mag,std_mag,max_mag,energy,dom_freq_hz,vibration_rate,actual_fs,label,ground_truth\n';
        const label  = diag === 'TREMOR' ? 1 : 0;
        const csv    = header + rows.map(r =>
            `${r.mean_mag},${r.std_mag},${r.max_mag},${r.energy},${r.dom_freq_hz},${r.vibration_rate},${r.actual_fs},${label},${diag}`
        ).join('\n');
        const a = document.createElement('a');
        a.href     = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
        a.download = `data_${pid}_${pos.key}_${diag}_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(a.href);
    });
}

// ============================================================
// INLINE REPORT GENERATION (port of reportv15.py)
// Opens in new tab — requires Plotly CDN
// ============================================================
function generateReport() {
    if (!session.complete) return;

    const pid   = session.participantId;
    const diag  = session.diagnosis;
    const isTremor = diag === 'TREMOR';
    const statusColor = isTremor ? '#ff4b4b' : '#4bff4b';
    const statusText  = isTremor ? 'TREMOR' : 'CONTROL / NON-TREMOR';
    const ft    = freqThreshold;

    // Build per-position stats (mirrors reportv15.py trial loop, SKIP_INITIAL applied)
    let trialRowsHTML = '';
    let allSev = [], allHz = [], allMag = [], allEngy = [], allVrate = [], allTime = [];
    let subjectPeakSev = 0;
    let subjectAvgSevs = [];

    POSITIONS.forEach((pos, i) => {
        const rows = session.positionData[i].slice(SKIP_INITIAL); // skip first N windows
        if (!rows.length) return;

        // Apply report sigmoid (-2.5 steepness, matching reportv15.py)
        const sevs   = rows.map(r => r.raw_prob * sigmoidReport(r.dom_freq_hz, ft) * 100);
        const hzs    = rows.map(r => r.dom_freq_hz);
        const mags   = rows.map(r => r.max_mag);
        const energs = rows.map(r => r.energy);
        const vrates = rows.map(r => r.vibration_rate);
        const times  = rows.map((_, idx) => idx);

        const avgSev   = sevs.reduce((a,b) => a+b,0) / sevs.length;
        const peakSev  = Math.max(...sevs);
        const avgHz    = hzs.reduce((a,b) => a+b,0) / hzs.length;
        const avgVrate = vrates.reduce((a,b) => a+b,0) / vrates.length;
        const avgMag   = mags.reduce((a,b) => a+b,0) / mags.length;
        const avgEngy  = energs.reduce((a,b) => a+b,0) / energs.length;

        subjectPeakSev = Math.max(subjectPeakSev, peakSev);
        subjectAvgSevs.push(avgSev);

        allSev.push(...sevs); allHz.push(...hzs); allMag.push(...mags);
        allEngy.push(...energs); allVrate.push(...vrates); allTime.push(...times);

        const sevColor = avgSev > 50 ? '#ff4b4b' : avgSev > 20 ? '#ff9f1c' : '#4bff4b';
        trialRowsHTML += `
        <tr>
            <td><b>${pos.label}</b></td>
            <td style="color:#00d4ff">${avgHz.toFixed(1)} Hz</td>
            <td>${avgVrate.toFixed(1)}</td>
            <td>${avgMag.toFixed(1)}</td>
            <td>${avgEngy.toFixed(0)}</td>
            <td style="color:${sevColor};font-weight:bold">${avgSev.toFixed(1)}%</td>
            <td>${peakSev.toFixed(1)}%</td>
        </tr>`;
    });

    const overallAvgSev = subjectAvgSevs.length
        ? subjectAvgSevs.reduce((a,b)=>a+b,0) / subjectAvgSevs.length
        : 0;

    // Fatigue drift (linear regression on hz over time, matching reportv15.py)
    let fatigueShift = 0;
    let fatigueB = 0, fatigueM = 0;
    const hzFiltered = allHz.map((h,i) => h > 1.0 ? [allTime[i], h] : null).filter(Boolean);
    if (hzFiltered.length > 10) {
        const xs = hzFiltered.map(p => p[0]);
        const ys = hzFiltered.map(p => p[1]);
        const n  = xs.length;
        const mx = xs.reduce((a,b)=>a+b,0)/n;
        const my = ys.reduce((a,b)=>a+b,0)/n;
        fatigueM = xs.reduce((s,x,i) => s+(x-mx)*(ys[i]-my), 0) / xs.reduce((s,x) => s+(x-mx)**2, 0);
        fatigueB = my - fatigueM * mx;
        fatigueShift = fatigueM * 40; // 40 captures (scaled from 100 in original)
    }

    // Encode chart data as JSON for inline Plotly
    const chartData = JSON.stringify({
        sev: allSev, hz: allHz, mag: allMag, engy: allEngy,
        time: allTime.slice(0, allSev.length),
        statusColor,
        fatigueM, fatigueB,
        hasFatigue: hzFiltered.length > 10
    });

    const reportHTML = `<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>TremorPause Report — ${pid}</title>
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"><\/script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async><\/script>
<style>
body{background:#0b0e14;color:#c9d1d9;font-family:'Segoe UI',sans-serif;padding:40px;margin:0;}
h1{color:#fff;margin-bottom:5px;}
.audit-guide{background:#1c2128;border:1px solid #30363d;padding:25px;border-radius:12px;margin-bottom:30px;line-height:1.5;border-left:5px solid #58a6ff;}
.guide-title{color:#58a6ff;font-weight:bold;text-transform:uppercase;font-size:13px;}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:30px;margin-bottom:20px;}
.card-header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #30363d;padding-bottom:20px;margin-bottom:20px;}
.subj-title{font-size:26px;font-weight:bold;color:#fff;}
.hero-hz{font-size:36px;font-weight:bold;}
.hero-label{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;}
table{width:100%;border-collapse:collapse;margin-bottom:30px;}
th{text-align:left;color:#8b949e;font-size:12px;text-transform:uppercase;border-bottom:2px solid #30363d;padding:12px 10px;}
td{padding:12px 10px;border-bottom:1px solid #21262d;font-size:15px;}
tr:hover td{background:#1c2128;}
.analysis-box{background:#0d1117;padding:20px;border-radius:10px;border:1px solid #21262d;}
.fatigue-hero{margin-bottom:15px;color:#c9d1d9;font-size:15px;padding-left:10px;border-left:3px solid #ff9f1c;}
</style>
</head><body>
<h1>TremorPause AI Report Auditor</h1>
<p style="color:#8b949e;margin-top:0">Powered by Random Forest Engine &amp; Statistical Analysis</p>

<div class="audit-guide">
<h2 style="margin-top:0;border-bottom:1px solid #30363d;padding-bottom:10px">Value Definitions</h2>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:25px">
<div><span class="guide-title">1. AI Severity Score (%)</span>
<p>Confidence is calculated by passing a 6-feature motion vector through a Random Forest Classifier. The result is then gated by the frequency response to ensure probability correlates with clinical tremor bands.</p>
<p style="color:#fff;font-size:14px">$$Sev = P(tremor) \\cdot \\sigma(Hz) \\cdot 100$$</p></div>
<div><span class="guide-title">2. Frequency Filter (Sigmoid)</span>
<p>A logistic sigmoid function creates a soft cutoff and outlier handler. It negligates signals far below and above the calculated center (${ft.toFixed(2)} Hz) while allowing tremor frequencies to pass through with a weight nearing 1.0.</p>
<p style="color:#fff;font-size:14px">$$\\sigma(Hz) = \\frac{1}{1 + e^{-2.5(Hz - ${ft.toFixed(2)})}}$$</p></div>
<div><span class="guide-title">3. Data-Driven Threshold (Youden's J)</span>
<p>The system scans the dataset to find the frequency that maximizes the separation between Control and Tremor populations; this is the center of the sigmoid filter.</p>
<p style="color:#fff;font-size:14px">$$max[J] = max[\\text{Sensitivity} + \\text{Specificity} - 1]$$</p></div>
<div><span class="guide-title">4. Kinetic Energy &amp; Max Mag</span>
<p>Magnitude uses the Euclidean Norm to find the largest single displacement. Energy sums the squared magnitudes.</p>
<p style="color:#fff;font-size:14px">$$Mag = ||\\vec{v}||_2, \\quad Energy = \\sum ||\\vec{v}||^2$$</p></div>
<div><span class="guide-title">5. Mean Frequency &amp; FFT</span>
<p>The FFT isolates hand-specific oscillations from complex multi-vector motion by decomposing raw velocity into constituent frequencies.</p>
<p style="color:#fff;font-size:14px">$$f_{dom} = \\frac{\\text{argmax}(|X_k|) \\cdot f_s}{N}$$</p></div>
<div><span class="guide-title">6. Fatigue Drift</span>
<p>A linear regression models the change in tremor speed over time. A negative slope suggests neuromuscular exhaustion; positive suggests tremor amplification.</p>
<p style="color:#fff;font-size:14px">$$\\hat{y} = mt + b$$</p></div>
</div></div>

<div class="card">
<div class="card-header">
<div>
<span class="subj-title">${pid}</span><br>
<span style="color:${statusColor};font-weight:bold;letter-spacing:1px">${statusText}</span>
</div>
<div style="text-align:center">
<span class="hero-hz" style="color:${statusColor}">${overallAvgSev.toFixed(1)}%</span><br>
<span class="hero-label">AVG AI SEVERITY</span>
</div>
<div style="text-align:right">
<span class="hero-hz">${subjectPeakSev.toFixed(1)}%</span><br>
<span class="hero-label">PEAK AI SEVERITY</span>
</div>
</div>
<table>
<thead><tr>
<th>Trial Block</th><th>Mean Hz</th><th>Vib. Rate</th>
<th>Max Mag</th><th>Energy</th><th>Avg AI Severity</th><th>Peak Severity</th>
</tr></thead>
<tbody>${trialRowsHTML}</tbody>
</table>
<div class="analysis-box">
<div class="fatigue-hero"><b>Intra-Trial Fatigue Effect: ${fatigueShift >= 0 ? '+' : ''}${fatigueShift.toFixed(2)} Hz Shift</b> over 40 captures</div>
<div id="charts"></div>
</div>
</div>

<script>
(function(){
var d = ${chartData};
var sc = d.statusColor;

var fig = {
data: [
{type:'histogram', x:d.sev, nbinsx:20, marker:{color:sc, opacity:0.8}, xaxis:'x1', yaxis:'y1'},
{type:'histogram', x:d.hz,  nbinsx:30, marker:{color:'#00d4ff', opacity:0.8}, xaxis:'x2', yaxis:'y2'},
{type:'scatter',   x:d.time, y:d.mag, mode:'lines', line:{color:'#ff9f1c', width:2}, xaxis:'x3', yaxis:'y3'},
{type:'scatter',   x:d.time, y:d.hz,  mode:'markers', marker:{size:4, color:sc, opacity:0.25}, xaxis:'x4', yaxis:'y4'}
],
layout:{
template:'plotly_dark', height:650, showlegend:false, margin:{l:40,r:40,t:40,b:40},
grid:{rows:2, columns:2, pattern:'independent'},
annotations:[
{text:'AI Severity Distribution', xref:'paper', yref:'paper', x:0.2,  y:1.05, showarrow:false, font:{color:'#c9d1d9'}},
{text:'Frequency Mapping (Hz)',   xref:'paper', yref:'paper', x:0.78, y:1.05, showarrow:false, font:{color:'#c9d1d9'}},
{text:'Kinetic Magnitude vs Time',xref:'paper', yref:'paper', x:0.2,  y:0.45, showarrow:false, font:{color:'#c9d1d9'}},
{text:'Intra-Trial Fatigue Trend',xref:'paper', yref:'paper', x:0.78, y:0.45, showarrow:false, font:{color:'#c9d1d9'}}
],
xaxis1:{title:'AI Severity (%)', domain:[0,0.45]},   yaxis1:{domain:[0.55,1]},
xaxis2:{title:'Frequency (Hz)',   domain:[0.55,1]},   yaxis2:{domain:[0.55,1]},
xaxis3:{title:'Timeline (Captures)', domain:[0,0.45]},yaxis3:{domain:[0,0.45]},
xaxis4:{title:'Timeline (Captures)', domain:[0.55,1]},yaxis4:{domain:[0,0.45]}
}
};

if (d.hasFatigue) {
fig.data.push({
type:'scatter', x:[0,40],
y:[d.fatigueB, d.fatigueM*40+d.fatigueB],
mode:'lines', line:{color:'white', width:3, dash:'dash'},
xaxis:'x4', yaxis:'y4'
});
}
Plotly.newPlot('charts', fig.data, fig.layout);
})();
<\/script>
</body></html>`;

    const blob = new Blob([reportHTML], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    window.open(url, '_blank');
    setTimeout(() => URL.revokeObjectURL(url), 60000);
}

// ============================================================
// DEBUG PANEL
// ============================================================
const diagLogs = { left: [], right: [] };

function addDiagLog(side, msg) {
    const ts = new Date().toISOString().slice(11, 23);
    diagLogs[side].unshift(`[${ts}] ${msg}`);
    if (diagLogs[side].length > 30) diagLogs[side].pop();
    const el = document.getElementById(`${side}-diag-log`);
    if (el) el.textContent = diagLogs[side].join('\n');
}

function updateDebugBias(side, bias) {
    const el = document.getElementById(`${side}-debug-bias`);
    if (el) el.textContent = `[${bias.map(v=>v.toFixed(4)).join(', ')}]`;
}

function updateDebugPanel(side, fa, rawProb, freqWeight, severity, actualFs) {
    const energyPath = fa[3] > 622.51 ? 'TREMOR branch ✓' : 'NORMAL branch ✗';
    const lines = [
        `actual_fs   : ${actualFs.toFixed(2)} Hz`,
        ``,
        `mean_mag    : ${fa[0].toFixed(4)}`,
        `std_mag     : ${fa[1].toFixed(4)}`,
        `max_mag     : ${fa[2].toFixed(4)}`,
        `energy      : ${fa[3].toFixed(2)}  → ${energyPath}`,
        `dom_freq_hz : ${fa[4].toFixed(4)} Hz`,
        `vib_rate    : ${fa[5]}`,
        ``,
        `raw_prob    : ${rawProb.toFixed(4)}`,
        `freq_weight : ${freqWeight.toFixed(4)}  sigmoid(hz - ${freqThreshold.toFixed(2)}Hz)`,
        `SEVERITY    : ${severity.toFixed(2)}%`
    ];
    const el = document.getElementById(`${side}-debug-features`);
    if (el) el.textContent = lines.join('\n');
}

function toggleDiag(side) {
    const el = document.getElementById(`${side}-diag`);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function toggleDebug(side) {
    const el = document.getElementById(`${side}-debug-panel`);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function toggleProtocol() {
    const el = document.getElementById('protocol-panel');
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// ============================================================
// UI HELPERS
// ============================================================
const STATUS_LEVELS = [
    { max: 15,  label: 'STABLE',   color: '#34c759' },
    { max: 45,  label: 'MILD',     color: '#ff9500' },
    { max: 75,  label: 'MODERATE', color: '#ff6b00' },
    { max: 101, label: 'SEVERE',   color: '#ff3b30' }
];
const severityMeta = sev => STATUS_LEVELS.find(s => sev < s.max) ?? STATUS_LEVELS.at(-1);

function updateSideUI(side, severity, hz, fs) {
    const m = severityMeta(severity);
    const s = sideState[side];
    document.getElementById(`${side}-sev`).textContent   = `${severity.toFixed(1)}%`;
    document.getElementById(`${side}-label`).textContent = m.label;
    document.getElementById(`${side}-label`).style.color = m.color;
    const bar = document.getElementById(`${side}-bar`);
    bar.style.width = `${Math.min(severity,100)}%`; bar.style.background = m.color;
    document.getElementById(`${side}-hz`).textContent    = `${hz.toFixed(1)} Hz`;
    document.getElementById(`${side}-motor`).textContent = severity > 15 ? 'ACTIVE' : 'IDLE';
    document.getElementById(`${side}-motor`).style.color = severity > 15 ? '#ff9500' : '#8e8e93';
    const fsEl   = document.getElementById(`${side}-fs`);
    const peakEl = document.getElementById(`${side}-peak`);
    if (fsEl)   fsEl.textContent   = `${fs.toFixed(0)} Hz`;
    if (peakEl) peakEl.textContent = `${s.peakSeverity.toFixed(1)}%`;
}

function resetSideUI(side) {
    for (const k of ['sev','hz','label','peak']) {
        const el = document.getElementById(`${side}-${k}`);
        if (el) { el.textContent = '--'; el.style.color = '#8e8e93'; }
    }
    const bar = document.getElementById(`${side}-bar`);
    if (bar) { bar.style.width = '0%'; bar.style.background = '#34c759'; }
    const df = document.getElementById(`${side}-debug-features`);
    if (df) df.textContent = 'Waiting for 128 samples…';
}

function setConnStatus(side, msg, color) {
    const el = document.getElementById(`${side}-conn-status`);
    if (el) { el.textContent = msg; el.style.color = color; }
}

function setCalibBadge(side, done) {
    const el = document.getElementById(`${side}-calib-badge`);
    if (!el) return;
    el.textContent = done ? '✓ Calibrated' : 'Not calibrated';
    el.style.color  = done ? '#34c759' : '#8e8e93';
}

function setModelStatus(msg, color) {
    const el = document.getElementById('model-status');
    if (el) { el.textContent = msg; el.style.color = color; }
}

// ============================================================
// SETTINGS
// ============================================================
function applySettings() {
    const ft = parseFloat(document.getElementById('freq-input').value);
    if (!isNaN(ft) && ft > 0 && ft <= 20) {
        freqThreshold = ft;
        document.getElementById('threshold-display').textContent = `${ft.toFixed(4)} Hz`;
    }
    deviceNames.left  = document.getElementById('left-name-input').value.trim();
    deviceNames.right = document.getElementById('right-name-input').value.trim();
    showToast('Settings applied');
}

function showToast(msg) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg; el.style.opacity = '1';
    setTimeout(() => el.style.opacity = '0', 2000);
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('DOMContentLoaded', () => {
    loadModel();
    document.getElementById('freq-input').value = freqThreshold;
    document.getElementById('threshold-display').textContent = `${freqThreshold.toFixed(4)} Hz`;
    document.getElementById('left-name-input').value  = deviceNames.left;
    document.getElementById('right-name-input').value = deviceNames.right;
    // Auto-initialize Firebase with hardcoded config
    initFirebase(FIREBASE_CONFIG);
    const cfgEl = document.getElementById('firebase-config-input');
    if (cfgEl) cfgEl.value = JSON.stringify(FIREBASE_CONFIG, null, 2);
});

// ============================================================
// FIREBASE INTEGRATION
// ============================================================
let db = null;

const FIREBASE_CONFIG = {
    apiKey:            "AIzaSyAzMGMviZmrwthvTIPXlGk5VmrqQT2b5NM",
    authDomain:        "tremorpauseweb.firebaseapp.com",
    projectId:         "tremorpauseweb",
    storageBucket:     "tremorpauseweb.firebasestorage.app",
    messagingSenderId: "553085922178",
    appId:             "1:553085922178:web:e6f2ebdafddd9520b77d9b"
};

function initFirebase(config) {
    try {
        if (typeof firebase === 'undefined') {
            console.warn('Firebase SDK not yet loaded');
            return false;
        }
        if (!firebase.apps.length) {
            firebase.initializeApp(config);
        }
        db = firebase.firestore();
        setFirebaseStatus('\u2713 Connected to cloud', '#34c759');
        return true;
    } catch (e) {
        setFirebaseStatus(`\u2717 ${e.message}`, '#ff3b30');
        return false;
    }
}

function saveFirebaseConfig() {
    const raw = document.getElementById('firebase-config-input').value.trim();
    try {
        // Accept either raw JSON or the full firebaseConfig = {...} assignment
        const cleaned = raw
            .replace(/^.*?firebaseConfig\s*=\s*/, '')
            .replace(/;?\s*$/, '')
            .trim();
        const config = JSON.parse(cleaned);
        localStorage.setItem('tp_firebase_config', JSON.stringify(config));
        initFirebase(config);
        showToast('Firebase config saved');
    } catch (e) {
        showToast('Invalid config — paste the JSON object exactly');
    }
}

function setFirebaseStatus(msg, color) {
    const el = document.getElementById('firebase-status');
    if (el) { el.textContent = msg; el.style.color = color; }
}

async function uploadToFirestore(sessionId, participantMode, positionData) {
    if (!db) { showToast('No cloud connection — data not uploaded'); return false; }
    try {
        const batch = db.batch();
        const POSITIONS_P = [
            { key: 'Still' }, { key: 'Hover' }, { key: 'Spoon' }, { key: 'Point' }
        ];
        POSITIONS_P.forEach((pos, i) => {
            const rows = positionData[i] || [];
            if (!rows.length) return;
            const docRef = db.collection('tremor_data').doc(`${sessionId}_${pos.key}`);
            batch.set(docRef, {
                session_id:  sessionId,
                position:    pos.key,
                mode:        participantMode, // 'participant'
                diagnosis:   pSession.diagnosis,
                timestamp:   firebase.firestore.FieldValue.serverTimestamp(),
                windows:     rows.map(r => ({
                    mean_mag:       r.mean_mag,
                    std_mag:        r.std_mag,
                    max_mag:        r.max_mag,
                    energy:         r.energy,
                    dom_freq_hz:    r.dom_freq_hz,
                    vibration_rate: r.vibration_rate,
                    actual_fs:      r.actual_fs,
                    raw_prob:       r.raw_prob,   // stored for accurate population comparison
                    label:          pSession.diagnosis === 'TREMOR' ? 1 : 0,
                    ground_truth:   pSession.diagnosis
                }))
            });
        });
        await batch.commit();
        return true;
    } catch (e) {
        console.error('Firestore upload error:', e);
        showToast(`Upload failed: ${e.message}`);
        return false;
    }
}

// Fetch anonymised population averages per position from Firestore
// Returns { Still: {avg, n}, Hover: {avg, n}, Spoon: {avg, n}, Point: {avg, n}, total: n }
// or null if not enough data (< 5 sessions)
async function fetchPopulationAverages() {
    if (!db) return null;
    try {
        // Each doc = one position for one session
        // We query up to 500 docs (plenty for comparison)
        const snap = await db.collection('tremor_data').limit(500).get();
        if (snap.empty) return null;

        // Accumulate per-position severity sums
        const buckets = { Still: [], Hover: [], Spoon: [], Point: [] };
        let sessionIds = new Set();

        snap.forEach(doc => {
            const data = doc.data();
            const pos  = data.position;
            if (!pos || !buckets[pos] || !data.windows) return;

            sessionIds.add(data.session_id);

            // Compute severity for each window in this doc
            data.windows.forEach(w => {
                // Windows stored from participant mode include raw_prob indirectly
                // via the severity stored at capture time. We recompute from features
                // using the same sigmoid as the report (steepness -2.5).
                // raw_prob is not stored — but we stored all 6 features + label.
                // Use label as a proxy: label=1 (tremor) windows contribute their
                // freq-weighted severity as raw_prob * freq_weight * 100.
                // For a proper comparison we use mean severity of the full dataset
                // as computed by the app at recording time.
                // Since we store windows with all features, we can at minimum
                // compute freq_weight and use label as a binary proxy.
                if (w.dom_freq_hz !== undefined) {
                    const fw = 1 / (1 + Math.exp(-2.5 * (w.dom_freq_hz - freqThreshold)));
                    // Use the label as raw_prob proxy: 1→1.0, 0→0.0
                    // This is a rough estimate; real raw_prob would need the model
                    // but we don't run inference on server-side data here.
                    // We store 'severity' on upload for a better comparison —
                    // use it if present, else use label-proxy.
                    const rawProb = (w.raw_prob !== undefined) ? w.raw_prob : (w.label || 0);
                    buckets[pos].push(rawProb * fw * 100);
                }
            });
        });

        const nSessions = sessionIds.size;
        if (nSessions < 3) return null; // Not enough data to be meaningful

        const result = { total: nSessions };
        for (const pos of Object.keys(buckets)) {
            const vals = buckets[pos];
            result[pos] = {
                avg: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null,
                n:   vals.length
            };
        }
        return result;
    } catch (e) {
        console.warn('fetchPopulationAverages error:', e);
        return null;
    }
}

// ============================================================
// PARTICIPANT SESSION STATE
// ============================================================
const P_POSITIONS = [
    {
        key:   'Still',
        label: 'Position 1 — Resting',
        short: 'Rest your hand flat on the table with your forearm supported.',
        instruction: 'Place your forearm on the table with your hand lying flat and relaxed. Stay as still as possible for 40 seconds.'
    },
    {
        key:   'Hover',
        label: 'Position 2 — Hover',
        short: 'Hold your forearm just above your knee, elbow slightly bent.',
        instruction: 'Lift your forearm just above your knee without resting it. Keep your elbow slightly unlocked. Hold this position for 40 seconds.'
    },
    {
        key:   'Spoon',
        label: 'Position 3 — Spoon',
        short: 'Raise your hand as if holding a spoon to your mouth.',
        instruction: 'Raise your hand toward your mouth as if holding a spoon, with your elbow pointing outward. Hold this position for 40 seconds.'
    },
    {
        key:   'Point',
        label: 'Position 4 — Extend',
        short: 'Extend your arm outward, above shoulder height.',
        instruction: 'Stretch your arm fully outward in front of you, raised above shoulder level. Hold this position for 40 seconds.'
    }
];

const P_RECORDING_SECONDS = 40;
const P_SKIP_INITIAL      = 6;

let pSession = {
    sessionId:    '',
    diagnosis:    'TREMOR',
    currentPos:   0,
    started:      false,
    complete:     false,
    positionData: [[], [], [], []]
};

function generateSessionId() {
    return 'P-' + Date.now().toString(36).toUpperCase() + '-' +
           Math.random().toString(36).slice(2, 6).toUpperCase();
}

function startParticipantSession() {
    const diag = document.getElementById('p-diagnosis-select').value;
    pSession = {
        sessionId:    generateSessionId(),
        diagnosis:    diag,
        currentPos:   0,
        started:      true,
        complete:     false,
        positionData: [[], [], [], []]
    };

    document.getElementById('p-setup').style.display       = 'none';
    document.getElementById('p-recording').style.display   = 'block';
    document.getElementById('p-report-section').style.display = 'none';
    updateParticipantUI();
}

function updateParticipantUI() {
    const pos   = P_POSITIONS[pSession.currentPos];
    const total = P_POSITIONS.length;

    let dots = '';
    for (let i = 0; i < total; i++) {
        const done   = i < pSession.currentPos;
        const active = i === pSession.currentPos;
        const color  = done ? '#34c759' : active ? '#00c2ff' : '#252f3a';
        const border = active ? '2px solid #00c2ff' : '2px solid transparent';
        dots += `<div style="width:14px;height:14px;border-radius:50%;background:${color};border:${border};"></div>`;
    }
    document.getElementById('p-pos-dots').innerHTML = dots;
    document.getElementById('p-pos-counter').textContent = `Step ${pSession.currentPos + 1} of ${total}`;
    document.getElementById('p-pos-label').textContent   = pos.label;
    document.getElementById('p-pos-instruction').textContent = pos.instruction;

    const btn = document.getElementById('p-record-btn');
    btn.textContent = 'Start Recording';
    btn.disabled    = false;
    document.getElementById('p-countdown').textContent = '';
}

let pAssessmentTimer = null;

function startParticipantRecording() {
    if (isRecording) return;

    // Switch IMU data to participant session
    activeMode = 'participant';
    isRecording = true;

    const btn = document.getElementById('p-record-btn');
    const cd  = document.getElementById('p-countdown');
    btn.disabled = true;

    let rem = P_RECORDING_SECONDS;
    cd.textContent = `${rem}s remaining`; cd.style.color = '#ff9500';

    pAssessmentTimer = setInterval(() => {
        rem--;
        cd.textContent = rem > 0 ? `${rem}s remaining` : 'Finishing…';
        if (rem <= 0) {
            clearInterval(pAssessmentTimer);
            isRecording = false;
            activeMode  = 'researcher'; // reset so researcher mode still works
            finishParticipantPosition();
        }
    }, 1000);
}

function finishParticipantPosition() {
    const posIdx = pSession.currentPos;
    const count  = pSession.positionData[posIdx].length;
    const cd     = document.getElementById('p-countdown');
    cd.textContent = `✓ ${P_POSITIONS[posIdx].key} complete — ${count} windows`;
    cd.style.color = '#34c759';

    pSession.currentPos++;

    if (pSession.currentPos >= P_POSITIONS.length) {
        pSession.complete = true;
        document.getElementById('p-record-btn').textContent = 'All positions complete';
        document.getElementById('p-record-btn').disabled    = true;
        document.getElementById('p-report-section').style.display = 'block';
    } else {
        setTimeout(() => updateParticipantUI(), 800);
    }
}

function resetParticipantSession() {
    isRecording = false;
    clearInterval(pAssessmentTimer);
    activeMode = 'researcher';
    document.getElementById('p-setup').style.display        = 'block';
    document.getElementById('p-recording').style.display    = 'none';
    document.getElementById('p-report-section').style.display = 'none';
}

// ============================================================
// PARTICIPANT IMU ROUTING
// The handleIMU function already calls isRecording.
// We need to route data to pSession when in participant mode.
// Patch: override the push inside handleIMU via mode check.
// ============================================================
// NOTE: handleIMU checks activeMode to decide where to push data.
// We patch it by adding a data router — see the modified handleIMU
// section below. The original handleIMU pushes to session.positionData
// when isRecording && activeMode === 'researcher', and to
// pSession.positionData when activeMode === 'participant'.
// This is handled by replacing the push line in handleIMU.

// ============================================================
// PARTICIPANT DATA CONSENT + UPLOAD
// ============================================================
async function submitAndGenerateReport() {
    const consentEl = document.getElementById('p-consent-check');
    const consented = consentEl && consentEl.checked;

    // Show loading state
    const btn = document.querySelector('#p-report-section .btn-purple');
    if (btn) { btn.textContent = 'Generating…'; btn.disabled = true; }

    let uploaded = false;
    if (consented && db) {
        showToast('Uploading data…');
        uploaded = await uploadToFirestore(pSession.sessionId, 'participant', pSession.positionData);
        if (uploaded) showToast('Data shared ✓ — fetching comparison…');
    }

    // Fetch population averages for comparison (works even without consent)
    let popAverages = null;
    if (db) {
        popAverages = await fetchPopulationAverages();
    }

    if (btn) { btn.textContent = 'Get My Report'; btn.disabled = false; }
    generateParticipantReport(consented, uploaded, popAverages);
}

// ============================================================
// PARTICIPANT REPORT (simplified, plain-language, with charts)
// ============================================================
function generateParticipantReport(consented, uploaded, popAverages) {
    if (!pSession.complete) return;

    const ft          = freqThreshold;
    const isTremor    = pSession.diagnosis === 'TREMOR';
    const statusColor = isTremor ? '#ff4b4b' : '#4bff4b';

    // Per-position stats
    let positionCards = '';
    let allSev = [], allHz = [];
    let overallPeak = 0, overallAvgSevs = [];

    P_POSITIONS.forEach((pos, i) => {
        const rows = pSession.positionData[i].slice(P_SKIP_INITIAL);
        if (!rows.length) return;

        const sevs  = rows.map(r => r.raw_prob * sigmoidReport(r.dom_freq_hz, ft) * 100);
        const hzs   = rows.map(r => r.dom_freq_hz);
        const avg   = sevs.reduce((a,b)=>a+b,0) / sevs.length;
        const peak  = Math.max(...sevs);

        overallPeak = Math.max(overallPeak, peak);
        overallAvgSevs.push(avg);
        allSev.push(...sevs);
        allHz.push(...hzs);

        const barColor = avg > 50 ? '#ff4b4b' : avg > 20 ? '#ff9f1c' : '#34c759';
        const barPct   = Math.min(avg, 100).toFixed(1);
        const label    = avg < 15 ? 'Low activity' : avg < 45 ? 'Mild activity' : avg > 75 ? 'High activity' : 'Moderate activity';

        positionCards += `
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:16px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <span style="font-weight:bold;color:#c9d1d9;">${pos.label}</span>
                <span style="color:${barColor};font-weight:bold;font-size:18px;">${avg.toFixed(1)}%</span>
            </div>
            <div style="height:8px;background:#21262d;border-radius:4px;overflow:hidden;margin-bottom:6px;">
                <div style="height:100%;width:${barPct}%;background:${barColor};border-radius:4px;transition:width .5s;"></div>
            </div>
            <div style="font-size:12px;color:#8b949e;">${label} · Peak: ${peak.toFixed(1)}%</div>
        </div>`;
    });

    const overallAvg = overallAvgSevs.length
        ? overallAvgSevs.reduce((a,b)=>a+b,0) / overallAvgSevs.length
        : 0;

    // Plain-language assessment
    let assessment = '';
    if (overallAvg < 15) {
        assessment = 'Your motion patterns showed <strong>minimal tremor activity</strong> across all four positions. This is consistent with typical voluntary movement.';
    } else if (overallAvg < 45) {
        assessment = 'Your motion patterns showed <strong>mild tremor activity</strong> in some positions. This is worth discussing with a healthcare professional if it persists in daily life.';
    } else if (overallAvg < 75) {
        assessment = 'Your motion patterns showed <strong>moderate tremor activity</strong> across positions. TremorPause detected consistent oscillatory motion characteristic of hand tremor.';
    } else {
        assessment = 'Your motion patterns showed <strong>high tremor activity</strong> across positions. TremorPause detected strong, consistent oscillatory motion in multiple positions.';
    }

    const shareNote = consented && uploaded
        ? `<div style="background:#0f2010;border:1px solid #1a4020;border-radius:8px;padding:12px;font-size:12px;color:#4bff4b;margin-bottom:16px;">✓ Your anonymised data has been shared with the TremorPause research dataset. Session ID: <code style="color:#00d4ff">${pSession.sessionId}</code></div>`
        : `<div style="background:#1a1208;border:1px solid #302010;border-radius:8px;padding:12px;font-size:12px;color:#8b949e;margin-bottom:16px;">Your data was not shared with the dataset.</div>`;

    // Population comparison section
    let comparisonSection = '';
    if (popAverages && popAverages.total >= 3) {
        const nSessions = popAverages.total;
        let compRows = '';
        P_POSITIONS.forEach((pos, i) => {
            const myRows = pSession.positionData[i].slice(P_SKIP_INITIAL);
            if (!myRows.length) return;
            const mySevs = myRows.map(r => r.raw_prob * sigmoidReport(r.dom_freq_hz, ft) * 100);
            const myAvg  = mySevs.reduce((a,b)=>a+b,0) / mySevs.length;
            const pop    = popAverages[pos.key];
            if (!pop || pop.avg === null) return;

            const diff      = myAvg - pop.avg;
            const diffColor = diff > 10 ? '#ff4b4b' : diff < -10 ? '#4bff4b' : '#8b949e';
            const diffText  = diff > 0 ? `+${diff.toFixed(1)}%` : `${diff.toFixed(1)}%`;
            const myBarPct  = Math.min(myAvg, 100);
            const popBarPct = Math.min(pop.avg, 100);
            const myColor   = myAvg > 50 ? '#ff4b4b' : myAvg > 20 ? '#ff9f1c' : '#34c759';
            const popColor  = pop.avg > 50 ? '#ff4b4b' : pop.avg > 20 ? '#ff9f1c' : '#34c759';

            compRows += `
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <span style="font-weight:bold;color:#c9d1d9;font-size:13px;">${pos.label}</span>
                    <span style="color:${diffColor};font-size:12px;font-weight:bold;">${diffText} vs average</span>
                </div>
                <div style="display:grid;grid-template-columns:60px 1fr 44px;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="font-size:11px;color:#8b949e;">You</span>
                    <div style="height:7px;background:#21262d;border-radius:4px;overflow:hidden;">
                        <div style="height:100%;width:${myBarPct}%;background:${myColor};border-radius:4px;"></div>
                    </div>
                    <span style="font-size:12px;color:${myColor};font-weight:bold;text-align:right;">${myAvg.toFixed(1)}%</span>
                </div>
                <div style="display:grid;grid-template-columns:60px 1fr 44px;align-items:center;gap:8px;">
                    <span style="font-size:11px;color:#8b949e;">Average</span>
                    <div style="height:7px;background:#21262d;border-radius:4px;overflow:hidden;">
                        <div style="height:100%;width:${popBarPct}%;background:${popColor};border-radius:4px;opacity:0.5;"></div>
                    </div>
                    <span style="font-size:12px;color:#8b949e;text-align:right;">${pop.avg.toFixed(1)}%</span>
                </div>
            </div>`;
        });

        comparisonSection = `
        <h2>How You Compare</h2>
        <p style="font-size:12px;color:#8b949e;margin-bottom:14px;line-height:1.6;">
            Compared against ${nSessions} anonymised sessions in the TremorPause dataset.
            "Average" is the mean severity across all participants for that position.
        </p>
        ${compRows}`;
    } else if (db) {
        comparisonSection = `
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px;margin-bottom:20px;font-size:12px;color:#8b949e;line-height:1.6;">
            Dataset comparison will appear here once more participants have completed sessions.
        </div>`;
    }

    // Chart data
    const chartData = JSON.stringify({ sev: allSev, hz: allHz });

    const reportHTML = `<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TremorPause — Your Results</title>
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"><\/script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0b0e14;color:#c9d1d9;font-family:'Segoe UI',sans-serif;padding:30px 20px;max-width:680px;margin:0 auto;}
h1{color:#fff;margin-bottom:4px;font-size:24px;}
h2{color:#c9d1d9;font-size:16px;margin:20px 0 10px;}
.hero{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:24px;margin:20px 0;text-align:center;}
.hero-num{font-size:52px;font-weight:bold;}
.hero-sub{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
.assessment{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;margin-bottom:20px;line-height:1.7;font-size:15px;}
.disclaimer{font-size:11px;color:#4a5060;line-height:1.6;margin-top:24px;padding-top:16px;border-top:1px solid #21262d;}
code{background:#1c2128;padding:2px 5px;border-radius:3px;font-size:12px;}
</style>
</head><body>
<h1>TremorPause</h1>
<p style="color:#8b949e;margin-bottom:16px">Your Personal Motion Assessment</p>

${shareNote}

<div class="hero">
    <div class="hero-num" style="color:${statusColor}">${overallAvg.toFixed(1)}%</div>
    <div class="hero-sub">Average AI Severity Score</div>
    <div style="font-size:13px;color:#8b949e;margin-top:8px;">Peak: ${overallPeak.toFixed(1)}%</div>
</div>

<div class="assessment">
    <strong style="color:#fff;display:block;margin-bottom:8px;">What does this mean?</strong>
    <p>${assessment}</p>
    <p style="margin-top:10px;font-size:13px;color:#8b949e;">
        The AI Severity Score reflects how closely your motion patterns matched those of participants
        with diagnosed tremor in the TremorPause training dataset.
        A higher score indicates more tremor-like oscillatory motion — it is <em>not</em> a clinical diagnosis.
    </p>
</div>

<h2>Results by Position</h2>
${positionCards}
${comparisonSection}
<h2>Motion Frequency &amp; Severity Charts</h2>
<div id="charts" style="margin-top:10px;"></div>

<div class="disclaimer">
    <strong>Important:</strong> This report is generated by an AI model and is for informational purposes only.
    It is not a medical diagnosis. If you have concerns about tremor or movement disorders,
    please consult a qualified healthcare professional.<br><br>
    Session ID: <code>${pSession.sessionId}</code>
</div>

<script>
(function(){
var d = ${chartData};
var fig = {
data:[
{type:'histogram', x:d.sev, nbinsx:20,
 marker:{color:'#00c2ff', opacity:0.8},
 name:'Severity'},
{type:'histogram', x:d.hz,  nbinsx:25,
 marker:{color:'#ff9f1c', opacity:0.8},
 name:'Frequency', xaxis:'x2', yaxis:'y2'}
],
layout:{
template:'plotly_dark',
height:320,
showlegend:false,
margin:{l:40,r:20,t:30,b:40},
grid:{rows:1, columns:2, pattern:'independent'},
annotations:[
{text:'AI Severity Distribution', xref:'paper',yref:'paper',x:0.18,y:1.08,showarrow:false,font:{color:'#c9d1d9',size:12}},
{text:'Frequency Distribution (Hz)',xref:'paper',yref:'paper',x:0.82,y:1.08,showarrow:false,font:{color:'#c9d1d9',size:12}}
],
xaxis:{title:'Severity (%)',domain:[0,0.45]},
yaxis:{domain:[0,1]},
xaxis2:{title:'Frequency (Hz)',domain:[0.55,1]},
yaxis2:{domain:[0,1]}
}
};
Plotly.newPlot('charts', fig.data, fig.layout);
})();
<\/script>
</body></html>`;

    const blob = new Blob([reportHTML], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    window.open(url, '_blank');
    setTimeout(() => URL.revokeObjectURL(url), 120000);
}

