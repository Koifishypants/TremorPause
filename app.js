// ============================================================
// TremorPause AI — app.js
// Faithful port of tremorpredictorv4.py + tremordatav3.py
// ============================================================

// ---------- CONFIG ----------
const WINDOW_SIZE   = 128;
const STEP_SIZE     = 32;
const CALIB_SAMPLES = 50;
const FEATURES      = ['mean_mag','std_mag','max_mag','energy','dom_freq_hz','vibration_rate'];

// Default sigmoid center (ft). In Python this is computed from training data
// via get_optimal_hz_via_j_index(). Set this to match your trained model's
// characteristic tremor frequency. Can be changed in the Settings panel.
let freqThreshold = 4.0;

const DEVICEMAP = {
    left: {
        label:   'Left Hand',
        service: '12345678-1234-5678-1234-56789abcdef0',
        char:    'abcdef01-1234-5678-1234-56789abcdef0',
        motor:   'abcdef02-1234-5678-1234-56789abcdef0'
    },
    right: {
        label:   'Right Hand (Pro-X1)',
        name:    'TREMOR-PRO-X1',
        service: '19b10000-e8f2-537e-4f6c-d104768a1214',
        char:    '19b10001-e8f2-537e-4f6c-d104768a1214',
        motor:   null   // add UUID here if Pro-X1 has motor output
    }
};

// ---------- GLOBAL STATE ----------
let model         = null;
let sessionData   = [];
let isRecording   = false;

// Per-side state — mirrors LiveTremorPredictor in Python
const sideState = {
    left:  makeSideState(),
    right: makeSideState()
};

function makeSideState() {
    return {
        device:       null,
        gattService:  null,
        buffer:       [],       // [[ax,ay,az], ...]
        timestamps:   [],       // ms
        calibBuf:     [],       // raw xyz during calibration
        bias:         [0,0,0],
        calibrated:   false,
        connected:    false,
        peakSeverity: 0
    };
}

// ============================================================
// 1. MODEL LOADING
// ============================================================
function loadModel() {
    setModelStatus('Loading model…', '#ff9500');
    fetch('TremorModel.json')
        .then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        })
        .then(data => {
            model = data;
            const treeCount = Array.isArray(data) ? data.length : '?';
            setModelStatus(`✓ Model ready — ${treeCount} trees`, '#34c759');
        })
        .catch(err => {
            setModelStatus(`✗ Model load failed: ${err.message}`, '#ff3b30');
        });
}

// ============================================================
// 2. FFT (Cooley-Tukey, radix-2, in-place)
//    Matches np.fft.fft used in Python for dom_freq extraction
// ============================================================
function computeFFTMagnitudes(signal) {
    const N  = signal.length;
    const re = new Float64Array(signal);
    const im = new Float64Array(N);

    // Bit-reversal
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }

    // Butterfly
    for (let len = 2; len <= N; len <<= 1) {
        const ang  = -2 * Math.PI / len;
        const wRe  = Math.cos(ang);
        const wIm  = Math.sin(ang);
        for (let i = 0; i < N; i += len) {
            let cRe = 1, cIm = 0;
            for (let k = 0; k < len / 2; k++) {
                const uRe = re[i+k],         uIm = im[i+k];
                const vRe = re[i+k+len/2] * cRe - im[i+k+len/2] * cIm;
                const vIm = re[i+k+len/2] * cIm + im[i+k+len/2] * cRe;
                re[i+k]         = uRe + vRe;
                im[i+k]         = uIm + vIm;
                re[i+k+len/2]   = uRe - vRe;
                im[i+k+len/2]   = uIm - vIm;
                const newCRe = cRe*wRe - cIm*wIm;
                cIm = cRe*wIm + cIm*wRe;
                cRe = newCRe;
            }
        }
    }

    // Return one-sided magnitude spectrum
    const mag = new Float64Array(N / 2);
    for (let i = 0; i < N / 2; i++) {
        mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);
    }
    return mag;
}

// ============================================================
// 3. FEATURE EXTRACTION
//    Direct port of extract_features() in tremordatav3.py
// ============================================================
function extractFeatures(rawWindow, timestamps) {
    const n = rawWindow.length;

    // Actual sampling frequency (mirrors: actual_fs = WINDOW_SIZE / time_diff)
    const timeDiffSec = (timestamps[n-1] - timestamps[0]) / 1000.0;
    const actualFs    = timeDiffSec > 0 ? n / timeDiffSec : 50;

    // 1. Magnitude
    const mag = rawWindow.map(([ax, ay, az]) => Math.sqrt(ax*ax + ay*ay + az*az));

    const meanVal = mag.reduce((s, v) => s + v, 0) / n;
    const stdVal  = Math.sqrt(mag.map(v => (v-meanVal)**2).reduce((s, v) => s + v, 0) / n);
    const maxVal  = Math.max(...mag);
    const energy  = mag.reduce((s, v) => s + v*v, 0);

    // 2. Stillness guard (matches Python: mean_val < 0.05 or std < 0.01)
    if (meanVal < 0.05 || stdVal < 0.01) {
        return { features: [meanVal, 0, maxVal, 0, 0, 0], actualFs };
    }

    // 3. FFT on mean-centered signal (matches: sig = mag - mean_val)
    const sig     = mag.map(v => v - meanVal);
    const fftMag  = computeFFTMagnitudes(sig);

    // 4. Dominant frequency in positive range (matches: freqs = fftfreq(n, 1/fs))
    let domFreq = 0, maxAmp = -Infinity;
    for (let i = 1; i < fftMag.length; i++) {
        const freq = i * actualFs / n;
        if (freq >= 1 && freq <= 20 && fftMag[i] > maxAmp) {
            maxAmp  = fftMag[i];
            domFreq = freq;
        }
    }

    // 5. Vibration rate — zero-crossings of mean-centred signal
    //    matches: np.where(np.diff(np.sign(sig)))[0].size
    let crossings = 0;
    for (let i = 1; i < sig.length; i++) {
        if (sig[i-1] * sig[i] < 0) crossings++;
    }

    return {
        features: [meanVal, stdVal, maxVal, energy, domFreq, crossings],
        actualFs
    };
}

// ============================================================
// 4. ML INFERENCE — predict_proba
//    Matches pklConverter.py JSON format:
//    model = [ tree, ... ]   tree = [ {id,feature,threshold,left,right,value}, ... ]
//    Returns probability of class index 1 (tremor), 0-1
// ============================================================
function predictProba(features) {
    if (!model || !model.length) return 0;

    let totalTremorProb = 0;

    for (const nodes of model) {
        // Traverse tree
        let nodeId = 0;
        while (nodes[nodeId].feature !== -2) {
            const { feature, threshold, left, right } = nodes[nodeId];
            nodeId = features[feature] <= threshold ? left : right;
        }
        // Leaf: value = [[n_class0, n_class1]]
        const leafValues = nodes[nodeId].value[0];
        const total      = leafValues.reduce((a, b) => a + b, 0);
        totalTremorProb += total > 0 ? leafValues[1] / total : 0;
    }

    return totalTremorProb / model.length;
}

// ============================================================
// 5. SIGMOID GATE
//    Matches: freq_weight = 1 / (1 + np.exp(-1.0 * (hz - ft)))
//             severity_pct = raw_severity * freq_weight * 100
// ============================================================
function sigmoidGate(hz, ft) {
    return 1.0 / (1.0 + Math.exp(-1.0 * (hz - ft)));
}

// ============================================================
// 6. IMU DATA HANDLER  (notification_handler equivalent)
// ============================================================
function handleIMU(event, side) {
    const decoded = new TextDecoder().decode(event.target.value).trim();
    const parts   = decoded.split(',').map(Number);
    if (parts.length < 3 || parts.some(isNaN)) return;

    const s   = sideState[side];
    const now = Date.now();
    const xyz = parts.slice(0, 3);

    // ---- CALIBRATION PHASE (first 50 samples) ----
    if (!s.calibrated) {
        s.calibBuf.push(xyz);
        const pct = Math.round((s.calibBuf.length / CALIB_SAMPLES) * 100);
        setConnStatus(side, `Calibrating… ${pct}%`, '#ff9500');

        if (s.calibBuf.length >= CALIB_SAMPLES) {
            s.bias = [0, 1, 2].map(
                i => s.calibBuf.reduce((sum, r) => sum + r[i], 0) / CALIB_SAMPLES
            );
            s.calibrated = true;
            s.calibBuf   = [];
            setConnStatus(side, 'Connected ✓', '#34c759');
            setCalibBadge(side, true);
        }
        return;
    }

    // ---- BIAS CORRECTION ----
    const corrected = xyz.map((v, i) => v - s.bias[i]);
    s.buffer.push(corrected);
    s.timestamps.push(now);

    // ---- WINDOWED INFERENCE ----
    if (s.buffer.length >= WINDOW_SIZE) {
        const winXYZ  = s.buffer.slice(-WINDOW_SIZE);
        const winTime = s.timestamps.slice(-WINDOW_SIZE);

        const { features, actualFs } = extractFeatures(winXYZ, winTime);
        const hz          = features[4];
        const rawProb     = predictProba(features);
        const freqWeight  = sigmoidGate(hz, freqThreshold);
        const severity    = rawProb * freqWeight * 100;

        if (severity > s.peakSeverity) s.peakSeverity = severity;

        // Update display
        updateSideUI(side, severity, hz, actualFs);

        // Recording vs. live motor control
        if (isRecording) {
            const position = document.getElementById('position-select').value;
            sessionData.push({
                side, timestamp: now, position, severity,
                mean_mag:       features[0].toFixed(4),
                std_mag:        features[1].toFixed(4),
                max_mag:        features[2].toFixed(4),
                energy:         features[3].toFixed(2),
                dom_freq_hz:    features[4].toFixed(2),
                vibration_rate: features[5],
                raw_prob:       rawProb.toFixed(4),
                freq_weight:    freqWeight.toFixed(4),
                actual_fs:      actualFs.toFixed(1)
            });
        } else {
            sendMotorFeedback(side, severity);
        }

        // Slide window (STEP_SIZE = 32, 75% overlap)
        s.buffer     = s.buffer.slice(STEP_SIZE);
        s.timestamps = s.timestamps.slice(STEP_SIZE);
    }
}

// ============================================================
// 7. BLUETOOTH MANAGEMENT
// ============================================================
async function connectBluetooth(side) {
    const cfg = DEVICEMAP[side];
    const btn = document.getElementById(`${side}-conn-btn`);
    try {
        btn.disabled = true;
        setConnStatus(side, 'Requesting…', '#ff9500');

        const device = await navigator.bluetooth.requestDevice({
            filters:          cfg.name ? [{ name: cfg.name }] : [{ services: [cfg.service] }],
            optionalServices: [cfg.service]
        });

        const server  = await device.gatt.connect();
        const service = await server.getPrimaryService(cfg.service);
        const charObj = await service.getCharacteristic(cfg.char);

        const s       = sideState[side];
        s.device      = device;
        s.gattService = service;
        s.connected   = true;

        setConnStatus(side, 'Calibrating…', '#ff9500');
        btn.textContent = 'Disconnect';
        btn.disabled    = false;
        btn.onclick     = () => disconnectBluetooth(side);

        await charObj.startNotifications();
        charObj.addEventListener('characteristicvaluechanged', e => handleIMU(e, side));

        device.addEventListener('gattserverdisconnected', () => onDisconnect(side, btn));

    } catch (err) {
        console.error(err);
        setConnStatus(side, `Failed — ${err.message.slice(0, 40)}`, '#ff3b30');
        btn.disabled = false;
    }
}

async function disconnectBluetooth(side) {
    const s = sideState[side];
    if (s.device && s.device.gatt.connected) {
        await s.device.gatt.disconnect();
    }
    onDisconnect(side, document.getElementById(`${side}-conn-btn`));
}

function onDisconnect(side, btn) {
    const s       = sideState[side];
    s.connected   = false;
    s.calibrated  = false;
    s.buffer      = [];
    s.timestamps  = [];
    s.calibBuf    = [];
    s.bias        = [0, 0, 0];
    s.peakSeverity = 0;

    setConnStatus(side, 'Disconnected', '#ff3b30');
    setCalibBadge(side, false);
    btn.textContent = 'Connect';
    btn.onclick     = () => connectBluetooth(side);
    btn.disabled    = false;
    resetSideUI(side);
}

// ============================================================
// 8. MOTOR FEEDBACK
// ============================================================
async function sendMotorFeedback(side, severity) {
    const cfg = DEVICEMAP[side];
    if (!cfg.motor) return;
    const s = sideState[side];
    if (!s.gattService) return;
    try {
        const mc  = await s.gattService.getCharacteristic(cfg.motor);
        const val = Math.min(Math.floor(severity), 100);
        await mc.writeValueWithoutResponse(new Uint8Array([val]));
    } catch (_) { /* fail silently */ }
}

// ============================================================
// 9. CLINICAL ASSESSMENT MODE
// ============================================================
let assessmentInterval = null;

function startAssessment() {
    if (isRecording) return;
    sessionData   = [];
    isRecording   = true;
    const btn     = document.getElementById('record-btn');
    const cd      = document.getElementById('countdown');
    const section = document.getElementById('report-section');
    section.style.display = 'none';
    btn.disabled  = true;

    let remaining = 10;
    cd.textContent = `${remaining}s remaining`;
    cd.style.color = '#ff9500';

    assessmentInterval = setInterval(() => {
        remaining--;
        cd.textContent = remaining > 0 ? `${remaining}s remaining` : 'Finishing…';
        if (remaining <= 0) {
            clearInterval(assessmentInterval);
            isRecording           = false;
            btn.disabled          = false;
            cd.textContent        = `✓ Done — ${sessionData.length} windows captured`;
            cd.style.color        = '#34c759';
            section.style.display = 'block';
            renderSummary();
        }
    }, 1000);
}

function renderSummary() {
    if (!sessionData.length) return;
    const severities = sessionData.map(d => parseFloat(d.severity));
    const avg = severities.reduce((a,b) => a+b, 0) / severities.length;
    const max = Math.max(...severities);
    const label = avg < 15 ? 'STABLE' : avg < 45 ? 'MILD' : avg < 75 ? 'MODERATE' : 'SEVERE';
    document.getElementById('summary-text').textContent =
        `Avg: ${avg.toFixed(1)}%  |  Peak: ${max.toFixed(1)}%  |  Classification: ${label}`;
}

function downloadReport() {
    if (!sessionData.length) return;
    const header = Object.keys(sessionData[0]).join(',') + '\n';
    const rows   = sessionData.map(r => Object.values(r).join(',')).join('\n');
    const blob   = new Blob([header + rows], { type: 'text/csv' });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement('a');
    a.href       = url;
    a.download   = `TremorPause_${sessionData[0]?.position ?? 'session'}_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================================
// 10. UI HELPERS
// ============================================================
const STATUS_MAP = [
    { max: 15,  label: 'STABLE',   color: '#34c759' },
    { max: 45,  label: 'MILD',     color: '#ff9500' },
    { max: 75,  label: 'MODERATE', color: '#ff6b00' },
    { max: 101, label: 'SEVERE',   color: '#ff3b30' }
];

function severityMeta(sev) {
    return STATUS_MAP.find(s => sev < s.max) || STATUS_MAP[STATUS_MAP.length - 1];
}

function updateSideUI(side, severity, hz, fs) {
    const meta = severityMeta(severity);

    const sevEl   = document.getElementById(`${side}-sev`);
    const labelEl = document.getElementById(`${side}-label`);
    const barEl   = document.getElementById(`${side}-bar`);
    const hzEl    = document.getElementById(`${side}-hz`);
    const motorEl = document.getElementById(`${side}-motor`);
    const fsEl    = document.getElementById(`${side}-fs`);
    const peakEl  = document.getElementById(`${side}-peak`);

    sevEl.textContent   = `${severity.toFixed(1)}%`;
    labelEl.textContent = meta.label;
    labelEl.style.color = meta.color;
    barEl.style.width   = `${Math.min(severity, 100)}%`;
    barEl.style.background = meta.color;
    hzEl.textContent    = `${hz.toFixed(1)} Hz`;
    motorEl.textContent = severity > 15 ? 'ACTIVE' : 'IDLE';
    motorEl.style.color = severity > 15 ? '#ff9500' : '#8e8e93';
    if (fsEl)  fsEl.textContent  = `${fs.toFixed(0)} Hz`;
    if (peakEl) peakEl.textContent = `${sideState[side].peakSeverity.toFixed(1)}%`;
}

function resetSideUI(side) {
    ['sev','hz'].forEach(k => {
        const el = document.getElementById(`${side}-${k}`);
        if (el) el.textContent = '--';
    });
    const bar = document.getElementById(`${side}-bar`);
    if (bar) { bar.style.width = '0%'; bar.style.background = '#34c759'; }
    const label = document.getElementById(`${side}-label`);
    if (label) { label.textContent = '--'; label.style.color = '#8e8e93'; }
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
// 11. SETTINGS
// ============================================================
function applyFreqThreshold() {
    const val = parseFloat(document.getElementById('freq-input').value);
    if (isNaN(val) || val <= 0 || val > 20) {
        alert('Enter a value between 0.1 and 20 Hz');
        return;
    }
    freqThreshold = val;
    document.getElementById('threshold-display').textContent = `${val.toFixed(1)} Hz`;
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('DOMContentLoaded', () => {
    loadModel();
    document.getElementById('freq-input').value = freqThreshold;
    document.getElementById('threshold-display').textContent = `${freqThreshold.toFixed(1)} Hz`;
});
