// CONFIGURATION MAPPING
const DEVICEMAP = {
    left: {
        service: '12345678-1234-5678-1234-56789abcdef0',
        char: 'abcdef01-1234-5678-1234-56789abcdef0',
        motor: 'abcdef02-1234-5678-1234-56789abcdef0'
    },
    right: {
        name: 'TREMOR-PRO-X1',
        service: '19b10000-e8f2-537e-4f6c-d104768a1214',
        char: '19b10001-e8f2-537e-4f6c-d104768a1214',
        motor: null // Add if the Pro-X1 has a separate motor characteristic
    }
};

let model = null;
let sessionData = [];
let isRecording = false;
let buffers = { left: [], right: [] };
let devices = { left: null, right: null };

// 1. Load the Model you exported to JSON
fetch('TremorModel.json')
    .then(response => response.json())
    .then(data => { model = data; console.log("AI Model Loaded Successfully"); })
    .catch(err => alert("Model Load Error: " + err));

// 2. Bluetooth Management
async function connectBluetooth(side) {
    const config = DEVICEMAP[side];
    try {
        const device = await navigator.bluetooth.requestDevice({
            filters: config.name ? [{ name: config.name }] : [{ services: [config.service] }],
            optionalServices: [config.service]
        });
        
        const server = await device.gatt.connect();
        const service = await server.getPrimaryService(config.service);
        const char = await service.getCharacteristic(config.char);
        
        devices[side] = { server, service, char };
        document.getElementById(`${side}-status`).innerText = `${side.toUpperCase()}: Active`;
        document.getElementById(`${side}-status`).style.color = "#34c759";

        await char.startNotifications();
        char.addEventListener('characteristicvaluechanged', (event) => handleIMU(event, side));
    } catch (error) {
        console.error(error);
        alert(`Connection to ${side} failed.`);
    }
}

// 3. Signal Processing (Python extract_features equivalent)
function handleIMU(event, side) {
    const value = new TextDecoder().decode(event.target.value);
    const parts = value.split(',').map(Number);
    if (parts.length < 3) return;

    // Magnitude Calculation
    const mag = Math.sqrt(parts[0]**2 + parts[1]**2 + parts[2]**2);
    buffers[side].push(mag);

    // Windowing (WINDOW_SIZE = 128, STEP_SIZE = 32)
    if (buffers[side].length >= 128) {
        const window = buffers[side].slice(-128);
        const features = calculateFeatures(window);
        const severity = runInference(features);

        updateUI(side, severity, features[4]);
        
        if (isRecording) {
            sessionData.push({ side, timestamp: Date.now(), severity, ...features });
        } else {
            sendMotorFeedback(side, severity);
        }

        buffers[side] = buffers[side].slice(32); // Slide window
    }
}

function calculateFeatures(w) {
    const mean = w.reduce((a, b) => a + b) / w.length;
    const std = Math.sqrt(w.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / w.length);
    const max = Math.max(...w);
    const energy = w.reduce((a, b) => a + b * b, 0);
    
    // Simple Dominant Freq (Zero-crossing approximation for Hz)
    let crossings = 0;
    for(let i=1; i<w.length; i++) if((w[i]-mean)*(w[i-1]-mean) < 0) crossings++;
    const hz = (crossings / 2) * (1 / (w.length * 0.02)); // Assuming ~50Hz sampling

    return [mean, std, max, energy, hz, 0.1]; // 0.1 is placeholder for vibration_rate
}

// 4. ML Inference Engine
function runInference(features) {
    if (!model) return 0;
    let totalProb = 0;
    
    model.forEach(tree => {
        let node = 0;
        while (tree.feature[node] !== -2) {
            let fIdx = tree.feature[node];
            node = (features[fIdx] <= tree.threshold[node]) ? tree.children_left[node] : tree.children_right[node];
        }
        // Prob of tremor (Class 1)
        const leafValues = tree.values[node][0];
        totalProb += (leafValues[1] > leafValues[0] ? 1 : 0);
    });
    
    return (totalProb / model.length) * 100;
}

// 5. Motor Control (sendMotorFeedback)
async function sendMotorFeedback(side, severity) {
    const config = DEVICEMAP[side];
    if (!config.motor || !devices[side]) return;

    try {
        const motorChar = await devices[side].service.getCharacteristic(config.motor);
        const val = Math.floor(Math.min(severity, 100));
        await motorChar.writeValueWithoutResponse(new Uint8Array([val]));
    } catch (e) { /* Fail silently to maintain loop speed */ }
}

function updateUI(side, sev, hz) {
    document.getElementById(`${side}-sev`).innerText = `${sev.toFixed(1)}%`;
    document.getElementById(`${side}-meta`).innerText = `Hz: ${hz.toFixed(1)} | Motor: ${sev > 15 ? 'ON' : 'IDLE'}`;
    
    const bar = document.getElementById(`${side}-bar`);
    bar.style.width = `${sev}%`;
    bar.style.background = sev > 75 ? "#ff3b30" : (sev > 35 ? "#ff9500" : "#34c759");
}

function toggleAssessment() {
    isRecording = true;
    document.getElementById('record-btn').innerText = "RECORDING...";
    setTimeout(() => {
        isRecording = false;
        document.getElementById('record-btn').innerText = "Recording Complete";
        document.getElementById('report-link').style.display = "block";
    }, 10000);
}

function downloadReport() {
    const csvContent = "data:text/csv;charset=utf-8," 
        + "side,timestamp,severity,mean,std,max,energy,hz\n"
        + sessionData.map(e => Object.values(e).join(",")).join("\n");
    window.open(encodeURI(csvContent));
}