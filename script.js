let model;
let eyeClosedStart = null;
const threshold = 3;

async function setup() {
  model = await tf.loadLayersModel('web_model/model.json');
  document.getElementById("status").innerText = "Model Loaded";

  const webcam = document.getElementById("webcam");
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcam.srcObject = stream;

  webcam.addEventListener("loadeddata", detect);
}

async function detect() {
  const webcam = document.getElementById("webcam");
  const status = document.getElementById("status");
  const alarm = document.getElementById("alarm");

  const tensor = tf.browser.fromPixels(webcam)
    .resizeNearestNeighbor([24, 24])
    .mean(2)
    .expandDims(0)
    .expandDims(-1)
    .div(255.0);

  const prediction = await model.predict(tensor).data();

  if (prediction[0] < 0.5) {
    status.innerText = "EYES CLOSED";
    if (!eyeClosedStart) eyeClosedStart = Date.now();
    else if ((Date.now() - eyeClosedStart)/1000 > threshold) {
      alarm.play();
    }
  } else {
    status.innerText = "EYES OPEN";
    eyeClosedStart = null;
  }

  requestAnimationFrame(detect);
}

setup();