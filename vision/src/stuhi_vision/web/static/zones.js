// Drawing a doorframe zone on a camera's live frame.
//
// Coordinates are stored as fractions of the frame, never pixels: the image is displayed at
// whatever width the page gives it, and the pipeline works at the camera's own resolution.
// Converting at the edges keeps every threshold resolution-independent.

const REFRESH_MS = 4000;

async function load() {
  const response = await fetch('/api/cameras');
  if (!response.ok) return;
  const { cameras } = await response.json();
  const host = document.getElementById('cameras');

  for (const camera of cameras) {
    let section = host.querySelector(`[data-camera="${camera.name}"]`);
    if (!section) {
      section = build(camera);
      host.append(section);
    }
    describe(section, camera);
  }
}

function build(camera) {
  const fragment = document.getElementById('camera-template').content.cloneNode(true);
  const section = fragment.querySelector('.camera');
  section.dataset.camera = camera.name;
  section.querySelector('.name').textContent = camera.name;

  const shot = section.querySelector('.shot');
  const image = section.querySelector('.live');
  const box = section.querySelector('.zone-box');
  const save = section.querySelector('.save');
  refresh(image, camera.name);

  let start = null;
  let drawn = null;

  const asFraction = (event) => {
    const bounds = image.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height,
    };
  };

  const render = (a, b) => {
    const left = Math.min(a.x, b.x);
    const top = Math.min(a.y, b.y);
    box.hidden = false;
    box.style.left = `${left * 100}%`;
    box.style.top = `${top * 100}%`;
    box.style.width = `${Math.abs(b.x - a.x) * 100}%`;
    box.style.height = `${Math.abs(b.y - a.y) * 100}%`;
    section.querySelector('.numbers').textContent =
      `zone = [${left.toFixed(2)}, ${top.toFixed(2)}, ` +
      `${Math.max(a.x, b.x).toFixed(2)}, ${Math.max(a.y, b.y).toFixed(2)}]`;
  };

  shot.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    start = asFraction(event);
    drawn = null;
    save.disabled = true;
  });
  shot.addEventListener('pointermove', (event) => {
    if (start) render(start, asFraction(event));
  });
  shot.addEventListener('pointerup', (event) => {
    if (!start) return;
    drawn = { a: start, b: asFraction(event) };
    start = null;
    render(drawn.a, drawn.b);
    save.disabled = false;
  });

  save.addEventListener('click', () => drawn && store(section, camera.name, drawn));
  section.querySelector('.refresh').addEventListener('click', () =>
    refresh(image, camera.name)
  );
  return section;
}

function refresh(image, camera) {
  // The endpoint sends no-store, but a cache-busting parameter also defeats any proxy.
  image.src = `/frame/${camera}.jpg?t=${Date.now()}`;
}

function describe(section, camera) {
  const state = section.querySelector('.state');
  if (camera.moved) {
    state.className = 'state bad';
    state.textContent = `This camera seems to have moved (${camera.shift}). Redraw its zone.`;
  } else if (!camera.has_reference) {
    state.className = 'state';
    state.textContent = 'No zone saved yet, so movement is not being watched for.';
  } else {
    state.className = 'state ok';
    const drift = camera.shift_px === null ? 'no reading yet' : `${camera.shift_px} px off`;
    state.textContent = `Zone saved, view steady (${drift}).`;
  }

  const box = section.querySelector('.zone-box');
  if (camera.zone && box.hidden) {
    const [x1, y1, x2, y2] = camera.zone;
    box.hidden = false;
    box.style.left = `${x1 * 100}%`;
    box.style.top = `${y1 * 100}%`;
    box.style.width = `${(x2 - x1) * 100}%`;
    box.style.height = `${(y2 - y1) * 100}%`;
    section.querySelector('.numbers').textContent =
      `saved zone = [${x1.toFixed(2)}, ${y1.toFixed(2)}, ${x2.toFixed(2)}, ${y2.toFixed(2)}]`;
  }
}

async function store(section, camera, drawn) {
  const result = section.querySelector('.result');
  result.className = 'result';
  result.textContent = 'saving…';
  try {
    const response = await fetch('/api/zone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        camera,
        x1: drawn.a.x,
        y1: drawn.a.y,
        x2: drawn.b.x,
        y2: drawn.b.y,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      result.className = 'result bad';
      result.textContent = body.detail || 'failed';
      return;
    }
    result.className = 'result ok';
    // Being explicit: the running pipeline built its detector at startup, so it keeps the
    // old zone until it restarts. Letting somebody assume otherwise would waste their time.
    result.textContent = 'Saved, and this frame is now the reference. '
      + 'It takes effect when the pipeline next restarts.';
  } catch (error) {
    result.className = 'result bad';
    result.textContent = String(error);
  }
}

load();
setInterval(load, REFRESH_MS);
