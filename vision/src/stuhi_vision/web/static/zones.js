// Drawing a doorframe zone on a camera's live frame.
//
// Coordinates are stored as fractions of the frame, never pixels: the image is displayed at
// whatever width the page gives it, and the pipeline works at the camera's own resolution.
// Converting at the edges keeps every threshold resolution-independent.

const REFRESH_MS = 4000;
// The server refuses a zone smaller than this; the page has to agree, or a stray click looks
// like a drawn zone and only fails on save.
const MINIMUM = 0.01;

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

/** Put a rectangle on the picture, or clear it when given nothing. */
function paint(section, zone, label) {
  const box = section.querySelector('.zone-box');
  const numbers = section.querySelector('.numbers');
  if (!zone) {
    box.hidden = true;
    box.removeAttribute('style');
    numbers.textContent = '';
    return;
  }
  const [x1, y1, x2, y2] = zone;
  box.hidden = false;
  box.style.left = `${x1 * 100}%`;
  box.style.top = `${y1 * 100}%`;
  box.style.width = `${(x2 - x1) * 100}%`;
  box.style.height = `${(y2 - y1) * 100}%`;
  numbers.textContent = `${label} = [${zone.map((v) => v.toFixed(2)).join(', ')}]`;
}

/** Two corners in any order as a left-top-right-bottom rectangle. */
function rectangle(a, b) {
  return [Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x, b.x), Math.max(a.y, b.y)];
}

function build(camera) {
  const fragment = document.getElementById('camera-template').content.cloneNode(true);
  const section = fragment.querySelector('.camera');
  section.dataset.camera = camera.name;
  section.querySelector('.name').textContent = camera.name;

  const shot = section.querySelector('.shot');
  const image = section.querySelector('.live');
  const save = section.querySelector('.save');
  refresh(image, camera.name);

  let start = null;
  let drawn = null;

  // Clamped, because a drag that leaves the picture is a real gesture: people overshoot the
  // edge on purpose when the doorframe runs right up to it. Losing that drag would be worse
  // than treating it as "to the edge", which is what they meant.
  const asFraction = (event) => {
    const bounds = image.getBoundingClientRect();
    const within = (value) => Math.max(0, Math.min(1, value));
    return {
      x: within((event.clientX - bounds.left) / bounds.width),
      y: within((event.clientY - bounds.top) / bounds.height),
    };
  };

  const drawable = (a, b) => Math.abs(b.x - a.x) >= MINIMUM && Math.abs(b.y - a.y) >= MINIMUM;

  // Whether a rectangle is being drawn or is drawn but unsaved. Kept on the element because
  // the poll that refreshes the state has to know to leave the picture alone.
  const drawing = (yes) => {
    if (yes) section.dataset.drawing = '1';
    else delete section.dataset.drawing;
  };
  const showSaved = () => {
    drawing(false);
    paint(section, saved(section), 'saved zone');
  };

  // The move and release are watched on the window rather than the image: a drag that ends
  // off the picture must still finish the rectangle. Listening on the image alone dropped
  // every such drag silently, leaving nothing drawn and no hint as to why.
  const move = (event) => start && paint(section, rectangle(start, asFraction(event)), 'zone');
  const finish = (event) => {
    if (!start) return;
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', finish);
    const corners = [start, asFraction(event)];
    start = null;
    if (!drawable(...corners)) {
      // A click, not a drag: put back whatever was saved rather than leave a sliver of a
      // rectangle on screen looking like a zone.
      drawn = null;
      save.disabled = true;
      showSaved();
      return;
    }
    drawn = rectangle(...corners);
    paint(section, drawn, 'zone');
    save.disabled = false;
    section.querySelector('.remove').disabled = false;
  };

  shot.addEventListener('pointerdown', (event) => {
    event.preventDefault();   // no text selection, no image drag
    start = asFraction(event);
    drawn = null;
    drawing(true);
    save.disabled = true;
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', finish);
  });

  save.addEventListener('click', () => drawn && store(section, camera.name, drawn));
  section.querySelector('.refresh').addEventListener('click', () => refresh(image, camera.name));
  section.querySelector('.remove').addEventListener('click', () => {
    // One button for both meanings of "remove": abandon a rectangle just drawn, or delete the
    // saved one. Abandoning first, because that is the undo somebody reaches for mid-draw and
    // it should never cost them the zone that is actually in use.
    if (drawn) {
      drawn = null;
      save.disabled = true;
      showSaved();
      section.querySelector('.result').textContent = '';
      return;
    }
    discard(section, camera.name);
  });
  return section;
}

function refresh(image, camera) {
  // The endpoint sends no-store, but a cache-busting parameter also defeats any proxy.
  image.src = `/frame/${camera}.jpg?t=${Date.now()}`;
}

function saved(section) {
  return section.dataset.saved ? JSON.parse(section.dataset.saved) : null;
}

function describe(section, camera) {
  const state = section.querySelector('.state');
  // "moved" goes quiet once somebody has been warned, which is not the same as the view having
  // come back. So the reading itself decides the wording -- calling a view steady while it
  // sits 100 px from the frame its zone was drawn on is worse than saying nothing at all.
  const off = camera.shift_px !== null && camera.shift_px > camera.tolerance_px;
  if (camera.moved || off) {
    state.className = 'state bad';
    state.textContent = `This camera has moved (${camera.shift}) since its zone was drawn.`
      + ' Redraw the zone.';
  } else if (!camera.has_reference) {
    state.className = 'state';
    state.textContent = 'No zone saved yet, so movement is not being watched for.';
  } else {
    state.className = 'state ok';
    const drift = camera.shift_px === null ? 'no reading yet' : `${camera.shift_px} px off`;
    state.textContent = `Zone saved, view steady (${drift}).`;
  }

  section.dataset.saved = camera.zone ? JSON.stringify(camera.zone) : '';
  section.querySelector('.remove').disabled = !camera.zone && !section.dataset.drawing;
  // The picture is only repainted while nobody is drawing on it: this runs every four seconds,
  // and wiping a rectangle somebody was part-way through would be maddening.
  if (!section.dataset.drawing) {
    paint(section, camera.zone, 'saved zone');
  }
}

async function post(url, body) {
  const response = await fetch(url, {
    method: body ? 'POST' : 'DELETE',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  return { ok: response.ok, body: await response.json() };
}

async function store(section, camera, zone) {
  const result = section.querySelector('.result');
  result.className = 'result';
  result.textContent = 'saving…';
  const [x1, y1, x2, y2] = zone;
  try {
    const { ok, body } = await post('/api/zone', { camera, x1, y1, x2, y2 });
    if (!ok) {
      result.className = 'result bad';
      result.textContent = body.detail || 'failed';
      return;
    }
    delete section.dataset.drawing;   // the drawing is now the saved zone
    result.className = 'result ok';
    result.textContent = 'Saved. This frame is now the reference, and the box is in use from '
      + 'the next frame.';
  } catch (error) {
    result.className = 'result bad';
    result.textContent = String(error);
  }
}

async function discard(section, camera) {
  const result = section.querySelector('.result');
  result.className = 'result';
  result.textContent = 'removing…';
  try {
    const { ok, body } = await post(`/api/zone/${encodeURIComponent(camera)}`, null);
    if (!ok) {
      result.className = 'result bad';
      result.textContent = body.detail || 'failed';
      return;
    }
    section.dataset.saved = '';
    paint(section, null);
    result.className = 'result ok';
    result.textContent = 'Removed. This camera is back on the zone in the config file, and '
      + 'its view is no longer watched for movement.';
  } catch (error) {
    result.className = 'result bad';
    result.textContent = String(error);
  }
}

load();
setInterval(load, REFRESH_MS);
