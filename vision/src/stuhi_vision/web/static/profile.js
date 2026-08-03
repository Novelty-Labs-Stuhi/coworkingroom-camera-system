// One person's profile: their name, their own words, their picture, and their year of days.
//
// Everything here that can be edited edits the same record the rest of the system reads, through
// the same endpoints the labelling page uses. There is no profile-only copy of a name or a
// label to drift out of step with it.

let who = new URLSearchParams(location.search).get('name') || '';
let chosenDay = '';

const WINDOWS = {
  day: 'today',
  week: 'this week',
  month: 'this month',
  year: 'this year',
  all: 'all time',
};

async function load() {
  const response = await fetch(`/api/profile/${encodeURIComponent(who)}`);
  if (!response.ok) return;
  const profile = await response.json();

  document.title = `stuhi — ${profile.name}`;
  document.getElementById('name').textContent = profile.name || 'nobody';
  document.getElementById('in-now').hidden = !profile.inside;
  showPortrait(profile);
  showBio(profile.bio);
  showTotals(profile);
  await loadActivity();
}

function showPortrait(profile) {
  const face = document.getElementById('portrait-face');
  const missing = document.getElementById('no-face');
  const link = document.getElementById('portrait');
  link.href = `/gallery?name=${encodeURIComponent(profile.name)}`;

  face.hidden = !profile.picture;
  missing.hidden = Boolean(profile.picture);
  if (profile.picture) {
    face.src = `/media/${profile.picture}.jpg`;
    face.alt = profile.name;
  }
  document.getElementById('faces').textContent = profile.faces.kept
    ? `${profile.faces.in_use} of ${profile.faces.kept} faces are used to recognise them.`
    : 'No faces are filed under this name yet.';
}

function showBio(bio) {
  const text = document.getElementById('bio-text');
  text.textContent = bio;
  text.classList.toggle('none', !bio);
  if (!bio) text.textContent = 'No bio yet.';
  document.getElementById('edit-bio').textContent = bio ? 'edit bio' : 'add a bio';
  document.getElementById('bio-input').value = bio;
}

function showTotals(profile) {
  const totals = document.getElementById('totals');
  totals.replaceChildren(
    ...Object.entries(WINDOWS).map(([window_, label]) =>
      chip(profile.totals[window_].readable, ` ${label}`)
    ),
    chip(`${profile.streak_days} day(s)`, ' in a row'),
    chip(`${profile.visits}`, ' visit(s) recorded')
  );
}

function chip(strong, rest) {
  const item = document.createElement('li');
  const bold = document.createElement('b');
  bold.textContent = strong;
  const span = document.createElement('span');
  span.textContent = rest;
  item.append(bold, span);
  return item;
}

// --- the chart ------------------------------------------------------------------------------

async function loadActivity() {
  const response = await fetch(`/api/profile/${encodeURIComponent(who)}/activity`);
  if (!response.ok) return;
  const chart = await response.json();

  const anything = chart.days.some((day) => day.seconds > 0);
  document.getElementById('activity-empty').hidden = anything;
  drawActivityCalendar(document.getElementById('calendar'), chart, pickDay);
  if (chosenDay) markChosenDay(document.getElementById('calendar'), chosenDay);
}

// A day before counting began has nothing to open, and opening an empty panel on it would
// suggest the record was checked and found empty.
function pickDay(day) {
  if (!day.counted) return;
  chosenDay = day.date;
  markChosenDay(document.getElementById('calendar'), chosenDay);
  loadDay();
}

async function loadDay() {
  const response = await fetch(
    `/api/profile/${encodeURIComponent(who)}/day/${encodeURIComponent(chosenDay)}`
  );
  if (!response.ok) return;
  const day = await response.json();

  document.getElementById('day-panel').hidden = false;
  document.getElementById('day-title').textContent = asDay(day.date);
  document.getElementById('day-summary').textContent = day.crossings.length
    ? `${day.readable} in the room, from ${day.crossings.length} recorded crossing(s).`
    : 'Nothing was recorded for this person on this day.';

  const timeline = document.getElementById('day-timeline');
  timeline.replaceChildren(...day.crossings.map(crossingRow));
  document.getElementById('day-empty').hidden = day.crossings.length > 0;
}

function crossingRow(crossing) {
  const row = document
    .getElementById('crossing-template')
    .content.cloneNode(true)
    .querySelector('.crossing');

  row.querySelector('.at').textContent = asClock(crossing.at);
  const way = row.querySelector('.way');
  way.textContent = crossing.direction === 'in' ? 'came in' : 'left';
  way.classList.add(crossing.direction);

  showFrame(row, crossing);
  return row;
}

function showFrame(row, crossing) {
  const frame = row.querySelector('.frame');
  const filed = row.querySelector('.filed');
  const fix = row.querySelector('.correct');

  if (!crossing.sighting_id) {
    // Nothing to look at and nothing to relabel: the crossing was counted from the doorframe
    // alone. Saying so beats a row with a dead button on it.
    filed.textContent = 'no frame was saved for this crossing';
    filed.classList.add('none');
    fix.hidden = true;
    return;
  }

  if (crossing.has_picture) {
    frame.hidden = false;
    frame.src = `/media/${crossing.sighting_id}.jpg`;
    frame.addEventListener('error', () => { frame.hidden = true; });
  }
  filed.textContent = `filed as ${crossing.labelled_as}`;
  // The frame disagreeing with the crossing is the thing worth noticing on this page.
  filed.classList.toggle('disagrees', crossing.labelled_as !== who);
  wireCorrection(row, crossing);
}

function wireCorrection(row, crossing) {
  const form = row.querySelector('.correcting');
  const field = form.querySelector('.name');
  const fix = row.querySelector('.correct');

  fix.addEventListener('click', () => {
    form.hidden = !form.hidden;
    if (!form.hidden) field.focus();
  });
  form.querySelector('.cancel').addEventListener('click', () => { form.hidden = true; });
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    correct(row, crossing.sighting_id, field.value);
  });
}

async function correct(row, sightingId, name) {
  const said = row.querySelector('.result');
  if (!name.trim()) {
    say(said, 'bad', 'give a name');
    return;
  }
  say(said, '', 'saving…');
  const outcome = await post('/api/label', { sighting_id: sightingId, name });
  if (!outcome.ok) {
    say(said, 'bad', outcome.detail);
    return;
  }
  say(said, 'ok', outcome.body.outcome);
  // Both, because a correction moves hours as well as a label: the box for this day may be a
  // different shade now, and the crossing may no longer belong to this person at all.
  await loadDay();
  await load();
}

// --- editing the name and the bio -----------------------------------------------------------

function wireRename() {
  const form = document.getElementById('renaming');
  const field = document.getElementById('new-name');

  document.getElementById('rename').addEventListener('click', () => {
    form.hidden = !form.hidden;
    field.value = who;
    if (!form.hidden) field.focus();
  });
  document.getElementById('cancel-rename').addEventListener('click', () => { form.hidden = true; });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const said = document.getElementById('header-result');
    say(said, '', 'saving…');
    const outcome = await post('/api/rename', { old: who, new: field.value });
    if (!outcome.ok) {
      say(said, 'bad', outcome.detail);
      return;
    }
    // The page is about a name, so the address has to follow the correction -- otherwise a
    // refresh would open a profile that no longer exists.
    who = outcome.body.renamed;
    history.replaceState({}, '', `/profile?name=${encodeURIComponent(who)}`);
    form.hidden = true;
    say(said, 'ok', `renamed everywhere: ${outcome.body.clips} frame(s), ${outcome.body.events} crossing(s)`);
    load();
  });
}

function wireBio() {
  const form = document.getElementById('bio-form');
  const shown = document.getElementById('bio-shown');
  const field = document.getElementById('bio-input');
  const left = document.getElementById('bio-left');

  const countdown = () => {
    left.textContent = `${field.maxLength - field.value.length} characters left`;
  };

  document.getElementById('edit-bio').addEventListener('click', () => {
    form.hidden = false;
    shown.hidden = true;
    countdown();
    field.focus();
  });
  const close = () => { form.hidden = true; shown.hidden = false; };
  document.getElementById('cancel-bio').addEventListener('click', close);
  field.addEventListener('input', countdown);

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const said = document.getElementById('header-result');
    say(said, '', 'saving…');
    const outcome = await post('/api/bio', { name: who, bio: field.value });
    if (!outcome.ok) {
      say(said, 'bad', outcome.detail);
      return;
    }
    showBio(outcome.body.bio);
    close();
    say(said, 'ok', outcome.body.bio ? 'bio saved' : 'bio cleared');
  });
}

// --- odds and ends --------------------------------------------------------------------------

async function post(url, body) {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const parsed = await response.json();
    return response.ok
      ? { ok: true, body: parsed }
      : { ok: false, detail: parsed.detail || 'failed' };
  } catch (error) {
    return { ok: false, detail: String(error) };
  }
}

function say(node, kind, text) {
  node.className = kind ? `result ${kind}` : 'result';
  node.textContent = text;
}

function asClock(seconds) {
  return new Date(seconds * 1000).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function asDay(iso) {
  const [year, month, date] = iso.split('-').map(Number);
  return new Date(year, month - 1, date).toLocaleDateString(undefined, {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
}

document.getElementById('close-day').addEventListener('click', () => {
  document.getElementById('day-panel').hidden = true;
  chosenDay = '';
  markChosenDay(document.getElementById('calendar'), '');
});

wireRename();
wireBio();
load();
