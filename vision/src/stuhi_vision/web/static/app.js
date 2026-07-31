const MESSAGES = {
  enrolled: ['ok', 'Enrolled.'],
  corrected: ['ok', 'Corrected.'],
  unchanged: ['same', 'Already that name — nothing added.'],
  dismissed: ['same', 'Rejected — removed from the gallery.'],
  unlabelled: ['same', 'Label removed — back in the queue.'],
};

async function loadSightings() {
  const response = await fetch('/api/sightings');
  if (!response.ok) return;
  const data = await response.json();
  renderPeople(data.people);
  renderNames(Object.keys(data.people));
  renderCards('pending', data.pending);
  renderCards('labelled', data.labelled);
  document.getElementById('pending-count').textContent =
    data.pending.length ? `(${data.pending.length})` : '';
  await loadAudit();
}

// The audit is a separate call because it reads every enrolled embedding off disk, which
// is far heavier than listing records -- no reason to pay for it on every poll of the list.
async function loadAudit() {
  const response = await fetch('/api/audit');
  if (!response.ok) return;
  const data = await response.json();
  renderCards('suspects', data.suspects);
  document.getElementById('suspects-count').textContent =
    data.suspects.length ? `(${data.suspects.length})` : '';
  renderThin(data.thin);
}

function renderThin(thin) {
  const names = Object.keys(thin);
  const list = document.getElementById('thin');
  list.replaceChildren(
    ...names.map((name) => {
      const item = document.createElement('li');
      const label = document.createElement('b');
      label.textContent = name;
      const count = document.createElement('span');
      count.textContent = ` ${thin[name]} face${thin[name] === 1 ? '' : 's'}`;
      item.append(label, count);
      return item;
    })
  );
  document.getElementById('thin-empty').hidden = names.length > 0;
}

function renderPeople(people) {
  const list = document.getElementById('people');
  const names = Object.keys(people);
  list.replaceChildren(
    ...names.map((name) => {
      const item = document.createElement('li');
      const label = document.createElement('b');
      label.textContent = name;
      const count = document.createElement('span');
      // The count is the honest check that a label added a reference rather than not.
      count.textContent = ` ${people[name]}`;
      // A misspelling is one mistake, so it is corrected once here rather than clip by clip.
      // Relabelling each sighting by hand would discard and re-add every reference vector.
      const fix = document.createElement('button');
      fix.className = 'fix';
      fix.type = 'button';
      fix.textContent = 'rename';
      fix.title = `correct the spelling of "${name}" everywhere it was used`;
      fix.addEventListener('click', () => renameEverywhere(name, people[name]));
      item.append(label, count, fix);
      return item;
    })
  );
  document.getElementById('people-empty').hidden = names.length > 0;
}

async function renameEverywhere(name, references) {
  const corrected = window.prompt(
    `Correct the spelling of "${name}" on all ${references} reference(s), `
    + 'every clip labelled with it, and the recorded history.

'
    + 'Typing a name that already exists merges the two into one person.',
    name
  );
  if (!corrected || corrected.trim() === name) return;

  const response = await fetch('/api/rename', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ old: name, new: corrected.trim() }),
  });
  const body = await response.json();
  const note = document.getElementById('rename-result');
  if (!response.ok) {
    note.className = 'result bad';
    note.textContent = body.detail || 'rename failed';
    return;
  }
  note.className = 'result ok';
  note.textContent = `Now "${body.renamed}": ${body.clips} clip(s) and `
    + `${body.events} history entr(y/ies) corrected.`;
  load();
}

function renderNames(names) {
  const datalist = document.getElementById('known-names');
  datalist.replaceChildren(
    ...names.map((name) => {
      const option = document.createElement('option');
      option.value = name;
      return option;
    })
  );
}

// Refresh without rebuilding. Replacing the whole list destroyed and recreated every
// <video>, which blanked each clip, restarted it, and shifted the layout every poll. So
// existing cards are updated in place and only genuinely new sightings are inserted --
// a card's video element is never touched once it exists.
function renderCards(section, records) {
  const host = document.getElementById(section);
  const byId = new Map([...host.children].map((element) => [element.dataset.id, element]));
  const order = records.map((record) => record.id);

  for (const [id, element] of byId) {
    if (!order.includes(id)) {
      element.remove();
      byId.delete(id);
    }
  }

  records.forEach((record, index) => {
    const existing = byId.get(record.id);
    if (existing) {
      updateCard(existing, record);
      return;
    }
    const card = buildCard(record);
    // Insert ahead of the first card that should follow it, so new arrivals land in the
    // right place without moving anything that is already playing.
    const nextId = order.slice(index + 1).find((id) => byId.has(id));
    host.insertBefore(card, nextId ? byId.get(nextId) : null);
    byId.set(record.id, card);
  });

  document.getElementById(`${section}-empty`).hidden = records.length > 0;
}

function updateCard(article, record) {
  article.querySelector('.direction').textContent = record.direction;
  article.querySelector('.score').textContent =
    record.outcome === 'unidentified' ? 'no face seen' : `score ${record.score}`;
  setCurrent(article.querySelector('.current'), record);

  // A group label is assigned by crossing order, so say so -- it is the case most likely to
  // carry the right names on the wrong people.
  const group = article.querySelector('.group');
  const grouped = record.group_size > 1;
  group.hidden = !grouped;
  if (grouped) group.textContent = `${record.position} of ${record.group_size} together`;
  showTogether(article, record);

  const why = article.querySelector('.why');
  why.hidden = !record.reason;
  if (record.reason) {
    why.textContent = `${record.reason} — ${record.similarity} against ${record.average} average`;
  }

  const input = article.querySelector('.name');
  // Never overwrite what someone is in the middle of typing.
  if (record.labelled_as && document.activeElement !== input) input.value = record.labelled_as;
}

// Everybody who came through with this person, in the order they crossed. Shown because that
// order is the whole basis of a group label: seeing the faces in it turns "trust the numbering"
// into something checkable. One name still labels only this card; a comma-separated list
// labels the group in this order.
function showTogether(article, record) {
  const strip = article.querySelector('.together');
  const input = article.querySelector('.name');
  if (record.group_size < 2) {
    strip.replaceChildren();
    strip.hidden = true;
    input.placeholder = 'name';
    return;
  }
  strip.hidden = false;
  input.placeholder = `name, or ${record.group_size} names in this order`;
  if (strip.children.length === record.group_ids.length) return;   // already drawn

  strip.replaceChildren(
    ...record.group_ids.map((id, index) => {
      const figure = document.createElement('figure');
      figure.className = id === record.id ? 'mate is-this-one' : 'mate';
      const face = document.createElement('img');
      face.src = `/media/${id}.jpg`;
      face.alt = `person ${index + 1} of this group`;
      face.addEventListener('error', () => face.remove());
      const caption = document.createElement('figcaption');
      caption.textContent = index + 1;
      figure.append(face, caption);
      return figure;
    })
  );
}

function setCurrent(current, record) {
  if (record.labelled_as) {
    current.textContent = 'labelled ';
    const name = document.createElement('b');
    name.textContent = record.labelled_as;
    current.append(name);
  } else {
    current.textContent = record.id;
  }
}

function buildCard(record) {
  const fragment = document.getElementById('card-template').content.cloneNode(true);
  const article = fragment.querySelector('.card');
  article.dataset.id = record.id;

  const face = fragment.querySelector('img.face');
  face.src = `/media/${record.id}.jpg`;
  // An UNIDENTIFIED sighting has no crop; say so rather than showing a broken image.
  face.addEventListener('error', () => {
    const note = document.createElement('p');
    note.className = 'empty';
    note.textContent = 'no face was captured for this crossing';
    face.replaceWith(note);
  });

  // preload="none" until opened, so twenty cards do not fetch twenty videos.
  const details = fragment.querySelector('details.clip');
  const video = details.querySelector('video');
  details.addEventListener('toggle', () => {
    if (details.open && !video.src) {
      video.src = `/media/${record.id}.mp4`;
      video.play().catch(() => {});
    }
  });

  updateCard(article, record);

  const form = fragment.querySelector('.label-form');
  const input = fragment.querySelector('.name');
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    send(article, '/api/label', { sighting_id: article.dataset.id, name: input.value });
  });
  fragment.querySelector('.reject').addEventListener('click', () => {
    send(article, '/api/dismiss', { sighting_id: article.dataset.id });
  });
  // Two different mistakes, two different remedies: the clip is unusable (reject), or the
  // clip is fine but the name was wrong (put it back in the queue).
  fragment.querySelector('.undo').addEventListener('click', () => {
    send(article, '/api/unlabel', { sighting_id: article.dataset.id });
  });

  return article;
}

async function send(article, url, payload) {
  const result = article.querySelector('.result');
  const buttons = [...article.querySelectorAll('button')];
  buttons.forEach((button) => (button.disabled = true));
  result.className = 'result';
  result.textContent = 'saving…';

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      result.className = 'result bad';
      result.textContent = body.detail || 'failed';
      return;
    }
    const [tone, message] = MESSAGES[body.outcome] || ['ok', body.outcome];
    result.className = `result ${tone}`;
    result.textContent = message;
    renderPeople(body.people);
    renderNames(Object.keys(body.people));
    setTimeout(loadSightings, 800);
  } catch (error) {
    result.className = 'result bad';
    result.textContent = String(error);
  } finally {
    buttons.forEach((button) => (button.disabled = false));
  }
}

loadSightings();
setInterval(loadSightings, 15000);
