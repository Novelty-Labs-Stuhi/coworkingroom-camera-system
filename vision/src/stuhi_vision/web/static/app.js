const MESSAGES = {
  enrolled: ['ok', 'Enrolled.'],
  corrected: ['ok', 'Corrected.'],
  unchanged: ['same', 'Already that name — nothing added.'],
  dismissed: ['same', 'Rejected — removed from the gallery.'],
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
      item.append(label, count);
      return item;
    })
  );
  document.getElementById('people-empty').hidden = names.length > 0;
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
    // play() only after the element is in the document; a detached one just rejects.
    card.querySelector('video')?.play().catch(() => {});
  });

  document.getElementById(`${section}-empty`).hidden = records.length > 0;
}

function updateCard(article, record) {
  article.querySelector('.direction').textContent = record.direction;
  article.querySelector('.score').textContent =
    record.outcome === 'unidentified' ? 'no face seen' : `score ${record.score}`;
  setCurrent(article.querySelector('.current'), record);

  // A group label was assigned by crossing order, so say so -- it is the case most likely
  // to carry the right names on the wrong people.
  const group = article.querySelector('.group');
  const grouped = record.group_size > 1;
  group.hidden = !grouped;
  if (grouped) group.textContent = `${record.position} of ${record.group_size} together`;

  const why = article.querySelector('.why');
  why.hidden = !record.reason;
  if (record.reason) {
    why.textContent = `${record.reason} — ${record.similarity} against ${record.average} average`;
  }

  const input = article.querySelector('.name');
  // Never overwrite what someone is in the middle of typing.
  if (record.labelled_as && document.activeElement !== input) input.value = record.labelled_as;
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

  const video = fragment.querySelector('video');
  video.poster = `/media/${record.id}.jpg`;
  video.src = `/media/${record.id}.mp4`;
  // Fall back to the saved face crop when a clip could not be encoded.
  video.addEventListener('error', () => {
    const image = document.createElement('img');
    image.src = `/media/${record.id}.jpg`;
    image.alt = 'face crop';
    image.style.width = '100%';
    video.replaceWith(image);
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
