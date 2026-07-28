const MESSAGES = {
  enrolled: ['ok', 'Enrolled.'],
  corrected: ['ok', 'Corrected.'],
  unchanged: ['same', 'Already that name — nothing added.'],
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
    submitLabel(article, input.value);
  });

  return article;
}

async function submitLabel(article, name) {
  const result = article.querySelector('.result');
  const button = article.querySelector('button');
  button.disabled = true;
  result.className = 'result';
  result.textContent = 'saving…';

  try {
    const response = await fetch('/api/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sighting_id: article.dataset.id, name }),
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
    button.disabled = false;
  }
}

loadSightings();
setInterval(loadSightings, 15000);
