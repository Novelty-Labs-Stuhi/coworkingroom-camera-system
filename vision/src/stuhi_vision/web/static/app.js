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

function renderCards(section, records) {
  const host = document.getElementById(section);
  host.replaceChildren(...records.map(buildCard));
  document.getElementById(`${section}-empty`).hidden = records.length > 0;
}

function buildCard(record) {
  const card = document.getElementById('card-template').content.cloneNode(true);
  const article = card.querySelector('.card');
  article.dataset.id = record.id;

  const video = card.querySelector('video');
  video.src = `/media/${record.id}.mp4`;
  // Fall back to the saved face crop when a clip could not be encoded.
  video.addEventListener('error', () => {
    const image = document.createElement('img');
    image.src = `/media/${record.id}.jpg`;
    image.alt = 'face crop';
    image.style.width = '100%';
    video.replaceWith(image);
  });

  card.querySelector('.direction').textContent = record.direction;
  card.querySelector('.score').textContent =
    record.outcome === 'unidentified' ? 'no face seen' : `score ${record.score}`;

  const current = card.querySelector('.current');
  if (record.labelled_as) {
    current.innerHTML = 'labelled <b></b>';
    current.querySelector('b').textContent = record.labelled_as;
  } else {
    current.textContent = record.id;
  }

  const form = card.querySelector('.label-form');
  const input = card.querySelector('.name');
  if (record.labelled_as) input.value = record.labelled_as;
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    submitLabel(article, input.value);
  });

  return card;
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
