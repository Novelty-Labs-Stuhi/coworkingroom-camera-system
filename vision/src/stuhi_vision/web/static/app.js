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
  // Most recently used first: the answer is usually one of the last few people through.
  recentNames = data.recent_names || [];
  for (const [section, records] of [
    ['pending', data.pending],
    ['recheck', data.recheck],
    ['labelled', data.labelled],
  ]) {
    const shown = records.filter(about);
    renderCards(section, shown);
    const count = document.getElementById(`${section}-count`);
    if (count) count.textContent = shown.length ? `(${shown.length})` : '';
  }
  await loadAudit();
}

// Offered by the arrow beside each name box, in the order the names were last used.
let recentNames = [];
// When a person is picked from Enrolled, every pile narrows to them. Null shows everybody.
let onlyPerson = null;

// The audit is a separate call because it reads every enrolled embedding off disk, which
// is far heavier than listing records -- no reason to pay for it on every poll of the list.
async function loadAudit() {
  const response = await fetch('/api/audit');
  if (!response.ok) return;
  const data = await response.json();

  renderThin(data.thin);
}

function renderThin(thin) {
  const names = Object.keys(thin);
  const list = document.getElementById('thin');
  list.replaceChildren(
    ...names.map((name) => {
      const item = document.createElement('li');
      if (name === onlyPerson) item.classList.add('chosen');
      // The name itself is the control: picking somebody narrows every pile to them, which
      // is how you check one person's faces against each other rather than hunting for them
      // among everybody else's.
      const label = document.createElement('button');
      label.type = 'button';
      label.className = 'who';
      label.textContent = name;
      label.title = `show everything labelled or guessed as "${name}"`;
      label.addEventListener('click', () => showOnly(name));
      const count = document.createElement('span');
      count.textContent = ` ${thin[name]} face${thin[name] === 1 ? '' : 's'}`;
      item.append(label, count);
      return item;
    })
  );
  document.getElementById('thin-empty').hidden = names.length > 0;
}

// Whether a card belongs to the person being looked at. A guess counts: the point of picking
// a name is to see everything that claims to be them -- confirmed, guessed or flagged -- and a
// wrong guess sitting among their real faces is exactly what you are looking for.
function about(record) {
  if (!onlyPerson) return true;
  return (record.labelled_as || record.name) === onlyPerson;
}

function showOnly(name) {
  onlyPerson = onlyPerson === name ? null : name;
  const note = document.getElementById('showing');
  note.textContent = onlyPerson
    ? `Showing everything labelled or guessed as "${onlyPerson}".`
    : '';
  note.hidden = !onlyPerson;
  loadSightings();
}

function renderPeople(people) {
  const list = document.getElementById('people');
  const names = Object.keys(people);
  list.replaceChildren(
    ...names.map((name) => {
      const item = document.createElement('li');
      if (name === onlyPerson) item.classList.add('chosen');
      // The name itself is the control: picking somebody narrows every pile to them, which
      // is how you check one person's faces against each other rather than hunting for them
      // among everybody else's.
      const label = document.createElement('button');
      label.type = 'button';
      label.className = 'who';
      label.textContent = name;
      label.title = `show everything labelled or guessed as "${name}"`;
      label.addEventListener('click', () => showOnly(name));
      const count = document.createElement('span');
      // The count is the honest check that a label added a reference rather than not.
      count.textContent = ` ${people[name]}`;
      // A misspelling is one mistake, so it is corrected once here rather than clip by clip.
      // Relabelling each sighting by hand would discard and re-add every reference vector.
      const fix = document.createElement('button');
      fix.className = 'fix';
      fix.type = 'button';
      fix.textContent = 'rename / merge';
      fix.title = `rename "${name}" everywhere, or merge it into another person`;
      fix.addEventListener('click', () => askAbout(item, name, people[name]));
      item.append(label, count, fix);
      return item;
    })
  );
  document.getElementById('people-empty').hidden = names.length > 0;
}

// One control for both, because they are the same operation: a name is replaced everywhere it
// was used, and if the replacement already exists the two become one person. Known names are
// offered, so merging is picking from a list rather than spelling something exactly right.
function askAbout(item, name, references) {
  if (item.querySelector('form')) return;
  const form = document.createElement('form');
  form.className = 'renaming';

  const field = document.createElement('input');
  field.className = 'name';
  field.value = name;
  field.setAttribute('list', 'known-names');
  field.title = 'a new spelling, or an existing person to merge into';

  const apply = document.createElement('button');
  apply.type = 'submit';
  apply.textContent = 'apply';

  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'reject';
  cancel.textContent = 'cancel';
  cancel.addEventListener('click', () => form.remove());

  form.append(field, apply, cancel);
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const wanted = field.value.trim();
    if (!wanted || wanted === name) {
      form.remove();
      return;
    }
    form.remove();
    renameEverywhere(name, wanted, references);
  });
  item.append(form);
  field.focus();
  field.select();
}

async function renameEverywhere(name, corrected, references) {
  const response = await fetch('/api/rename', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ old: name, new: corrected }),
  });
  const body = await response.json();
  const note = document.getElementById('rename-result');
  if (!response.ok) {
    note.className = 'result bad';
    note.textContent = body.detail || 'rename failed';
    return;
  }
  note.className = 'result ok';
  const merged = body.people[body.renamed] > references;
  note.textContent = `${merged ? 'Merged into' : 'Now'} "${body.renamed}": ${body.clips} `
    + `clip(s) and ${body.events} history entr(y/ies) corrected.`;
  loadSightings();
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
  // The system's own answer goes in the field: a name it recognised, or nothing when it did
  // not. So the job is confirming or correcting rather than typing every name from scratch --
  // and a guess sitting in the box is visible, which typing into an empty box never made it.
  // Never overwrite what somebody is part-way through typing.
  if (document.activeElement !== input) {
    const guess = record.labelled_as || (record.outcome === 'named' ? record.name : '');
    input.value = guess || '';
    input.classList.toggle('guessed', !record.labelled_as && Boolean(guess));
  }
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
  offerNames(fragment.querySelector('.picker'), input);

  return article;
}

// The recent names on a button. A datalist only opens once somebody types, and the answer
// here is usually one of the last few people through the door -- so it is worth one click.
function offerNames(picker, input) {
  const choices = picker.querySelector('.choices');
  const close = () => { choices.hidden = true; };

  picker.querySelector('.choose').addEventListener('click', () => {
    if (!choices.hidden) { close(); return; }
    choices.replaceChildren(
      ...recentNames.map((name) => {
        const option = document.createElement('li');
        const pick = document.createElement('button');
        pick.type = 'button';
        pick.textContent = name;
        pick.addEventListener('click', () => {
          input.value = name;
          input.classList.remove('guessed');   // chosen by a person now, not guessed
          close();
          input.focus();
        });
        option.append(pick);
        return option;
      })
    );
    if (!recentNames.length) {
      const empty = document.createElement('li');
      empty.className = 'empty';
      empty.textContent = 'no names yet';
      choices.append(empty);
    }
    choices.hidden = false;
  });

  // Any click elsewhere puts it away, which is what a dropdown is expected to do.
  document.addEventListener('click', (event) => {
    if (!picker.contains(event.target)) close();
  });
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
    // Gone from this pile at once, rather than after the next poll: the grid closes the gap
    // by itself, and a card that lingers invites labelling the same clip twice.
    article.remove();
    loadSightings();
  } catch (error) {
    result.className = 'result bad';
    result.textContent = String(error);
  } finally {
    buttons.forEach((button) => (button.disabled = false));
  }
}

loadSightings();
setInterval(loadSightings, 15000);
