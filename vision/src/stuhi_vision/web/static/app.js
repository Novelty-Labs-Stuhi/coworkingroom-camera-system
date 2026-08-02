const MESSAGES = {
  enrolled: ['ok', 'Enrolled.'],
  corrected: ['ok', 'Corrected.'],
  unchanged: ['same', 'Already that name — nothing added.'],
  dismissed: ['same', 'Rejected — removed from the gallery.'],
  set_aside: ['ok', 'Saved as unknown — checked, and enrolled under nobody.'],
  attributed: ['ok', 'Saved — no usable face here, so the name counts but teaches nothing.'],
  unlabelled: ['same', 'Label removed — back in the queue.'],
};

// Newest first, or furthest from that person's average face first -- which puts the likeliest
// mistakes at the top instead of the most recent.
let sortBy = 'latest';
// Which pile is on screen. One at a time: with eight of them, stacking made the one being
// worked through impossible to find. Opens on the pile the system is actually asking about.
let showing = 'unnamed';

// The two piles that are not sightings but *pairs* of crossings, and come from the accounting
// rather than the review queue. They are about whether the count adds up, not about labelling.
// What each pile is, shown under the buttons. Only the ones whose contents are not obvious
// from the button: a description on every pile is a paragraph nobody reads.
const ABOUT = new Map([
  [
    'unnamed',
    'People the system gave itself a name for, because it recognised nobody. One card each, '
    + 'showing their clearest face -- not one per sighting, since naming any one of them '
    + 'names the person. These are the only cards that also reach the chat.',
  ],
  [
    'recheck',
    'Faces enrolled under a name they do not look much like, compared against the other '
    + 'faces enrolled for that person. Either the label is wrong, or the face is a poor '
    + 'one to recognise from.',
  ],
]);

const PAIRED = new Map([
  [
    'renamed_exits',
    'Exits the room named by matching against whoever was inside, rather than from the face '
    + 'itself. If one of these is wrong, two people are wrong at once -- so the exit is shown '
    + 'with the entry it was matched to, and both need to be right.',
  ],
  [
    'missing_exits',
    'Somebody the record says came in and never left, since the office day began at 04:30. '
    + 'They are still counted as inside. Underneath each is what the doorway saw afterwards '
    + 'and did not count -- the lost exit is usually one of them.',
  ],
]);

async function loadSightings() {
  const response = await fetch(`/api/sightings?sort=${sortBy}`);
  if (!response.ok) return;
  const data = await response.json();
  renderPeople(data.people);
  renderNames(Object.keys(data.people));
  // Most recently used first: the answer is usually one of the last few people through.
  recentNames = data.recent_names || [];
  // Every pile's size is shown on its button, so the one worth opening is visible without
  // opening it. Only the chosen pile's cards are built.
  for (const button of document.querySelectorAll('.pick')) {
    const rows = (data[button.dataset.pile] || []).filter(about);
    button.querySelector('.count').textContent = rows.length ? `(${rows.length})` : '';
  }
  if (!PAIRED.has(showing)) {
    const rows = (data[showing] || []).filter(about);
    renderCards(document.getElementById('cards'), rows);
    document.getElementById('cards-empty').hidden = rows.length > 0;
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
      // Their whole gallery, in one list, on its own page: which faces recognition uses is a
      // different question from which sightings need checking, and mixing the two made both
      // harder to see.
      label.addEventListener('click', () => {
        location.href = `/gallery?name=${encodeURIComponent(name)}`;
      });
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
function renderCards(host, records) {
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
  const rejecting = fragment.querySelector('.rejecting');
  const labelling = fragment.querySelector('.label-form');
  fragment.querySelector('.label-form .reject').addEventListener('click', () => {
    labelling.hidden = true;
    rejecting.hidden = false;
    rejecting.querySelector('.note').focus();
  });
  rejecting.querySelector('.cancel').addEventListener('click', () => {
    rejecting.hidden = true;
    labelling.hidden = false;
  });
  rejecting.addEventListener('submit', (event) => {
    event.preventDefault();
    send(article, '/api/dismiss', {
      sighting_id: article.dataset.id,
      note: rejecting.querySelector('.note').value,
    });
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

// A pile can be put away. With five of them, the one being worked through should not be
// pushed off the screen by the ones that are not.
for (const section of document.querySelectorAll('.pile')) {
  const arrow = section.querySelector('.fold');
  const shut = arrow.getAttribute('aria-expanded') === 'false';
  section.classList.toggle('folded', shut);
  arrow.addEventListener('click', () => {
    const folded = section.classList.toggle('folded');
    arrow.setAttribute('aria-expanded', String(!folded));
    arrow.textContent = folded ? '▸' : '▾';
  });
}

for (const button of document.querySelectorAll('.sort')) {
  button.addEventListener('click', () => {
    sortBy = button.dataset.sort;
    for (const other of document.querySelectorAll('.sort')) {
      other.classList.toggle('chosen', other === button);
    }
    loadSightings();
  });
}

async function loadAccounting() {
  const response = await fetch(`/api/accounting?pile=${showing}`);
  if (!response.ok) return;
  const data = await response.json();
  const host = document.getElementById('cards');
  host.replaceChildren(...data.entries.map(buildPair));
  document.getElementById('cards-empty').hidden = data.entries.length > 0;
  const button = document.querySelector(`.pick[data-pile="${showing}"] .count`);
  if (button) button.textContent = data.entries.length ? `(${data.entries.length})` : '';
}

// A pair is shown as a pair: two pictures side by side, because the question is whether the
// two belong together and that cannot be judged from one of them.
function buildPair(entry) {
  const article = document.createElement('article');
  article.className = 'card pair';

  const said = document.createElement('p');
  said.className = 'current';
  said.textContent = entry.readable;
  article.append(said);

  const faces = document.createElement('div');
  faces.className = 'together';
  for (const [id, caption] of [
    [entry.entry_frame, 'came in'],
    [entry.exit_frame, 'went out'],
  ]) {
    if (!id) continue;
    const figure = document.createElement('figure');
    figure.className = 'mate';
    const face = document.createElement('img');
    face.src = `/media/${id}.jpg`;
    face.addEventListener('error', () => figure.remove());
    const label = document.createElement('figcaption');
    label.textContent = caption;
    figure.append(face, label);
    faces.append(figure);
  }
  article.append(faces);

  if (entry.missed && entry.missed.length) {
    const seen = document.createElement('p');
    seen.className = 'why';
    seen.textContent = `${entry.missed.length} passage(s) seen afterwards and not counted: `
      + entry.missed
        .map((passage) => `${passage.direction || 'no direction'} over ${passage.frames} frames`)
        .join('; ');
    article.append(seen);
  }
  return article;
}

// Everything that follows from *which* pile is on screen, in one place -- so the pile the
// page opens on is described as fully as one arrived at by clicking, which it was not.
function describePile() {
  document.getElementById('about-pile').textContent =
    ABOUT.get(showing) || PAIRED.get(showing) || '';
  // Ordering is a question about sightings; a pair is already in the order it happened.
  document.getElementById('sorting').hidden = PAIRED.has(showing);
}

for (const button of document.querySelectorAll('.pick')) {
  button.addEventListener('click', () => {
    showing = button.dataset.pile;
    for (const other of document.querySelectorAll('.pick')) {
      other.classList.toggle('chosen', other === button);
    }
    describePile();
    document.getElementById('cards').replaceChildren();
    refresh();
  });
}

function refresh() {
  return PAIRED.has(showing) ? Promise.all([loadSightings(), loadAccounting()]) : loadSightings();
}

describePile();
refresh();
setInterval(refresh, 15000);
