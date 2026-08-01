// One person: every face kept under their name, and which ones recognition may use.
//
// The list is deliberately one list rather than piles. The question here is not "what needs
// doing" but "what is this person's gallery made of", and that is answered by seeing all of it
// in one order -- newest first to check what has just been added, or in-use first to check what
// is actually deciding who somebody is.

const name = new URLSearchParams(location.search).get('name') || '';
let sortBy = 'latest';

async function load() {
  const response = await fetch(`/api/person/${encodeURIComponent(name)}/faces?sort=${sortBy}`);
  if (!response.ok) return;
  const person = await response.json();

  document.getElementById('who').textContent = person.name || 'nobody';
  document.getElementById('summary').textContent =
    `${person.in_use} of ${person.kept} faces are used for recognition.`;

  const host = document.getElementById('frames');
  host.replaceChildren(...person.frames.map(build));
  document.getElementById('frames-empty').hidden = person.frames.length > 0;
}

function build(frame) {
  const article = document
    .getElementById('frame-template')
    .content.cloneNode(true)
    .querySelector('.frame');
  article.dataset.id = frame.id;

  const face = article.querySelector('img.face');
  face.src = `/media/${frame.id}.jpg`;
  face.addEventListener('error', () => face.remove());

  paint(article, frame);

  article.querySelector('.use').addEventListener('click', () => decide(article, frame, true));
  article.querySelector('.drop').addEventListener('click', () => decide(article, frame, false));
  // Undoing a decision is not the opposite decision: it is handing the frame back to the rule.
  article.querySelector('.auto').addEventListener('click', () => decide(article, frame, null));
  return article;
}

function paint(article, frame) {
  article.classList.toggle('using', frame.in_use);
  article.classList.toggle('idle', !frame.in_use);
  article.classList.toggle('decided', frame.decided !== null);

  article.querySelector('.state').textContent = frame.in_use
    ? `used for recognition — ${frame.why}`
    : `not used — ${frame.why}`;
  const when = new Date(frame.timestamp * 1000).toLocaleString();
  article.querySelector('.numbers').textContent =
    `${when} · ${frame.direction} · ${frame.closeness} like the average`;

  // Only the actions that would change something are offered.
  article.querySelector('.use').disabled = frame.decided === true;
  article.querySelector('.drop').disabled = frame.decided === false;
  article.querySelector('.auto').disabled = frame.decided === null;
}

async function decide(article, frame, wanted) {
  const result = article.querySelector('.result');
  result.className = 'result';
  result.textContent = 'saving…';
  try {
    const response = await fetch('/api/use-face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sighting_id: frame.id, wanted }),
    });
    if (!response.ok) {
      const body = await response.json();
      result.className = 'result bad';
      result.textContent = body.detail || 'failed';
      return;
    }
    result.className = 'result ok';
    result.textContent = wanted === null ? 'back to automatic' : wanted ? 'in use' : 'excluded';
    // Reloaded, because one decision changes the average and so what the rule picks for every
    // other face: showing this card's new state alone would be showing a stale page.
    load();
  } catch (error) {
    result.className = 'result bad';
    result.textContent = String(error);
  }
}

for (const button of document.querySelectorAll('.sort')) {
  button.addEventListener('click', () => {
    sortBy = button.dataset.sort;
    for (const other of document.querySelectorAll('.sort')) {
      other.classList.toggle('chosen', other === button);
    }
    load();
  });
}

load();
