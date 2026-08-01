// Time in the room: a leaderboard over a chosen period, and one person's own figures.
//
// Its own file, alongside its own stylesheet, so this page cannot collide with the labelling
// page -- which somebody else may be editing at the same time.

// Which period is shown, and how many of them back. Offset 0 is the one running now.
let window_ = 'streak';
let offset = 0;
let filter = '';

async function load() {
  const response = await fetch(`/api/leaderboard?window=${window_}&offset=${offset}`);
  if (!response.ok) return;
  const board = await response.json();

  showCovering(board);
  const rows = board.standings.filter(
    (row) => !filter || row.name.toLowerCase().includes(filter)
  );
  renderBoard(rows);
  document.getElementById('board-empty').hidden = rows.length > 0;
  await loadInside();
}

function showCovering(board) {
  const note = document.getElementById('covering');
  if (window_ === 'streak') {
    note.textContent = 'days in a row, counted to today';
  } else if (window_ === 'all') {
    note.textContent = 'everything recorded';
  } else {
    note.textContent = `${asDay(board.from)} to ${asDay(board.until)}`;
  }
  // There is no window after the one running now, so stepping forward from it is meaningless.
  document.getElementById('later').disabled = offset === 0;
  const stepping = window_ === 'streak' || window_ === 'all';
  document.getElementById('earlier').disabled = stepping;
  if (stepping) document.getElementById('later').disabled = true;
}

function asDay(seconds) {
  return new Date(seconds * 1000).toLocaleDateString(undefined, {
    day: 'numeric',
    month: 'short',
  });
}

function renderBoard(rows) {
  const list = document.getElementById('board');
  list.replaceChildren(
    ...rows.map((row, place) => {
      const item = document.createElement('li');

      const rank = document.createElement('span');
      rank.className = 'rank';
      rank.textContent = place + 1;

      // The name opens their own figures: a board answers "who", a person answers "when".
      const name = document.createElement('button');
      name.type = 'button';
      name.className = 'who';
      name.textContent = row.name;
      name.addEventListener('click', () => showPerson(row.name));

      const time = document.createElement('b');
      time.className = 'held';
      time.textContent = row.readable;

      const detail = document.createElement('span');
      detail.className = 'detail';
      detail.textContent = row.days
        ? `${row.visits} visit(s) over ${row.days} day(s)`
        : `${row.visits} visit(s)`;

      item.append(rank, name, time, detail);
      if (row.still_inside) {
        const here = document.createElement('span');
        here.className = 'here';
        here.textContent = 'in now';
        item.append(here);
      }
      return item;
    })
  );
}

async function showPerson(name) {
  const response = await fetch(`/api/person/${encodeURIComponent(name)}`);
  if (!response.ok) return;
  const figures = await response.json();

  document.getElementById('person-panel').hidden = false;
  document.getElementById('person-name').textContent =
    figures.inside ? `${name} — in the room now` : name;

  const totals = document.getElementById('person-totals');
  const labels = { day: 'today', week: 'this week', month: 'this month', year: 'this year', all: 'all time' };
  totals.replaceChildren(
    ...Object.entries(labels).map(([window, label]) => {
      const item = document.createElement('li');
      const strong = document.createElement('b');
      strong.textContent = figures.totals[window].readable;
      const span = document.createElement('span');
      span.textContent = ` ${label}`;
      item.append(strong, span);
      return item;
    }),
    streakChip(figures.streak_days)
  );

  const visits = document.getElementById('person-visits');
  visits.replaceChildren(
    ...figures.visits.map((visit) => {
      const item = document.createElement('li');
      item.textContent = visit.left
        ? `${asMoment(visit.entered)} → ${asMoment(visit.left)} · ${visit.readable}`
        : `${asMoment(visit.entered)} → still inside · ${visit.readable}`;
      return item;
    })
  );
  document.getElementById('person-panel').scrollIntoView({ behavior: 'smooth' });
}

function streakChip(days) {
  const item = document.createElement('li');
  const strong = document.createElement('b');
  strong.textContent = `${days} day(s)`;
  const span = document.createElement('span');
  span.textContent = ' in a row';
  item.append(strong, span);
  return item;
}

function asMoment(seconds) {
  return new Date(seconds * 1000).toLocaleString(undefined, {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
}

async function loadInside() {
  const response = await fetch('/api/inside');
  if (!response.ok) return;
  const { inside } = await response.json();

  document.getElementById('inside-count').textContent = inside.length ? `(${inside.length})` : '';
  document.getElementById('inside-empty').hidden = inside.length > 0;
  document.getElementById('inside').replaceChildren(
    ...inside.map((entry) => {
      const item = document.createElement('li');
      const name = document.createElement('button');
      name.type = 'button';
      name.className = 'who';
      name.textContent = entry.name;
      name.addEventListener('click', () => showPerson(entry.name));

      const when = document.createElement('span');
      when.textContent = ` came in ${asMoment(entry.entered)}, ${entry.for} ago`;
      item.append(name, when);

      // The clip for that entry, which is the thing to correct: the exit could not be named
      // because this entry carries the wrong name.
      if (entry.sighting_id) {
        const still = document.createElement('img');
        still.className = 'entry-face';
        still.src = `/media/${entry.sighting_id}.jpg`;
        still.alt = `the entry recorded as ${entry.name}`;
        still.addEventListener('error', () => still.remove());
        item.append(still);
      }
      return item;
    })
  );
}

for (const button of document.querySelectorAll('.window')) {
  button.addEventListener('click', () => {
    window_ = button.dataset.window;
    offset = 0;   // a different period starts at the one running now
    for (const other of document.querySelectorAll('.window')) {
      other.classList.toggle('chosen', other === button);
    }
    load();
  });
}

document.getElementById('earlier').addEventListener('click', () => {
  offset += 1;
  load();
});
document.getElementById('later').addEventListener('click', () => {
  offset = Math.max(0, offset - 1);
  load();
});
document.getElementById('search').addEventListener('input', (event) => {
  filter = event.target.value.trim().toLowerCase();
  load();
});
document.getElementById('close-person').addEventListener('click', () => {
  document.getElementById('person-panel').hidden = true;
});

load();
setInterval(load, 30000);
