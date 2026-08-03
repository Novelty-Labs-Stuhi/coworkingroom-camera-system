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
  sayWhy(board, rows.length);
  await loadInside();
}

// An empty board has two quite different meanings, and saying which is the whole difference
// between a page that looks broken and one that is waiting.
function sayWhy(board, shown) {
  const note = document.getElementById('board-empty');
  note.hidden = shown > 0;
  if (shown > 0) return;
  if (board.ready) {
    note.textContent = filter
      ? `Nobody matching “${filter}” appeared in this period.`
      : 'Nobody appeared in this period.';
    return;
  }
  const when = new Date(board.ready_at * 1000);
  const started = new Date(board.from * 1000);
  note.textContent =
    started > new Date()
      ? `Counting starts at ${asMoment(board.from)}. Nothing is recorded against these `
        + 'figures until then.'
      : `Counting began at ${asMoment(board.from)}, moments ago. The boards fill in from `
        + `${when.toLocaleTimeString()}.`;
}

function showCovering(board) {
  const note = document.getElementById('covering');
  if (window_ === 'streak') {
    note.textContent = 'days in a row, counted to today';
  } else if (window_ === 'all') {
    note.textContent = 'everything recorded';
  } else if (!board.ready) {
    note.textContent = 'not yet';
  } else {
    // A window the record does not fully cover says so. These are real figures over a shorter
    // period than the tab names, and reading them as a whole week's would be the mistake.
    const span = `${asDay(board.from)} to ${asDay(board.until)}`;
    note.textContent = board.partial
      ? `${span} — ${board.days_covered} of ${board.days_in_window} days so far`
      : span;
  }
  note.classList.toggle('short', Boolean(board.partial));
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

      // The name opens their profile: a board answers "who", a profile answers everything
      // else about one of them. A link rather than a button, so it can be opened in a new tab
      // and so the address of a profile is a thing that can be sent to somebody.
      const name = profileLink(row.name);

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

function profileLink(name) {
  const link = document.createElement('a');
  link.className = 'who';
  link.href = `/profile?name=${encodeURIComponent(name)}`;
  link.textContent = name;
  return link;
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
      const name = profileLink(entry.name);

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

load();
setInterval(load, 30000);
