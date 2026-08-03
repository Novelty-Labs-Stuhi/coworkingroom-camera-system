// The activity chart: a year of days as boxes, shaded by how much time was spent in the room.
//
// It copies GitHub's contribution graph on purpose. That shape is worth copying because it
// answers "is this a habit" at a glance and needs no legend to be read wrongly -- and because
// everybody who will look at this page has already learnt how to read it.
//
// Wrapped so it exposes exactly one name. It is a widget rather than a page, so the page using
// it must be able to declare whatever variables it likes without colliding with this file.

(function () {
  // Monday first, not Sunday: the office is in Helsinki, and a week here starts on a Monday.
  // The server already aligns the first day of the chart to one, so the rows line up.
  const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  // Only every other row is labelled, as on the chart this copies: seven labels in 77 pixels
  // cannot be read, and three are enough to tell which row is which.
  const LABELLED = [0, 2, 4];
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  // Days are contiguous and start on a Monday, so seven of them are one column.
  function columns(days) {
    const weeks = [];
    for (let at = 0; at < days.length; at += 7) weeks.push(days.slice(at, at + 7));
    return weeks;
  }

  function asDate(day) {
    // Read as parts rather than parsed: `new Date('2026-08-03')` is UTC midnight, which in a
    // negative offset renders as the day before -- the box would carry the wrong month.
    const [year, month, date] = day.date.split('-').map(Number);
    return new Date(year, month - 1, date);
  }

  // The month a column belongs to is the month its first day is in, and a month is labelled
  // once, above the first column that reaches it.
  function months(weeks) {
    const row = document.createElement('div');
    row.className = 'cal-months';
    row.style.setProperty('--weeks', weeks.length);
    let previous = null;
    weeks.forEach((week, column) => {
      if (!week.length) return;
      const month = asDate(week[0]).getMonth();
      if (month === previous) return;
      previous = month;
      // The first column is skipped: it is usually a few days of the previous month, and
      // labelling it puts a month name over days that are not in it.
      if (column === 0) return;
      // A label spans the columns it has room for, never more. Asking for four when two remain
      // makes the grid invent the missing columns, which widens this row past the boxes below
      // and slides every month name off the weeks it is naming. One column is not enough room
      // for a month name, so the last few days of the chart go unlabelled rather than crooked.
      const room = weeks.length - column;
      if (room < 2) return;
      const label = document.createElement('span');
      label.textContent = MONTHS[month];
      label.style.gridColumn = `${column + 1} / span ${Math.min(4, room)}`;
      row.append(label);
    });
    return row;
  }

  function weekdays() {
    const column = document.createElement('div');
    column.className = 'cal-weekdays';
    for (let row = 0; row < WEEKDAYS.length; row += 1) {
      const label = document.createElement('span');
      label.textContent = LABELLED.includes(row) ? WEEKDAYS[row] : '';
      column.append(label);
    }
    return column;
  }

  // What a box means, in words. Every box says it, because a shade on its own is a guess and
  // the difference between "here for ten minutes" and "here all day" is the whole point.
  function saying(day) {
    const when = asDate(day).toLocaleDateString(undefined, {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    });
    if (!day.counted) return `${when} — before counting began`;
    if (!day.seconds) return `${when} — not here`;
    const visits = day.visits === 1 ? '1 visit' : `${day.visits} visits`;
    return `${when} — ${day.readable} over ${visits}`;
  }

  function box(day, onPick) {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = `box l${day.level}${day.counted ? '' : ' uncounted'}`;
    cell.dataset.date = day.date;
    // Both, deliberately: the title is the hover, the label is what a screen reader says, and
    // a box with no text has nothing else to offer either of them.
    cell.title = saying(day);
    cell.setAttribute('aria-label', cell.title);
    cell.addEventListener('click', () => onPick(day));
    return cell;
  }

  function grid(days, onPick) {
    const host = document.createElement('div');
    host.className = 'cal-grid';
    host.style.setProperty('--weeks', Math.ceil(days.length / 7));
    host.append(...days.map((day) => box(day, onPick)));
    return host;
  }

  function legend(shades) {
    const row = document.createElement('div');
    row.className = 'cal-legend';
    const less = document.createElement('span');
    less.textContent = 'Less';
    const more = document.createElement('span');
    more.textContent = 'More';
    row.append(less);
    for (let level = 0; level <= shades; level += 1) {
      const swatch = document.createElement('i');
      swatch.className = `box l${level}`;
      row.append(swatch);
    }
    row.append(more);
    return row;
  }

  // `onPick` is handed the whole day, not just its date: a caller that wants to say "nothing
  // happened" should not have to fetch the day to find that out.
  function drawActivityCalendar(host, chart, onPick) {
    const weeks = columns(chart.days);
    const body = document.createElement('div');
    body.className = 'cal-body';
    body.append(weekdays(), grid(chart.days, onPick));

    const scroller = document.createElement('div');
    scroller.className = 'cal-scroller';
    scroller.append(months(weeks), body);

    host.replaceChildren(scroller, legend(chart.shades));
    // Opened showing today. A year is wider than most windows, and the interesting end of it
    // is the recent one -- landing on last August would look like an empty chart.
    scroller.scrollLeft = scroller.scrollWidth;
  }

  function markChosen(host, iso) {
    for (const cell of host.querySelectorAll('.box')) {
      cell.classList.toggle('chosen', cell.dataset.date === iso);
    }
  }

  window.drawActivityCalendar = drawActivityCalendar;
  window.markChosenDay = markChosen;
})();
